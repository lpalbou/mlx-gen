"""The one-step chunked DiT driver.

``StreamingDiT`` owns the two pieces of bookkeeping that make a chunked restore a single
continuous clip: the accumulating rotary offset, and the front-padding that keeps a short
LAST chunk at the sequence length every other chunk used. Both are silent when wrong - a
mis-accumulated offset produces a plausible video with a temporal discontinuity, and a
mis-sliced LAST chunk returns the right number of latents taken from the wrong place.

The model here is a real ``SwiftVRTransformer`` built small (2 blocks, 24-dim inner) with
random weights. It is a small model, not a stand-in for one: every assertion is about the
driver's arithmetic and its use of the transformer, never about matching the published
checkpoint, which is tests/swiftvr/parity's job.
"""

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.swiftvr.model.swiftvr_transformer.swiftvr_transformer import SwiftVRTransformer
from mflux.models.swiftvr.streaming.chunk import ChunkSpec, ChunkType, build_chunk_specs
from mflux.models.swiftvr.streaming.streaming_dit import INFERENCE_TIMESTEP, StreamingDiT

LATENT_CHANNELS = 8
LATENT_HW = 8
PROMPT_TOKENS = 6
TEXT_DIM = 16


def _transformer(**overrides) -> SwiftVRTransformer:
    kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": LATENT_CHANNELS,
        "out_channels": LATENT_CHANNELS,
        "text_dim": TEXT_DIM,
        "freq_dim": 16,
        "ffn_dim": 32,
        "num_layers": 2,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 64,
    }
    kwargs.update(overrides)
    mx.random.seed(19)
    transformer = SwiftVRTransformer(window_hw=(2, 2), shift_alternate_layers=True, **kwargs)
    transformer.install_mfswa()
    return transformer


def _latents(frames: int, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, LATENT_CHANNELS, frames, LATENT_HW, LATENT_HW))


def _prompt() -> mx.array:
    mx.random.seed(101)
    return mx.random.normal((1, PROMPT_TOKENS, TEXT_DIM))


def _max_difference(first: mx.array, second: mx.array) -> float:
    mx.eval(first, second)
    return float(mx.max(mx.abs(first - second)))


def _matmul_relative_error() -> float:
    # The float32 matmul precision of the default device, measured against a float64 product. Apple's
    # M1 to M4 GPUs and the CPU round at about 3e-7; an M5 GPU returns float32 products at about 8e-4.
    a = np.random.default_rng(0).standard_normal((64, 96))
    b = np.random.default_rng(1).standard_normal((96, 48))
    exact = np.einsum("ik,kj->ij", a, b)  # float64, without the BLAS path that warns spuriously on this chip
    product = np.array(mx.matmul(mx.array(a, dtype=mx.float32), mx.array(b, dtype=mx.float32)), dtype=np.float64)
    return float(np.max(np.abs(product - exact)) / np.max(np.abs(exact)))


def _offset_tolerance() -> float:
    # A uniform offset cancels exactly in the math, so what is left is rounding: 9e-8 of the output on the CPU,
    # 6.5e-5 on an M5 GPU, where the matmul error is 8e-4 (ratio 0.08). 1e-5 is the bound that held on M1-M4
    # GPUs; four times the device's matmul error keeps the same margin where the device rounds coarser.
    return max(1e-5, 4 * _matmul_relative_error())


class TestOneStepRestoration:
    def test_the_result_is_the_latents_minus_the_predicted_velocity(self):
        """z_hq = z_lq - v(z_lq, t = 1000): one pass, no sampler, no sigma schedule."""
        transformer = _transformer()
        latents = _latents(3)
        velocity = transformer.predict_velocity(
            latents,
            timestep=mx.full((1,), INFERENCE_TIMESTEP, dtype=mx.float32),
            prompt_embeds=_prompt(),
            t_offset=0,
        )
        restored = StreamingDiT(transformer, clear_cache_each_block=False).denoise(latents, _prompt())
        assert _max_difference(restored, latents - velocity) == 0.0

    def test_the_conditioning_timestep_is_the_flow_endpoint(self):
        dit = StreamingDiT(_transformer())
        conditioning = dit._conditioning(1, _prompt(), mx.float32)
        assert float(conditioning.timestep[0]) == INFERENCE_TIMESTEP == 1000.0

    def test_the_timestep_comes_from_the_catalog_through_the_constructor(self):
        """Metadata reports what the run used, so the two must be the same value."""
        dit = StreamingDiT(_transformer(), inference_timestep=750.0)
        assert dit.inference_timestep == 750.0
        assert float(dit._conditioning(1, _prompt(), mx.float32).timestep[0]) == 750.0

    @pytest.mark.parametrize("timestep", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_timestep_that_is_not_a_finite_positive_value_raises(self, timestep):
        with pytest.raises(ValueError, match="finite and positive"):
            StreamingDiT(_transformer(), inference_timestep=timestep)

    def test_the_restore_is_deterministic(self):
        """No seed, no noise: the same clip must restore to the same latents every time."""
        transformer = _transformer()
        first = StreamingDiT(transformer, clear_cache_each_block=False).denoise(_latents(3), _prompt())
        second = StreamingDiT(transformer, clear_cache_each_block=False).denoise(_latents(3), _prompt())
        assert _max_difference(first, second) == 0.0

    def test_running_without_mfswa_installed_raises(self):
        """Wan's global attention here would be a different model that raises nothing."""
        transformer = _transformer()
        transformer.uninstall_mfswa()
        with pytest.raises(RuntimeError, match="requires MFSWA to be installed"):
            StreamingDiT(transformer).denoise(_latents(3), _prompt())


class TestTemporalOffset:
    def test_the_offset_advances_by_the_latent_count_of_each_chunk(self):
        dit = StreamingDiT(_transformer(), clear_cache_each_block=False)
        assert dit.latent_offset == 0
        dit.denoise(_latents(3, seed=1), _prompt())
        assert dit.latent_offset == 3
        dit.denoise(_latents(2, seed=2), _prompt())
        assert dit.latent_offset == 5

    def test_the_offset_reaches_the_rotary_tables_the_transformer_is_given(self):
        """The offset is observable in the tables, which is where it must be checked."""
        transformer = _transformer()
        dit = StreamingDiT(transformer, clear_cache_each_block=False)
        dit.latent_offset = 7
        dit.denoise(_latents(3, seed=3), _prompt())

        expected = transformer.rope_offset(token_grid=(3, 4, 4), t_offset=7)
        supplied = transformer.window_runtime.rotary_emb
        assert _max_difference(supplied[0], expected[0]) == 0.0
        assert _max_difference(supplied[1], expected[1]) == 0.0
        # And the offset is not a no-op on the tables themselves.
        assert _max_difference(expected[0], transformer.rope_offset(token_grid=(3, 4, 4), t_offset=0)[0]) > 1e-2

    def test_a_uniform_offset_does_not_change_a_chunks_output(self):
        """Pinned because it is surprising and it bounds what these tests can catch.

        RoPE encodes RELATIVE position and SwiftVR keeps no KV cache, so a chunk only ever
        attends within itself and a uniform shift of every position cancels. The same is
        true of the reference. A t_offset regression is therefore invisible at the DiT
        output and can only be caught at the table level, in test_rope_offset.py and in
        tests/swiftvr/parity - which is where it is caught.
        """
        transformer = _transformer()
        latents = _latents(3, seed=3)
        first = StreamingDiT(transformer, clear_cache_each_block=False).denoise(latents, _prompt())
        later = StreamingDiT(transformer, clear_cache_each_block=False)
        later.latent_offset = 7
        shifted = later.denoise(latents, _prompt())
        magnitude = float(mx.max(mx.abs(first)))
        assert _max_difference(first, shifted) < _offset_tolerance() * magnitude

    def test_reset_returns_the_driver_to_the_start_of_a_clip(self):
        dit = StreamingDiT(_transformer(), clear_cache_each_block=False)
        dit.denoise(_latents(3), _prompt())
        dit.reset()
        assert dit.latent_offset == 0
        assert dit._previous_lq is None
        assert dit.transformer.window_runtime.token_grid is None

    def test_a_last_chunk_advances_by_its_own_latent_count_not_the_padded_one(self):
        specs = build_chunk_specs(29, 24)
        dit = StreamingDiT(_transformer(), clear_cache_each_block=False)
        dit.denoise(_latents(specs[0].latent_count, seed=4), _prompt())
        assert dit.latent_offset == 7
        dit.denoise_last_chunk(
            _latents(specs[1].latent_count, seed=5),
            specs[1],
            _prompt(),
            previous_latents=None,
            chunk_latent_frames=6,
        )
        assert dit.latent_offset == 8


class TestLastChunkPadding:
    @staticmethod
    def _last_spec(total_frames: int = 33, clip_len: int = 24) -> ChunkSpec:
        spec = build_chunk_specs(total_frames, clip_len)[-1]
        assert spec.ctype is ChunkType.LAST
        return spec

    def test_only_the_new_latents_come_back(self):
        spec = self._last_spec()
        restored = StreamingDiT(_transformer(), clear_cache_each_block=False).denoise_last_chunk(
            _latents(spec.latent_count, seed=6),
            spec,
            _prompt(),
            previous_latents=None,
            chunk_latent_frames=6,
        )
        assert restored.shape == (1, LATENT_CHANNELS, spec.latent_count, LATENT_HW, LATENT_HW)

    def test_the_padded_sequence_length_matches_a_middle_chunk_plus_one(self):
        """Holding the sequence length constant is what keeps the window tables cacheable."""
        spec = self._last_spec()
        transformer = _transformer()
        seen = {}
        original = transformer.token_grid

        def record(latents):
            seen["frames"] = latents.shape[2]
            return original(latents)

        transformer.token_grid = record
        StreamingDiT(transformer, clear_cache_each_block=False).denoise_last_chunk(
            _latents(spec.latent_count, seed=7),
            spec,
            _prompt(),
            previous_latents=None,
            chunk_latent_frames=6,
        )
        assert seen["frames"] == 7

    def test_the_previous_chunk_supplies_the_front_padding(self):
        """The prefix is discarded, but it conditions the kept latents, so a run with a
        real prefix must differ from one padded with zeros."""
        spec = self._last_spec()
        transformer = _transformer()
        latents = _latents(spec.latent_count, seed=8)
        previous = _latents(6, seed=9)

        with_zeros = StreamingDiT(transformer, clear_cache_each_block=False).denoise_last_chunk(
            latents, spec, _prompt(), previous_latents=None, chunk_latent_frames=6
        )
        with_previous = StreamingDiT(transformer, clear_cache_each_block=False).denoise_last_chunk(
            latents, spec, _prompt(), previous_latents=previous, chunk_latent_frames=6
        )
        assert with_zeros.shape == with_previous.shape
        assert _max_difference(with_zeros, with_previous) > 1e-4

    def test_a_last_chunk_that_needs_no_padding_is_left_alone(self):
        spec = self._last_spec(total_frames=53, clip_len=24)
        dit = StreamingDiT(_transformer(), clear_cache_each_block=False)
        restored = dit.denoise_last_chunk(
            _latents(spec.latent_count, seed=10),
            spec,
            _prompt(),
            previous_latents=None,
            chunk_latent_frames=spec.latent_count - 1,
        )
        assert restored.shape[2] == spec.latent_count

    def test_a_non_last_spec_is_refused(self):
        first = build_chunk_specs(101, 24)[0]
        with pytest.raises(ValueError, match="only valid for LAST chunks"):
            StreamingDiT(_transformer()).denoise_last_chunk(
                _latents(7), first, _prompt(), previous_latents=None, chunk_latent_frames=6
            )

    def test_a_latent_count_that_disagrees_with_the_spec_is_refused(self):
        spec = self._last_spec()
        with pytest.raises(ValueError, match="but received"):
            StreamingDiT(_transformer()).denoise_last_chunk(
                _latents(spec.latent_count + 1), spec, _prompt(), previous_latents=None, chunk_latent_frames=6
            )

    def test_a_chunk_longer_than_the_padded_length_is_refused(self):
        spec = self._last_spec()
        with pytest.raises(ValueError, match="chunk_latent_frames must be"):
            StreamingDiT(_transformer()).denoise_last_chunk(
                _latents(spec.latent_count), spec, _prompt(), previous_latents=None, chunk_latent_frames=0
            )

    def test_a_previous_chunk_with_the_wrong_geometry_is_refused(self):
        spec = self._last_spec()
        wrong = mx.zeros((1, LATENT_CHANNELS, 6, LATENT_HW + 2, LATENT_HW))
        with pytest.raises(ValueError, match="but the LAST chunk needs"):
            StreamingDiT(_transformer()).denoise_last_chunk(
                _latents(spec.latent_count), spec, _prompt(), previous_latents=wrong, chunk_latent_frames=6
            )

    def test_a_previous_chunk_too_short_to_pad_with_is_refused(self):
        spec = self._last_spec()
        with pytest.raises(ValueError, match="Carry the preceding chunk"):
            StreamingDiT(_transformer()).denoise_last_chunk(
                _latents(spec.latent_count),
                spec,
                _prompt(),
                previous_latents=_latents(1, seed=11),
                chunk_latent_frames=6,
            )

    def test_last_clears_the_overlap_carry(self):
        """Upstream leaves it, which would feed a two-chunks-old prefix at a stale offset."""
        spec = self._last_spec()
        dit = StreamingDiT(_transformer(), dit_overlap=0, clear_cache_each_block=False)
        dit._previous_lq = _latents(2, seed=12)
        dit.denoise_last_chunk(
            _latents(spec.latent_count, seed=13),
            spec,
            _prompt(),
            previous_latents=None,
            chunk_latent_frames=6,
        )
        assert dit._previous_lq is None


class TestConditioning:
    def test_a_rank_two_prompt_embedding_is_broadcast_to_the_batch(self):
        dit = StreamingDiT(_transformer())
        conditioning = dit._conditioning(2, mx.zeros((PROMPT_TOKENS, TEXT_DIM)), mx.float32)
        assert conditioning.prompt_embeds.shape == (2, PROMPT_TOKENS, TEXT_DIM)

    def test_a_rank_three_prompt_embedding_with_batch_one_is_broadcast(self):
        dit = StreamingDiT(_transformer())
        conditioning = dit._conditioning(3, mx.zeros((1, PROMPT_TOKENS, TEXT_DIM)), mx.float32)
        assert conditioning.prompt_embeds.shape == (3, PROMPT_TOKENS, TEXT_DIM)

    def test_the_conditioning_is_reused_across_chunks(self):
        dit = StreamingDiT(_transformer())
        first = dit._conditioning(1, _prompt(), mx.float32)
        assert dit._conditioning(1, _prompt(), mx.float32) is first

    def test_a_prompt_batch_that_matches_neither_one_nor_the_latents_raises(self):
        dit = StreamingDiT(_transformer())
        with pytest.raises(ValueError, match="neither 1 nor the latent batch"):
            dit._conditioning(3, mx.zeros((2, PROMPT_TOKENS, TEXT_DIM)), mx.float32)

    @pytest.mark.parametrize("shape", [(TEXT_DIM,), (1, 1, PROMPT_TOKENS, TEXT_DIM)])
    def test_a_prompt_embedding_of_the_wrong_rank_raises(self, shape):
        dit = StreamingDiT(_transformer())
        with pytest.raises(ValueError, match=r"\[512, text_dim\]"):
            dit._conditioning(1, mx.zeros(shape), mx.float32)


class TestDriverContract:
    def test_a_negative_overlap_is_refused(self):
        with pytest.raises(ValueError, match="zero or positive"):
            StreamingDiT(_transformer(), dit_overlap=-1)

    @pytest.mark.parametrize("shape", [(1, LATENT_CHANNELS, 3, LATENT_HW), (LATENT_CHANNELS, 3, LATENT_HW, LATENT_HW)])
    def test_latents_that_are_not_rank_five_are_refused(self, shape):
        with pytest.raises(ValueError, match=r"\[B, C, F, H, W\]"):
            StreamingDiT(_transformer()).denoise(mx.zeros(shape), _prompt())

    def test_a_latent_grid_that_the_patch_embed_cannot_divide_is_refused(self):
        with pytest.raises(ValueError, match="not divisible by the transformer"):
            StreamingDiT(_transformer(), clear_cache_each_block=False).denoise(
                mx.zeros((1, LATENT_CHANNELS, 3, 7, LATENT_HW)), _prompt()
            )

    def test_the_overlap_path_keeps_the_emitted_frame_count(self):
        """Overlap extends the chunk backwards for context and slices the prefix away; it
        is not a supported route, but the shape contract must hold if it is exercised."""
        dit = StreamingDiT(_transformer(), dit_overlap=2, clear_cache_each_block=False)
        first = dit.denoise(_latents(3, seed=14), _prompt())
        second = dit.denoise(_latents(3, seed=15), _prompt())
        assert first.shape == second.shape == (1, LATENT_CHANNELS, 3, LATENT_HW, LATENT_HW)
        assert dit.latent_offset == 6
