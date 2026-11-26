from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.prompt_embed_store import WanPromptEmbedStore
from mflux.models.wan.variants.wan_bernini import BerniniRenderer, _MomentumBuffer
from mflux.models.wan.wan_text_encoder_loader import WanTextEncoderLoader
from mflux.utils.video_util import VideoUtil


def _numpy_normalize_diff(
    diff: np.ndarray,
    base: np.ndarray,
    *,
    eta: float,
    norm_threshold: float,
) -> np.ndarray:
    axes = (1, 3, 4) if diff.ndim == 5 else tuple(range(1, diff.ndim))
    if norm_threshold > 0:
        norm = np.sqrt(np.sum(np.square(diff), axis=axes, keepdims=True))
        diff = diff * np.minimum(np.ones_like(diff), norm_threshold / norm)
    diff64 = diff.astype(np.float64)
    base64 = base.astype(np.float64)
    base_norm = np.sqrt(np.sum(np.square(base64), axis=axes, keepdims=True))
    unit_base = base64 / np.maximum(base_norm, np.float64(1e-12))
    parallel64 = np.sum(diff64 * unit_base, axis=axes, keepdims=True) * unit_base
    parallel = parallel64.astype(diff.dtype)
    orthogonal = (diff64 - parallel64).astype(diff.dtype)
    return orthogonal + eta * parallel


def _guidance_model(monkeypatch, predictions):
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()
    calls = []

    def fake_predict(self, *, role, target, condition_segments, source_ids, text_embeds, **kwargs):
        calls.append(
            {
                "role": role,
                "condition_segments": condition_segments,
                "source_ids": source_ids,
                "text_embeds": text_embeds,
            }
        )
        prediction = predictions[role]
        if np.isscalar(prediction):
            return mx.full(target.shape, prediction, dtype=mx.float32)
        return mx.array(prediction, dtype=mx.float32)

    monkeypatch.setattr(BerniniRenderer, "_predict_branch", fake_predict)
    return model, calls


def _branch_kwargs(target, positive, negative):
    return {
        "target": target,
        "prompt_embeds": positive,
        "negative_prompt_embeds": negative,
        "timestep": 500,
        "step_number": 1,
        "total_steps": 2,
        "clear_cache_each_block": False,
        "clear_branch_cache": False,
        "check_tensors": False,
    }


def _numpy_r2v_apg_noise_prediction(
    *,
    target: np.ndarray,
    empty: np.ndarray,
    reference: np.ndarray,
    text: np.ndarray,
    reference_guidance: float,
    text_guidance: float,
    sigma: float,
    eta: float,
    norm_threshold: float,
) -> np.ndarray:
    empty_sample = target - sigma * empty
    reference_sample = target - sigma * reference
    text_sample = target - sigma * text
    guided = empty_sample + reference_guidance * _numpy_normalize_diff(
        reference_sample - empty_sample,
        reference_sample,
        eta=eta,
        norm_threshold=norm_threshold,
    )
    guided = guided + text_guidance * _numpy_normalize_diff(
        text_sample - reference_sample,
        text_sample,
        eta=eta,
        norm_threshold=norm_threshold,
    )
    return (target - guided) / sigma


def _numpy_v2v_apg_noise_prediction(
    *,
    target: np.ndarray,
    uncond: np.ndarray,
    cond: np.ndarray,
    text_guidance: float,
    sigma: float,
    eta: float,
    norm_threshold: float,
) -> np.ndarray:
    uncond_sample = target - sigma * uncond
    cond_sample = target - sigma * cond
    guided = uncond_sample + text_guidance * _numpy_normalize_diff(
        cond_sample - uncond_sample,
        cond_sample,
        eta=eta,
        norm_threshold=norm_threshold,
    )
    return (target - guided) / sigma


def test_bernini_source_ids_match_official_interpolation():
    assert BerniniRenderer._source_ids(0) == []
    assert BerniniRenderer._source_ids(1) == [1.0]
    assert BerniniRenderer._source_ids(5) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert BerniniRenderer._source_ids(8) == pytest.approx(np.linspace(1.0, 5.0, 8).tolist())


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (6, [1.0, 1.7999999523162842, 2.5999999046325684, 3.4000000953674316, 4.199999809265137, 5.0]),
        (
            7,
            [
                1.0,
                1.6666667461395264,
                2.3333334922790527,
                3.0,
                3.6666665077209473,
                4.333333492279053,
                5.0,
            ],
        ),
        (
            8,
            [
                1.0,
                1.5714285373687744,
                2.142857074737549,
                2.7142858505249023,
                3.2857141494750977,
                3.857142925262451,
                4.4285712242126465,
                5.0,
            ],
        ),
    ],
)
def test_bernini_interpolated_source_ids_match_official_torch_float32_exactly(count, expected):
    assert BerniniRenderer._source_ids(count) == expected


def test_bernini_source_id_policy_is_config_driven():
    model = BerniniRenderer.__new__(BerniniRenderer)
    overrides = dict(ModelConfig.bernini_r_1_3b().transformer_overrides)
    overrides.update(max_trained_source_id=3, interpolate_source_ids=True)
    model.model_config = SimpleNamespace(transformer_overrides=overrides)
    assert model._configured_source_ids(2) == [1.0, 2.0]
    assert model._configured_source_ids(5) == pytest.approx(np.linspace(1.0, 3.0, 5).tolist())

    overrides["interpolate_source_ids"] = False
    assert model._configured_source_ids(5) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_bernini_renderer_default_guidance_matches_official_renderer_cli_defaults():
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()

    assert model._resolved_task_or_config_float(
        None,
        task_default=None,
        config_key="default_reference_guidance",
        fallback=4.5,
        label="reference_guidance",
    ) == pytest.approx(4.5)
    assert model._resolved_task_or_config_float(
        None,
        task_default=None,
        config_key="default_source_guidance",
        fallback=1.25,
        label="source_guidance",
    ) == pytest.approx(1.25)
    assert model._resolved_task_or_config_float(
        None,
        task_default=None,
        config_key="default_reference_guidance",
        fallback=4.5,
        label="reference_guidance",
    ) == pytest.approx(4.5)
    assert model._resolved_task_or_config_float(
        2.25,
        task_default=3.0,
        config_key="default_reference_guidance",
        fallback=4.5,
        label="reference_guidance",
    ) == pytest.approx(2.25)


def test_bernini_effective_prompt_matches_official_concatenation():
    model = BerniniRenderer.__new__(BerniniRenderer)

    assert model._effective_prompt(system_prompt="system.", prompt="user prompt") == "system.user prompt"
    assert model._effective_prompt(system_prompt="system.\n", prompt="user prompt") == "system.\nuser prompt"
    assert model._effective_prompt(system_prompt="", prompt="user prompt") == "user prompt"


def test_bernini_trims_renderer_text_embeddings_to_true_token_lengths(monkeypatch):
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.tokenizers = {"wan": object()}
    prompt_embeds = mx.arange(24, dtype=mx.float32).reshape(1, 8, 3)
    negative_prompt_embeds = (mx.arange(24, dtype=mx.float32) + 100).reshape(1, 8, 3)

    monkeypatch.setattr(
        BerniniRenderer,
        "_tokenize_prompts",
        lambda self, *, cleaned, max_sequence_length: {
            "attention_mask": np.array(
                [
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        },
    )

    trimmed_prompt, trimmed_negative = model._trim_bernini_text_embeddings(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt="prompt",
        negative_prompt="",
        max_sequence_length=8,
    )

    assert trimmed_prompt.shape == (1, 3, 3)
    assert trimmed_negative.shape == (1, 2, 3)
    assert np.array_equal(np.asarray(trimmed_prompt), np.asarray(prompt_embeds)[:, :3, :])
    assert np.array_equal(np.asarray(trimmed_negative), np.asarray(negative_prompt_embeds)[:, :2, :])


def test_bernini_prompt_embed_fingerprint_uses_text_encoder_component_root(monkeypatch, tmp_path):
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.root_path = tmp_path / "base"
    model.root_path.mkdir()
    text_root = tmp_path / "bundle"
    (text_root / "text_encoder").mkdir(parents=True)
    model.component_roots = {"text_encoder": text_root}
    model.model_config = ModelConfig.bernini_r_1_3b()
    model._prompt_embed_store = SimpleNamespace(enabled=True)
    model._prompt_embed_fingerprint = None
    calls = {}

    def capture(path):
        calls["path"] = path
        return "fingerprint"

    monkeypatch.setattr(WanPromptEmbedStore, "compute_text_encoder_fingerprint", capture)

    model._prompt_embed_disk_key(
        text_inputs={
            "input_ids": np.array([[1, 2, 3]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int32),
        },
        max_sequence_length=8,
    )

    assert calls["path"] == text_root / "text_encoder"


def test_bernini_text_encoder_load_uses_text_encoder_component_root(monkeypatch, tmp_path):
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.root_path = tmp_path / "base"
    model.root_path.mkdir()
    text_root = tmp_path / "bundle"
    (text_root / "text_encoder").mkdir(parents=True)
    model.component_roots = {"text_encoder": text_root}
    model.model_config = ModelConfig.bernini_r_1_3b()
    model._resident_text_encoder = None
    calls = {}

    class StopLoad(RuntimeError):
        pass

    def fail_after_capture(*, text_encoder_path):
        calls["path"] = text_encoder_path
        raise StopLoad("captured")

    monkeypatch.setattr(WanTextEncoderLoader, "load", fail_after_capture)

    with pytest.raises(StopLoad, match="captured"):
        model._load_t5_prompt_embeds(
            text_inputs={
                "input_ids": np.array([[1, 2, 3]], dtype=np.int32),
                "attention_mask": np.array([[1, 1, 1]], dtype=np.int32),
            },
            max_sequence_length=8,
        )

    assert calls["path"] == text_root / "text_encoder"


def test_bernini_apg_matches_official_per_frame_axes_clipping_and_momentum():
    rng = np.random.default_rng(73)
    first = rng.normal(size=(1, 3, 2, 3, 4)).astype(np.float32)
    second = rng.normal(size=(1, 3, 2, 3, 4)).astype(np.float32)
    base = rng.normal(size=(1, 3, 2, 3, 4)).astype(np.float32)
    momentum = -0.35
    buffer = _MomentumBuffer(momentum)

    BerniniRenderer._normalize_diff(
        mx.array(first),
        mx.array(base),
        momentum_buffer=buffer,
        eta=0.4,
        norm_threshold=1.7,
    )
    actual = BerniniRenderer._normalize_diff(
        mx.array(second),
        mx.array(base),
        momentum_buffer=buffer,
        eta=0.4,
        norm_threshold=1.7,
    )
    expected_running = second + momentum * first
    expected = _numpy_normalize_diff(expected_running, base, eta=0.4, norm_threshold=1.7)

    assert np.allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
    # C,H,W reduce independently for each target time slice, exactly [-4,-2,-1]
    # in the official five-dimensional tensor.
    assert np.asarray(actual).shape == (1, 3, 2, 3, 4)


@pytest.mark.parametrize("base_scale", [1.0, 1e-14, 1e20])
def test_bernini_apg_stable_projection_matches_official_float64_path(base_scale):
    rng = np.random.default_rng(8104)
    diff = rng.normal(size=(1, 3, 2, 2, 2)).astype(np.float32)
    base = (rng.normal(size=(1, 3, 2, 2, 2)) * base_scale).astype(np.float32)

    actual = BerniniRenderer._normalize_diff(
        mx.array(diff),
        mx.array(base),
        momentum_buffer=None,
        eta=0.35,
        norm_threshold=0.0,
    )
    expected = _numpy_normalize_diff(diff, base, eta=0.35, norm_threshold=0.0)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-7, atol=5e-7)
    assert np.isfinite(np.asarray(actual)).all()


def test_bernini_selects_pinned_diffusers_0_35_2_unipc_grid():
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()

    scheduler = model._create_scheduler(flow_shift=5.0, solver="unipc")
    scheduler.set_timesteps(3)

    assert scheduler.flow_sigma_schedule == "diffusers-0.35.2"
    assert np.array(scheduler.timesteps).tolist() == [999, 908, 713]
    np.testing.assert_array_equal(
        np.array(scheduler.sigmas),
        np.array([0.99979985, 0.9088428, 0.7139794, 0.0], dtype=np.float32),
    )


def test_bernini_r2v_uses_empty_reference_reference_text_chained_apg(monkeypatch):
    target_np = np.linspace(-1.25, 1.75, 16, dtype=np.float32).reshape(1, 2, 2, 2, 2)
    predictions = {
        "r2v-empty": np.full(target_np.shape, -0.75, dtype=np.float32),
        "r2v-references": np.full(target_np.shape, 0.5, dtype=np.float32),
        "r2v-references-text": np.full(target_np.shape, 1.75, dtype=np.float32),
    }
    model, calls = _guidance_model(monkeypatch, predictions)
    target = mx.array(target_np, dtype=mx.float32)
    references = [
        mx.zeros((1, 2, 1, 2, 4)),
        mx.zeros((1, 2, 1, 4, 2)),
    ]
    positive = mx.ones((1, 4, 3))
    negative = mx.zeros((1, 4, 3))

    actual = model._r2v_noise_prediction(
        reference_conditions=references,
        reference_guidance=2.5,
        text_guidance=3.0,
        sigma=mx.array(0.5),
        buffers=[_MomentumBuffer(0.0), _MomentumBuffer(0.0)],
        eta=1.0,
        norm_threshold=0.0,
        **_branch_kwargs(target, positive, negative),
    )
    expected = _numpy_r2v_apg_noise_prediction(
        target=target_np,
        empty=predictions["r2v-empty"],
        reference=predictions["r2v-references"],
        text=predictions["r2v-references-text"],
        reference_guidance=2.5,
        text_guidance=3.0,
        sigma=0.5,
        eta=1.0,
        norm_threshold=0.0,
    )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
    assert [call["role"] for call in calls] == [
        "r2v-empty",
        "r2v-references",
        "r2v-references-text",
    ]
    assert calls[0]["condition_segments"] == []
    assert calls[1]["condition_segments"] == references
    assert calls[1]["source_ids"] == [1.0, 2.0]
    assert calls[1]["text_embeds"] is negative
    assert calls[2]["text_embeds"] is positive


def test_bernini_rv2v_matches_official_four_branch_additive_cfg(monkeypatch):
    predictions = {
        "rv2v-empty": 0.5,
        "rv2v-video": 1.5,
        "rv2v-video-references": 3.0,
        "rv2v-video-references-text": 5.0,
    }
    model, calls = _guidance_model(monkeypatch, predictions)
    target = mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)
    video = mx.zeros((1, 2, 3, 4, 4))
    references = [mx.zeros((1, 2, 1, 2 + index, 4)) for index in range(6)]
    positive = mx.ones((1, 4, 3))
    negative = mx.zeros((1, 4, 3))

    actual = model._rv2v_noise_prediction(
        video_condition=video,
        reference_conditions=references,
        source_guidance=1.25,
        reference_guidance=2.0,
        text_guidance=4.0,
        **_branch_kwargs(target, positive, negative),
    )
    expected = 0.5 + 1.25 * (1.5 - 0.5) + 2.0 * (3.0 - 1.5) + 4.0 * (5.0 - 3.0)

    assert np.asarray(actual) == pytest.approx(expected)
    assert [call["role"] for call in calls] == [
        "rv2v-empty",
        "rv2v-video",
        "rv2v-video-references",
        "rv2v-video-references-text",
    ]
    assert calls[0]["source_ids"] == []
    assert calls[1]["source_ids"] == [1.0]
    expected_combined_ids = np.linspace(1.0, 5.0, 7).tolist()
    assert calls[2]["source_ids"] == pytest.approx(expected_combined_ids)
    assert calls[3]["source_ids"] == pytest.approx(expected_combined_ids)
    assert calls[2]["condition_segments"] == [video, *references]


def test_bernini_v2v_apg_keeps_video_in_both_text_branches(monkeypatch):
    target_np = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(1, 2, 2, 2, 2)
    predictions = {
        "v2v-empty-text": np.full(target_np.shape, -1.5, dtype=np.float32),
        "v2v-video-text": np.full(target_np.shape, 0.75, dtype=np.float32),
    }
    model, calls = _guidance_model(monkeypatch, predictions)
    target = mx.array(target_np, dtype=mx.float32)
    video = mx.zeros((1, 2, 3, 4, 4))
    positive = mx.ones((1, 4, 3))
    negative = mx.zeros((1, 4, 3))

    actual = model._v2v_noise_prediction(
        video_condition=video,
        text_guidance=3.0,
        sigma=mx.array(0.4),
        buffer=_MomentumBuffer(0.0),
        eta=1.0,
        norm_threshold=0.0,
        **_branch_kwargs(target, positive, negative),
    )

    expected = _numpy_v2v_apg_noise_prediction(
        target=target_np,
        uncond=predictions["v2v-empty-text"],
        cond=predictions["v2v-video-text"],
        text_guidance=3.0,
        sigma=0.4,
        eta=1.0,
        norm_threshold=0.0,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
    assert [call["role"] for call in calls] == ["v2v-empty-text", "v2v-video-text"]
    for call in calls:
        assert call["condition_segments"] == [video]
        assert call["source_ids"] == [1.0]
    assert calls[0]["text_embeds"] is negative
    assert calls[1]["text_embeds"] is positive


def test_bernini_single_condition_apg_reuses_prepared_packed_branch(monkeypatch):
    target_np = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(1, 2, 2, 2, 2)
    predictions = {
        "apg-uncond": np.full(target_np.shape, -1.5, dtype=np.float32),
        "apg-cond": np.full(target_np.shape, 0.75, dtype=np.float32),
    }
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()
    target = mx.array(target_np, dtype=mx.float32)
    video = mx.zeros((1, 2, 3, 4, 4), dtype=mx.float32)
    positive = mx.ones((1, 4, 3), dtype=mx.float32)
    negative = mx.zeros((1, 4, 3), dtype=mx.float32)
    prepared = SimpleNamespace(target_shape=tuple(int(value) for value in target.shape))
    prepare_calls = []
    branch_calls = []

    class FakeTransformer:
        def prepare_packed_segments(self, **kwargs):
            prepare_calls.append(kwargs)
            return prepared

    model.transformer = FakeTransformer()

    def fake_predict(self, *, role, prepared, text_embeds, **kwargs):
        branch_calls.append({"role": role, "prepared": prepared, "text_embeds": text_embeds})
        return mx.array(predictions[role], dtype=mx.float32)

    monkeypatch.setattr(BerniniRenderer, "_predict_prepacked_branch", fake_predict)

    actual = model._single_condition_apg_noise_prediction(
        target=target,
        condition_segments=[video],
        prompt_embeds=positive,
        negative_prompt_embeds=negative,
        text_guidance=3.0,
        sigma=mx.array(0.4),
        buffer=_MomentumBuffer(0.0),
        eta=1.0,
        norm_threshold=0.0,
        timestep=500,
        step_number=1,
        total_steps=2,
        clear_cache_each_block=False,
        clear_branch_cache=False,
        check_tensors=False,
    )

    expected = _numpy_v2v_apg_noise_prediction(
        target=target_np,
        uncond=predictions["apg-uncond"],
        cond=predictions["apg-cond"],
        text_guidance=3.0,
        sigma=0.4,
        eta=1.0,
        norm_threshold=0.0,
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["source_ids"] == [1.0, 0.0]
    assert len(prepare_calls[0]["latent_segments"]) == 2
    assert [call["role"] for call in branch_calls] == ["apg-uncond", "apg-cond"]
    assert all(call["prepared"] is prepared for call in branch_calls)
    assert branch_calls[0]["text_embeds"] is negative
    assert branch_calls[1]["text_embeds"] is positive


def test_bernini_predict_branch_orders_video_refs_then_target_id_zero(monkeypatch):
    model = BerniniRenderer.__new__(BerniniRenderer)
    observed = {}

    class FakeTransformer:
        def forward_packed(self, **kwargs):
            observed.update(kwargs)
            return mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)

    model.transformer = FakeTransformer()
    monkeypatch.setattr(BerniniRenderer, "_require_tensor_health", lambda *args, **kwargs: None)
    video = mx.full((1, 2, 3, 4, 4), 1.0, dtype=mx.float32)
    reference = mx.full((1, 2, 1, 2, 4), 2.0, dtype=mx.float32)
    target = mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)

    model._predict_branch(
        role="test",
        target=target,
        condition_segments=[video, reference],
        source_ids=[1.0, 2.0],
        text_embeds=mx.zeros((1, 4, 3)),
        timestep=500,
        step_number=1,
        total_steps=1,
        clear_cache_each_block=False,
        clear_branch_cache=False,
        check_tensors=False,
    )

    assert observed["source_ids"] == [1.0, 2.0, 0.0]
    assert observed["target_segment_index"] == -1
    segments = observed["latent_segments"]
    assert [float(np.asarray(segment.astype(mx.float32)).reshape(-1)[0]) for segment in segments] == [1.0, 2.0, 0.0]
    assert len({segment.shape for segment in segments}) == 3
    assert len({segment.dtype for segment in segments}) == 1
    assert segments[0].dtype == ModelConfig.precision


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (3, [0, 0, 0, 0, 0]),
        (82, list(range(81))),
    ],
)
def test_bernini_smart_video_sampling_matches_official_short_and_same_fps(count, expected):
    fps = 30.0
    actual = BerniniRenderer._smart_video_indices(
        total_frames=count,
        video_fps=fps,
        fps=fps,
        frame_factor=4,
        max_frames=81,
        add_one=True,
    )
    assert actual == expected


def test_bernini_smart_video_sampling_truncates_at_official_4n_plus_one_boundary():
    actual = BerniniRenderer._smart_video_indices(
        total_frames=300,
        video_fps=30.0,
        fps=16.0,
        frame_factor=4,
        max_frames=81,
        add_one=True,
    )
    assert len(actual) == 81
    assert actual[:4] == [0, 2, 4, 6]
    assert actual[-1] == 150


def test_bernini_smart_video_sampling_clamps_official_add_one_index_like_decord():
    actual = BerniniRenderer._smart_video_indices(
        total_frames=100,
        video_fps=16.0,
        fps=16.0,
        frame_factor=4,
        max_frames=121,
        add_one=True,
    )
    assert len(actual) == 101
    assert actual[-2:] == [99, 99]


@pytest.mark.parametrize(
    ("total_frames", "video_fps", "expected"),
    [
        (
            59,
            30.0,
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                17,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
                33,
                35,
                37,
                39,
                41,
                44,
                46,
                48,
                50,
                52,
                54,
                56,
                58,
            ],
        ),
        (
            83,
            60.0,
            [0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 45, 49, 53, 57, 62, 66, 70, 74, 78, 82],
        ),
    ],
)
def test_bernini_smart_video_sampling_matches_official_torch_float32_half_boundaries(
    total_frames,
    video_fps,
    expected,
):
    actual = BerniniRenderer._smart_video_indices(
        total_frames=total_frames,
        video_fps=video_fps,
        fps=16.0,
        frame_factor=4,
        max_frames=81,
        add_one=True,
    )

    assert actual == expected


def test_bernini_condition_geometry_matches_official_stride_resizer():
    assert BerniniRenderer._condition_dimensions(width=1200, height=600, max_size=848) == (848, 416)
    assert BerniniRenderer._condition_dimensions(width=301, height=799, max_size=848) == (304, 800)
    assert BerniniRenderer._condition_dimensions(width=8, height=9, max_size=848) == (16, 16)


def test_bernini_condition_plan_resolves_incomplete_probe_metadata_once(monkeypatch, tmp_path):
    source = tmp_path / "incomplete-probe.mp4"
    source.touch()
    frames = [Image.new("RGB", (64, 48), color) for color in ("red", "green", "blue")]
    incomplete = SimpleNamespace(
        source_frame_count=3,
        source_width=None,
        source_height=None,
        fps=None,
    )
    monkeypatch.setattr(VideoUtil, "inspect_video", lambda path: incomplete)
    monkeypatch.setattr(
        VideoUtil,
        "read_video_clip",
        lambda path: SimpleNamespace(frames=frames, fps=8.0),
    )
    monkeypatch.setattr(
        BerniniRenderer,
        "_read_indexed_video_frames",
        staticmethod(lambda path, indices: [frames[index] for index in indices]),
    )
    model = BerniniRenderer.__new__(BerniniRenderer)

    plan = model._plan_condition_metadata(
        video_path=source,
        requested_height=48,
        requested_width=64,
        requested_frames=81,
        fps=8,
        canvas_policy="exact-resize",
        max_condition_size=848,
    )
    pixels, metadata = model._preprocess_video_condition(
        video_path=source,
        requested_height=48,
        requested_width=64,
        requested_frames=81,
        fps=8,
        canvas_policy="exact-resize",
        resize_mode="resize",
        max_condition_size=848,
        condition_plan=plan,
    )

    assert plan["source_width"] == 64
    assert plan["source_height"] == 48
    assert plan["source_fps"] == 8.0
    assert plan["source_sample_indices"] == [0, 0, 0, 0, 0]
    assert pixels.shape == (1, 3, 5, 48, 64)
    assert metadata["source_width"] == 64
    assert metadata["output_frames"] == 5


def test_bernini_source_aspect_plan_matches_official_long_edge_cap(monkeypatch, tmp_path):
    source = tmp_path / "portrait.mp4"
    source.touch()
    monkeypatch.setattr(
        VideoUtil,
        "inspect_video",
        lambda path: SimpleNamespace(
            source_frame_count=17,
            source_width=480,
            source_height=848,
            fps=16.0,
        ),
    )
    model = BerniniRenderer.__new__(BerniniRenderer)

    small = model._plan_condition_metadata(
        video_path=source,
        requested_height=64,
        requested_width=64,
        requested_frames=81,
        fps=16,
        canvas_policy="source-aspect",
        max_condition_size=1280,
    )
    capped = model._plan_condition_metadata(
        video_path=source,
        requested_height=848,
        requested_width=480,
        requested_frames=81,
        fps=16,
        canvas_policy="source-aspect",
        max_condition_size=320,
    )

    # Official renderer rule: the source video's long edge is capped at
    # max_condition_size and never upscaled; requested width/height do not
    # shrink or override a video-driven canvas.
    assert (small["output_width"], small["output_height"]) == (480, 848)
    assert (small["video_condition_width"], small["video_condition_height"]) == (480, 848)
    assert (small["requested_output_width"], small["requested_output_height"]) == (64, 64)
    assert (capped["output_width"], capped["output_height"]) == (176, 320)
    assert (capped["video_condition_width"], capped["video_condition_height"]) == (176, 320)


def test_bernini_reference_images_are_preprocessed_and_vae_encoded_independently(monkeypatch, tmp_path):
    first = tmp_path / "wide.png"
    second = tmp_path / "tall.png"
    Image.new("RGB", (640, 320), "red").save(first)
    Image.new("RGB", (301, 799), "blue").save(second)
    seen_shapes = []

    def fake_encode(pixels, *, clear_cache_each_slice=False, tile_spatial=False):
        assert clear_cache_each_slice is False
        assert tile_spatial is False
        seen_shapes.append(tuple(pixels.shape))
        return mx.zeros((1, 16, 1, pixels.shape[3] // 8, pixels.shape[4] // 8))

    model = BerniniRenderer.__new__(BerniniRenderer)
    model.vae = SimpleNamespace(encode_normalized=fake_encode)
    monkeypatch.setattr(BerniniRenderer, "_require_tensor_health", lambda *args, **kwargs: None)

    video, reference_videos, references, metadata = model._prepare_condition_latents(
        image_path=None,
        video_path=None,
        reference_video_paths=[],
        reference_image_paths=[first, second],
        requested_height=48,
        requested_width=64,
        requested_frames=5,
        fps=8,
        canvas_policy="exact-resize",
        resize_mode="resize",
        max_condition_size=848,
        clear_cache=False,
    )

    assert video is None
    assert reference_videos == []
    assert seen_shapes == [(1, 3, 1, 320, 640), (1, 3, 1, 800, 304)]
    assert [tuple(reference.shape) for reference in references] == [
        (1, 16, 1, 40, 80),
        (1, 16, 1, 100, 38),
    ]
    assert metadata["reference_pixel_shapes"] == [
        [1, 3, 1, 320, 640],
        [1, 3, 1, 800, 304],
    ]


def test_bernini_low_ram_condition_encode_routes_compaction_and_spatial_tiling(monkeypatch):
    observed = []

    def fake_encode(pixels, *, clear_cache_each_slice=False, tile_spatial=False):
        observed.append(
            {
                "shape": tuple(pixels.shape),
                "clear_cache_each_slice": clear_cache_each_slice,
                "tile_spatial": tile_spatial,
            }
        )
        return mx.zeros((1, 16, 1, pixels.shape[3] // 8, pixels.shape[4] // 8))

    model = BerniniRenderer.__new__(BerniniRenderer)
    model.vae = SimpleNamespace(encode_normalized=fake_encode)
    monkeypatch.setattr(BerniniRenderer, "_require_tensor_health", lambda *args, **kwargs: None)

    latents = model._encode_condition_pixels(
        mx.zeros((1, 3, 1, 480, 848), dtype=mx.float32),
        name="source_video",
        clear_cache=True,
    )
    mx.eval(latents)

    assert latents.shape == (1, 16, 1, 60, 106)
    assert observed == [
        {
            "shape": (1, 3, 1, 480, 848),
            "clear_cache_each_slice": True,
            "tile_spatial": True,
        }
    ]


def test_bernini_promotes_vae_weights_to_float32():
    params = {
        "conv": {
            "weight": mx.ones((2,), dtype=mx.bfloat16),
            "bias": mx.ones((2,), dtype=mx.bfloat16),
        },
        "norm": {
            "weight": mx.ones((2,), dtype=mx.float32),
        },
    }

    class FakeVae:
        def __init__(self):
            self.updated = None

        def parameters(self):
            return params

        def update(self, new_params):
            self.updated = new_params

    model = BerniniRenderer.__new__(BerniniRenderer)
    model.vae = FakeVae()

    model._promote_vae_to_float32()

    assert model.vae.updated["conv"]["weight"].dtype == mx.float32
    assert model.vae.updated["conv"]["bias"].dtype == mx.float32
    assert model.vae.updated["norm"]["weight"].dtype == mx.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"guidance_2": 3.0}, "guidance_2"),
        ({"denoising_step_list": [900, 500]}, "denoising_step_list"),
        ({"image_path": "first.png"}, "exactly one source image"),
        ({"last_image_path": "last.png"}, "last_image_path"),
        ({"context_image_paths": ["context.png"]}, "context_image_paths"),
        ({"context_noise": 10.0}, "context_noise"),
        ({"svi_anchor_image_path": "anchor.png"}, "SVI"),
        ({"svi_motion_latent_count": 2}, "SVI"),
        ({"video_strength": 0.75}, "never an SDEdit warm start"),
        ({"video_mask_path": "mask.png"}, "video_mask_path"),
        ({"release_inactive_denoiser": True}, "one renderer transformer"),
        ({"compile_transformer": True}, "compile_transformer"),
    ],
)
def test_bernini_rejects_shared_wan_options_before_model_or_tensor_access(kwargs, message):
    model = BerniniRenderer.__new__(BerniniRenderer)
    with pytest.raises(ValueError, match=message):
        model.generate_video(seed=1, prompt="x", reference_image_paths=["missing.png"], **kwargs)


def test_bernini_missing_reference_paths_still_fail_closed_before_model_access():
    model = BerniniRenderer.__new__(BerniniRenderer)
    with pytest.raises(ValueError, match="does not exist"):
        model.generate_video(seed=1, prompt="x", reference_image_paths=["missing.png"])


def test_bernini_text_only_modes_are_now_valid_backend_shapes():
    assert BerniniRenderer._resolved_guidance_mode(
        guidance_mode=None,
        task_type=None,
        has_video=False,
        has_image=False,
        num_reference_images=0,
        num_reference_videos=0,
    ) == "t2v_apg"
    assert BerniniRenderer._resolved_task_type(
        task_type=None,
        guidance_mode="t2v_apg",
        has_video=False,
        has_image=False,
        num_reference_images=0,
        num_reference_videos=0,
    ) == "t2v"


def test_bernini_ads2v_defaults_to_public_renderer_rv2v_mode():
    assert BerniniRenderer._resolved_guidance_mode(
        guidance_mode=None,
        task_type="ads2v",
        has_video=True,
        has_image=False,
        num_reference_images=0,
        num_reference_videos=1,
    ) == "rv2v"
    assert BerniniRenderer._resolved_guidance_mode(
        guidance_mode=None,
        task_type=None,
        has_video=True,
        has_image=False,
        num_reference_images=0,
        num_reference_videos=1,
    ) == "rv2v"
    assert BerniniRenderer._resolved_task_type(
        task_type=None,
        guidance_mode="rv2v",
        has_video=True,
        has_image=False,
        num_reference_images=0,
        num_reference_videos=1,
    ) == "ads2v"


def test_bernini_r2v_rejects_source_aspect_without_model_access(tmp_path):
    reference = tmp_path / "reference.png"
    reference.touch()
    model = BerniniRenderer.__new__(BerniniRenderer)

    with pytest.raises(ValueError, match="has no source canvas"):
        model.generate_video(
            seed=1,
            prompt="x",
            reference_image_paths=[reference],
            canvas_policy="source-aspect",
        )


def test_bernini_caps_references_and_condition_size_before_model_access(tmp_path):
    paths = []
    for index in range(9):
        path = tmp_path / f"reference-{index}.png"
        path.touch()
        paths.append(path)
    model = BerniniRenderer.__new__(BerniniRenderer)
    with pytest.raises(ValueError, match="at most 8"):
        model.generate_video(seed=1, prompt="x", reference_image_paths=paths)
    with pytest.raises(ValueError, match="1280"):
        model.generate_video(
            seed=1,
            prompt="x",
            reference_image_paths=paths[:1],
            max_condition_size=1281,
        )
    with pytest.raises(ValueError, match="multiple of 16"):
        model.generate_video(
            seed=1,
            prompt="x",
            reference_image_paths=paths[:1],
            max_condition_size=31,
        )


@pytest.mark.parametrize("max_sequence_length", [0, 256, 513])
def test_bernini_requires_the_official_512_token_contract_before_model_access(tmp_path, max_sequence_length):
    reference = tmp_path / "reference.png"
    reference.touch()
    model = BerniniRenderer.__new__(BerniniRenderer)

    with pytest.raises(ValueError, match="max_sequence_length=512"):
        model.generate_video(
            seed=1,
            prompt="x",
            reference_image_paths=[reference],
            max_sequence_length=max_sequence_length,
        )


@pytest.mark.parametrize(
    ("num_frames", "num_inference_steps"),
    [
        (1, 1),
        (17, 12),
        (24, 12),
        (25, 40),
        (33, 20),
    ],
)
def test_bernini_supported_frame_step_domain_preserves_debug_and_proven_converged_runs(
    num_frames,
    num_inference_steps,
):
    BerniniRenderer._validate_supported_frame_step_domain(
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    )


@pytest.mark.parametrize(
    ("num_frames", "num_inference_steps"),
    [
        (17, 13),
        (17, 20),
        (17, 40),
        (24, 13),
        (24, 20),
        (24, 40),
        (21, 40),
    ],
)
def test_bernini_supported_frame_step_domain_rejects_unproven_short_converged_runs(
    num_frames,
    num_inference_steps,
):
    with pytest.raises(
        ValueError,
        match=rf"resolved to {num_frames} effective frames for {num_inference_steps} inference steps",
    ):
        BerniniRenderer._validate_supported_frame_step_domain(
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        )


def test_bernini_validates_effective_video_frames_before_prompt_encoding(monkeypatch, tmp_path):
    source = tmp_path / "short-source.mp4"
    source.touch()
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()
    model.transformer_2 = None

    class FakeTransformer:
        patch_size = (1, 2, 2)
        vace_layers = None

        def forward_packed(self, **kwargs):
            raise AssertionError("the unsupported request must fail before denoising")

    class FakeVae:
        spatial_scale = 8
        temporal_scale = 4

    class FakeScheduler:
        def set_timesteps(self, steps):
            self.timesteps = np.arange(steps, dtype=np.float32)

    model.transformer = FakeTransformer()
    model.vae = FakeVae()
    monkeypatch.setattr(BerniniRenderer, "_create_scheduler", lambda *args, **kwargs: FakeScheduler())
    monkeypatch.setattr(
        BerniniRenderer,
        "_plan_condition_metadata",
        lambda self, **kwargs: {"output_width": 64, "output_height": 48, "output_frames": 17},
    )
    monkeypatch.setattr(
        BerniniRenderer,
        "encode_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the unsupported request must fail before prompt encoding")
        ),
    )

    with pytest.raises(
        ValueError,
        match="resolved to 17 effective frames for 20 inference steps",
    ):
        model.generate_video(
            seed=1,
            prompt="make the subject turn",
            video_path=source,
            width=64,
            height=48,
            num_frames=81,
            num_inference_steps=20,
            fps=16,
            canvas_policy="source-aspect",
        )


def test_bernini_metadata_records_role_truth_and_no_warm_start(tmp_path):
    reference = tmp_path / "reference.png"
    metadata = BerniniRenderer._bernini_extra_metadata(
        guidance_mode="rv2v",
        reference_image_paths=[reference],
        reference_video_paths=[],
        text_guidance=4.0,
        reference_guidance=3.0,
        source_guidance=2.5,
        apg_eta=0.8,
        apg_norm_threshold=40.0,
        apg_momentum=-0.25,
        max_condition_size=848,
        system_prompt="You are a helpful assistant specialized in video editing with reference.",
        effective_prompt="You are a helpful assistant specialized in video editing with reference. Replace the coat.",
        unipc_flow_sigma_schedule="diffusers-0.35.2",
        source_ids=[1.0, 2.0],
        condition_shapes=[[1, 16, 3, 60, 104], [1, 16, 1, 72, 48]],
        condition_metadata={"source_sample_indices": [0, 2, 4, 6, 8]},
        component_source_provenance={"transformer": {"revision": "abc123"}},
        factored_component_sources=True,
        vae_low_memory_policy_active=True,
        clear_cache_each_transformer_block=True,
        release_denoisers_before_decode=True,
        task_type="rv2v",
    )

    assert metadata["bernini_guidance_mode"] == "rv2v"
    assert metadata["mlx_version"] != "unknown"
    assert metadata["python_version"]
    assert metadata["python_implementation"]
    assert metadata["runtime_platform"]
    assert metadata["numpy_version"]
    assert metadata["python_executable"]
    assert metadata["text_encoder_precision_policy_id"] == "bernini-umt5-official-v2"
    assert metadata["transformer_precision_policy_id"] == "bernini-transformer-official-keep-set-v5"
    assert metadata["transformer_default_weight_precision"] == "bfloat16"
    assert metadata["transformer_fp32_weight_keys"] == ["scale_shift_table"]
    assert metadata["transformer_fp32_weight_prefixes"] == ["condition_embedder.time_embedder."]
    assert metadata["transformer_fp32_weight_fragments"] == [".scale_shift_table", ".norm1.", ".norm2.", ".norm3."]
    assert metadata["source_conditioning"] == "independent-vae-packed-segments"
    assert metadata["source_video_warm_start"] is False
    assert metadata["branch_evaluation"] == "sequential"
    assert metadata["reference_image_paths"] == [str(reference)]
    assert metadata["reference_video_paths"] == []
    assert metadata["bernini_task_type"] == "rv2v"
    assert metadata["condition_source_ids"] == [1.0, 2.0]
    assert metadata["active_guidance_parameters"] == [
        "text_guidance",
        "reference_guidance",
        "source_guidance",
    ]
    assert metadata["inactive_guidance_parameters"] == [
        "apg_eta",
        "apg_norm_threshold",
        "apg_momentum",
    ]
    assert metadata["apg_reduction_axes"] == [1, 3, 4]
    assert metadata["apg_accumulator_precision"] == "float64-projection"
    assert metadata["apg_reference_accumulator_precision"] == "float64"
    assert metadata["system_prompt"] == ("You are a helpful assistant specialized in video editing with reference.")
    assert metadata["effective_prompt"].endswith("Replace the coat.")
    assert metadata["unipc_flow_sigma_schedule"] == "diffusers-0.35.2"
    assert metadata["component_source_provenance"] == {"transformer": {"revision": "abc123"}}
    assert metadata["factored_component_sources"] is True
    assert metadata["low_ram"] is True
    assert metadata["vae_low_memory_policy_active"] is True
    assert metadata["clear_cache_each_transformer_block"] is True
    assert metadata["release_denoisers_before_decode"] is True
    assert metadata["vae_feature_cache_policy_id"] == "wan-compact-feature-cache-v1"
    assert metadata["vae_encode_cache_materialization"] == "eager-contiguous-per-slice"
    assert metadata["vae_decode_cache_materialization"] == "eager-contiguous-per-slice"
    assert metadata["vae_spatial_tiling"] is True
    assert metadata["vae_spatial_tiling_policy_id"] == "wan21-diffusers-0.35.2-256x256-stride192-v1"
    assert metadata["wan_decode_mode"] == "bounded_tile_major_spatial_vae"


def test_bernini_metadata_marks_current_step_clear_low_ram_policy_truthfully(tmp_path):
    reference = tmp_path / "reference.png"
    metadata = BerniniRenderer._bernini_extra_metadata(
        guidance_mode="r2v_apg",
        reference_image_paths=[reference],
        reference_video_paths=[],
        text_guidance=4.0,
        reference_guidance=4.5,
        source_guidance=1.25,
        apg_eta=0.5,
        apg_norm_threshold=50.0,
        apg_momentum=0.0,
        max_condition_size=848,
        system_prompt="You are a helpful assistant specialized in subject-to-video generation.",
        effective_prompt="You are a helpful assistant specialized in subject-to-video generation. Make the statue sway.",
        unipc_flow_sigma_schedule="diffusers-0.35.2",
        source_ids=[1.0],
        condition_shapes=[[1, 16, 1, 80, 106]],
        condition_metadata={},
        vae_low_memory_policy_active=True,
        clear_cache_each_transformer_block=False,
        release_denoisers_before_decode=True,
        task_type="r2v",
    )

    assert metadata["low_ram"] is True
    assert metadata["vae_low_memory_policy_active"] is True
    assert metadata["clear_cache_each_transformer_block"] is False
    assert metadata["release_denoisers_before_decode"] is True


@pytest.mark.parametrize(
    ("guidance_mode", "active", "inactive"),
    [
        (
            "r2v_apg",
            ["text_guidance", "reference_guidance", "apg_eta", "apg_norm_threshold", "apg_momentum"],
            ["source_guidance"],
        ),
        (
            "rv2v",
            ["text_guidance", "reference_guidance", "source_guidance"],
            ["apg_eta", "apg_norm_threshold", "apg_momentum"],
        ),
        (
            "v2v_apg",
            ["text_guidance", "apg_eta", "apg_norm_threshold", "apg_momentum"],
            ["reference_guidance", "source_guidance"],
        ),
    ],
)
def test_bernini_guidance_parameter_activity_is_mode_truthful(guidance_mode, active, inactive):
    assert BerniniRenderer._guidance_parameter_activity(guidance_mode) == (active, inactive)


def test_bernini_low_ram_lifecycle_flushes_branches_and_defers_decode(monkeypatch, tmp_path):
    reference = tmp_path / "reference.png"
    reference.touch()
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()
    model.transformer_2 = None
    model.bits = 4
    events = []
    condition_cache_flags = []
    step_cache_flags = []
    decode_observations = []

    class FakeTransformer:
        patch_size = (1, 2, 2)
        vace_layers = None

        def __init__(self):
            self.calls = []

        def forward_packed(self, **kwargs):
            self.calls.append(kwargs)
            events.append("forward")
            return mx.ones_like(kwargs["latent_segments"][-1])

    class FakeVae:
        spatial_scale = 8
        temporal_scale = 4
        z_dim = 16

        def iter_decode_normalized_latent_slices(
            self,
            latents,
            *,
            clear_cache_each_slice=False,
            tile_spatial=False,
        ):
            decode_observations.append(
                {
                    "latents_shape": tuple(latents.shape),
                    "transformer": model.transformer,
                    "transformer_2": model.transformer_2,
                    "clear_cache_each_slice": clear_cache_each_slice,
                    "tile_spatial": tile_spatial,
                }
            )
            events.append("decode")
            yield mx.zeros((1, 3, 1, 48, 64), dtype=mx.float32)

    class FakeScheduler:
        step_index = None

        def __init__(self):
            self.timesteps = mx.array([], dtype=mx.float32)
            self.sigmas = mx.array([], dtype=mx.float32)

        def set_timesteps(self, steps):
            assert steps == 1
            self.timesteps = mx.array([500.0], dtype=mx.float32)
            self.sigmas = mx.array([0.5, 0.0], dtype=mx.float32)

        def step(self, prediction, timestep, latents, return_dict=False):
            assert prediction.shape == latents.shape
            assert return_dict is False
            self.step_index = 1
            return (latents,)

    transformer = FakeTransformer()
    model.transformer = transformer
    model.vae = FakeVae()
    original_cleanup_step_cache = BerniniRenderer._cleanup_step_cache
    original_release_denoisers = BerniniRenderer._release_denoisers

    monkeypatch.setattr(
        BerniniRenderer,
        "encode_prompt",
        lambda self, **kwargs: (
            mx.ones((1, 2, 3), dtype=mx.float32),
            mx.zeros((1, 2, 3), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(BerniniRenderer, "_require_tensor_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(BerniniRenderer, "_emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(BerniniRenderer, "_create_scheduler", lambda *args, **kwargs: FakeScheduler())
    monkeypatch.setattr(
        BerniniRenderer,
        "_plan_condition_metadata",
        lambda self, **kwargs: {"output_width": 64, "output_height": 48, "output_frames": 1},
    )

    def prepare_conditions(self, **kwargs):
        condition_cache_flags.append(kwargs["clear_cache"])
        return (
            None,
            [],
            [mx.zeros((1, 16, 1, 6, 8), dtype=mx.float32)],
            {
                "output_width": 64,
                "output_height": 48,
                "output_frames": 1,
                "reference_pixel_shapes": [[1, 3, 1, 48, 64]],
            },
        )

    monkeypatch.setattr(BerniniRenderer, "_prepare_condition_latents", prepare_conditions)
    monkeypatch.setattr(
        BerniniRenderer,
        "prepare_latents",
        lambda self, **kwargs: mx.ones((1, 16, 1, 6, 8), dtype=mx.float32),
    )

    def cleanup_step_cache(*, clear_cache):
        step_cache_flags.append(clear_cache)
        events.append("step-cleanup")
        original_cleanup_step_cache(clear_cache=clear_cache)

    def release_denoisers(self):
        events.append("release")
        original_release_denoisers(self)

    monkeypatch.setattr(BerniniRenderer, "_cleanup_step_cache", staticmethod(cleanup_step_cache))
    monkeypatch.setattr(BerniniRenderer, "_release_denoisers", release_denoisers)
    monkeypatch.setattr("mflux.models.wan.variants.wan_bernini.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan_bernini.mx.synchronize", lambda: None)
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan_bernini.mx.clear_cache",
        lambda: events.append("cache-clear"),
    )

    video = model.generate_video(
        seed=7,
        prompt="make the subject turn",
        reference_image_paths=[reference],
        width=64,
        height=48,
        num_frames=1,
        num_inference_steps=1,
        fps=8,
        release_denoisers_before_decode=True,
        clear_cache_each_step=True,
        clear_cache_each_transformer_block=True,
    )

    assert condition_cache_flags == [True]
    assert step_cache_flags == [True]
    assert len(transformer.calls) == 3
    assert [call["clear_cache_each_block"] for call in transformer.calls] == [True, True, True]
    forward_indices = [index for index, event in enumerate(events) if event == "forward"]
    assert all(events[index + 1] == "cache-clear" for index in forward_indices)
    assert events.index("release") > forward_indices[-1]
    assert decode_observations == []
    assert model.transformer is None
    assert model.transformer_2 is None

    frame = video.first_frame()

    assert frame.size == (64, 48)
    assert decode_observations == [
        {
            "latents_shape": (1, 16, 1, 6, 8),
            "transformer": None,
            "transformer_2": None,
            "clear_cache_each_slice": True,
            "tile_spatial": True,
        }
    ]
    assert events.index("decode") > events.index("release")


def test_bernini_step_cache_only_skips_branch_cache_flushes(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    source.touch()
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()
    model.transformer_2 = None
    model.bits = 4
    branch_flags = {}

    class FakeTransformer:
        patch_size = (1, 2, 2)
        vace_layers = None

        def forward_packed(self, **kwargs):
            return kwargs["latent_segments"][-1]

    class FakeVae:
        spatial_scale = 8
        temporal_scale = 4
        z_dim = 16

        def iter_decode_normalized_latent_slices(
            self,
            latents,
            *,
            clear_cache_each_slice=False,
            tile_spatial=False,
        ):
            yield mx.zeros((1, 3, 1, 48, 64), dtype=mx.float32)

    class FakeScheduler:
        step_index = None

        def __init__(self):
            self.timesteps = mx.array([], dtype=mx.float32)
            self.sigmas = mx.array([], dtype=mx.float32)

        def set_timesteps(self, steps):
            assert steps == 1
            self.timesteps = mx.array([500.0], dtype=mx.float32)
            self.sigmas = mx.array([0.5, 0.0], dtype=mx.float32)

        def step(self, prediction, timestep, latents, return_dict=False):
            self.step_index = 1
            return (latents,)

    model.transformer = FakeTransformer()
    model.vae = FakeVae()

    monkeypatch.setattr(
        BerniniRenderer,
        "encode_prompt",
        lambda self, **kwargs: (
            mx.ones((1, 2, 3), dtype=mx.float32),
            mx.zeros((1, 2, 3), dtype=mx.float32),
        ),
    )
    monkeypatch.setattr(BerniniRenderer, "_require_tensor_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(BerniniRenderer, "_emit_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(BerniniRenderer, "_create_scheduler", lambda *args, **kwargs: FakeScheduler())
    monkeypatch.setattr(
        BerniniRenderer,
        "_plan_condition_metadata",
        lambda self, **kwargs: {"output_width": 64, "output_height": 48, "output_frames": 1},
    )
    monkeypatch.setattr(
        BerniniRenderer,
        "_prepare_condition_latents",
        lambda self, **kwargs: (
            None,
            [],
            [mx.zeros((1, 16, 1, 6, 8), dtype=mx.float32)],
            {
                "output_width": 64,
                "output_height": 48,
                "output_frames": 1,
                "source_image_pixel_shape": [1, 3, 1, 48, 64],
            },
        ),
    )
    monkeypatch.setattr(
        BerniniRenderer,
        "prepare_latents",
        lambda self, **kwargs: mx.ones((1, 16, 1, 6, 8), dtype=mx.float32),
    )

    def fake_cfg(self, **kwargs):
        branch_flags["clear_branch_cache"] = kwargs["clear_branch_cache"]
        branch_flags["clear_cache_each_block"] = kwargs["clear_cache_each_block"]
        return mx.zeros_like(kwargs["target"])

    monkeypatch.setattr(BerniniRenderer, "_cfg_noise_prediction", fake_cfg)
    monkeypatch.setattr(BerniniRenderer, "_cleanup_step_cache", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr("mflux.models.wan.variants.wan_bernini.gc.collect", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan_bernini.mx.synchronize", lambda: None)
    monkeypatch.setattr("mflux.models.wan.variants.wan_bernini.mx.clear_cache", lambda: None)

    video = model.generate_video(
        seed=7,
        prompt="make the subject turn",
        image_path=source,
        guidance_mode="v2v",
        width=64,
        height=48,
        num_frames=1,
        num_inference_steps=1,
        fps=8,
        clear_cache_each_step=True,
        clear_cache_each_transformer_block=False,
    )

    assert branch_flags == {
        "clear_branch_cache": False,
        "clear_cache_each_block": False,
    }
    assert video.first_frame().size == (64, 48)


def test_bernini_video_generation_starts_from_noise_not_scheduler_warm_start(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    model = BerniniRenderer.__new__(BerniniRenderer)
    model.model_config = ModelConfig.bernini_r_1_3b()
    model.transformer_2 = None
    model.bits = 4
    calls = {"prepare_latents": 0, "scheduler_steps": 0}
    encoded = {}
    lifecycle = []

    class FakeTransformer:
        patch_size = (1, 2, 2)
        vace_layers = None

        def forward_packed(self, **kwargs):
            return kwargs["latent_segments"][-1]

    class FakeVae:
        spatial_scale = 8
        temporal_scale = 4
        z_dim = 16

        def iter_decode_normalized_latent_slices(self, *args, **kwargs):
            raise AssertionError("deferred decode must not run while constructing the artifact")

    class FakeScheduler:
        step_index = None

        def __init__(self):
            self.timesteps = mx.array([], dtype=mx.float32)
            self.sigmas = mx.array([], dtype=mx.float32)

        def set_timesteps(self, steps):
            self.timesteps = mx.array([500.0], dtype=mx.float32)
            self.sigmas = mx.array([0.5, 0.0], dtype=mx.float32)

        def step(self, prediction, timestep, latents, return_dict=False):
            calls["scheduler_steps"] += 1
            self.step_index = 1
            return (latents - 0.1 * prediction,)

    model.transformer = FakeTransformer()
    model.vae = FakeVae()
    model.tokenizers = {"wan": object()}
    observed = {}

    def fake_encode_prompt(self, **kwargs):
        lifecycle.append("encode-prompt")
        encoded.update(kwargs)
        return mx.ones((1, 8, 3)), mx.zeros((1, 8, 3))

    monkeypatch.setattr(BerniniRenderer, "encode_prompt", fake_encode_prompt)
    monkeypatch.setattr(
        BerniniRenderer,
        "_tokenize_prompts",
        lambda self, *, cleaned, max_sequence_length: {
            "attention_mask": np.array(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        },
    )
    monkeypatch.setattr(BerniniRenderer, "_require_tensor_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        BerniniRenderer,
        "_emit_progress",
        lambda self, callback, *, phase, **kwargs: lifecycle.append(phase),
    )
    monkeypatch.setattr(BerniniRenderer, "_cleanup_step_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(BerniniRenderer, "_create_scheduler", lambda *args, **kwargs: FakeScheduler())

    def fake_prepare(self, seed, batch_size, height, width, num_frames):
        calls["prepare_latents"] += 1
        return mx.ones((1, 16, 2, height // 8, width // 8), dtype=mx.float32)

    def fake_conditions(self, **kwargs):
        lifecycle.append("encode-conditions")
        return (
            mx.zeros((1, 16, 2, 6, 8), dtype=mx.float32),
            [],
            [],
            {
                "source_width": 64,
                "source_height": 48,
                "output_width": 64,
                "output_height": 48,
                "output_frames": 5,
                "reference_pixel_shapes": [],
            },
        )

    def fake_plan(self, **kwargs):
        lifecycle.append("plan-conditions")
        return {"output_width": 64, "output_height": 48, "output_frames": 5}

    monkeypatch.setattr(BerniniRenderer, "prepare_latents", fake_prepare)
    monkeypatch.setattr(BerniniRenderer, "_plan_condition_metadata", fake_plan)
    monkeypatch.setattr(BerniniRenderer, "_prepare_condition_latents", fake_conditions)
    monkeypatch.setattr(
        BerniniRenderer,
        "_single_condition_apg_noise_prediction",
        lambda self, target, **kwargs: observed.update(
            prompt_len=int(kwargs["prompt_embeds"].shape[1]),
            negative_len=int(kwargs["negative_prompt_embeds"].shape[1]),
        )
        or mx.zeros_like(target),
    )
    monkeypatch.setattr(
        VideoUtil,
        "to_video_from_frame_batches",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    artifact = model.generate_video(
        seed=7,
        prompt="  make   the person crouch  ",
        video_path=source,
        width=64,
        height=48,
        num_frames=5,
        num_inference_steps=1,
        fps=8,
    )

    assert calls == {"prepare_latents": 1, "scheduler_steps": 1}
    assert lifecycle[:4] == ["plan-conditions", "start", "encode-prompt", "encode-conditions"]
    assert encoded["prompt"] == ("You are a helpful assistant specialized in video editing.make the person crouch")
    assert encoded["clean_prompts"] is False
    assert artifact.task == "video-to-video"
    assert artifact.extra_metadata["source_video_warm_start"] is False
    assert artifact.extra_metadata["bernini_guidance_mode"] == "v2v_apg"
    assert artifact.extra_metadata["condition_source_ids"] == [1.0]
    assert artifact.extra_metadata["system_prompt"] == ("You are a helpful assistant specialized in video editing.")
    assert observed == {"prompt_len": 8, "negative_len": 8}


@pytest.mark.parametrize(
    ("guidance_mode", "expected"),
    [
        ("r2v_apg", "You are a helpful assistant specialized in subject-to-video generation."),
        ("rv2v", "You are a helpful assistant specialized in video editing with reference."),
        ("v2v_apg", "You are a helpful assistant specialized in video editing."),
    ],
)
def test_bernini_system_prompt_defaults_match_official_task_prefixes(guidance_mode, expected):
    assert BerniniRenderer._resolved_system_prompt(guidance_mode=guidance_mode, system_prompt=None) == expected
    assert BerniniRenderer._resolved_system_prompt(guidance_mode=guidance_mode, system_prompt="") == expected
    assert (
        BerniniRenderer._resolved_system_prompt(guidance_mode=guidance_mode, system_prompt="custom-prefix:")
        == "custom-prefix:"
    )
