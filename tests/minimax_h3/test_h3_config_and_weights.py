"""Catalog, capability, weight-definition and LoRA-mapping contracts of the MiniMax-H3 family."""

from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
from mflux.models.minimax_h3.weights.h3_lora_mapping import MiniMaxH3LoRAMapping
from mflux.models.minimax_h3.weights.h3_weight_definition import MiniMaxH3WeightDefinition
from mflux.models.minimax_h3.weights.h3_weight_mapping import MiniMaxH3WeightMapping
from mflux.task_inference import get_model_capabilities


@pytest.mark.fast
def test_catalog_entries_and_capabilities():
    base = ModelConfig.from_name("minimax-h3")
    turbo = ModelConfig.from_name("minimax-h3-turbo")
    assert base.model_name == turbo.model_name == "MiniMaxAI/MiniMax-H3"
    assert base.transformer_overrides["default_steps"] == 50 and turbo.transformer_overrides["default_steps"] == 8
    assert turbo.transformer_overrides["default_video_shift"] == 6.0 and "turbo_lora" in turbo.transformer_overrides
    assert ModelConfig.from_name("MiniMaxAI/MiniMax-H3").aliases[0] == "minimax-h3"
    assert ModelConfig.minimax_h3_turbo_544p().transformer_overrides["default_width"] == 960
    assert ModelConfig.minimax_h3_turbo().transformer_overrides["default_video_shift"] == 6.0
    capabilities = get_model_capabilities(model="minimax-h3-turbo", model_config=turbo).to_dict()
    # Each entry's label names its canvas: choosing between them is the 11-versus-34-minute decision.
    assert capabilities["family"] == "minimax-h3" and capabilities["label"] == "MiniMax-H3 Turbo 768p"
    text_row, image_row = capabilities["capabilities"]
    assert text_row["handler_id"] == "minimax-h3.generate" and text_row["public_task"] == "text-to-video"
    assert (
        text_row["supports_negative_prompt"] is False
        and text_row["supports_lora"] is True
        and text_row["dimension_multiple"] == 32
    )
    assert image_row["id"] == "minimax-h3.first-frame" and image_row["public_task"] == "image-to-video"
    assert (
        image_row["min_images"] == 1
        and image_row["max_images"] == 1
        and image_row["default_canvas_policy"] == "source-aspect"
    )


@pytest.mark.fast
def test_weight_definition_components_and_quantization_predicate():
    names = [component.name for component in MiniMaxH3WeightDefinition.get_components()]
    assert names == ["text_encoder", "transformer", "vae", "audio_vae"]
    predicate = MiniMaxH3WeightDefinition.quantization_predicate
    assert predicate("transformer_blocks.0.attn.to_q", nn.Linear(5376, 7168, bias=False)) is True
    assert predicate("proj_in", nn.Linear(96, 5376)) is False  # fp32 keep-set
    assert predicate("time_embedder.linear_1", nn.Linear(256, 5376)) is False
    assert (
        predicate("proj_out", nn.Linear(5376, 96)) is False
        and predicate("audio_proj_out", nn.Linear(5376, 32)) is False
    )
    assert predicate("layers.0.self_attn.q_norm", nn.RMSNorm(128)) is False
    assert predicate("transformer_blocks.7.adaln_proj.linear", nn.Linear(2688, 6 * 5376 * 3)) is False
    assert predicate("norm_out.linear", nn.Linear(2688, 2 * 5376)) is False
    assert predicate("transformer_blocks.7.ff.net.2", nn.Linear(14336, 5376, bias=False)) is True
    assert MiniMaxH3WeightDefinition.transformer_precision_override("audio_proj_out.weight") == mx.float32
    assert MiniMaxH3WeightDefinition.transformer_precision_override("transformer_blocks.3.ff.net.2.weight") is None
    (tokenizer,) = MiniMaxH3WeightDefinition.get_tokenizers()
    assert tokenizer.add_special_tokens is False and tokenizer.padding == "longest"


@pytest.mark.fast
def test_text_encoder_key_selection_and_vae_transforms():
    needed = MiniMaxH3WeightMapping.text_encoder_key_is_needed
    assert needed("model.language_model.embed_tokens.weight")
    assert needed("model.language_model.layers.49.mlp.up_proj.weight")
    assert not needed("model.language_model.layers.50.mlp.up_proj.weight")
    assert not needed("model.language_model.norm.weight") and not needed("lm_head.weight")
    assert needed("model.visual.blocks.0.attn.qkv.weight") and needed("model.visual.merger.linear_fc2.weight")
    assert MiniMaxH3WeightMapping.video_vae_transform(mx.zeros((8, 3, 3, 3, 3))).shape == (8, 3, 3, 3, 3)
    assert MiniMaxH3WeightMapping.video_vae_transform(mx.zeros((8, 4, 3, 3, 3))).shape == (8, 3, 3, 3, 4)
    g, v = mx.random.normal((6, 1, 1)), mx.random.normal((6, 4, 7))
    folded = MiniMaxH3WeightMapping.fold_weight_norm(g, v)
    norms = mx.sqrt(mx.sum(mx.square(folded), axis=(1, 2)))
    assert mx.allclose(norms, mx.abs(g[:, 0, 0]), atol=1e-5).item()
    state = MiniMaxH3WeightMapping.convert_audio_vae_state(
        {"encoder.block.0.weight_g": g, "encoder.block.0.weight_v": v, "decoder.ups.0.0.weight": mx.zeros((4, 8, 9))}
    )
    assert state["encoder.block.0.weight"].shape == (6, 7, 4) and state["decoder.ups.0.0.weight"].shape == (8, 9, 4)


@pytest.mark.fast
def test_turbo_lora_keys_map_onto_every_target_and_alpha_scale_is_read(tmp_path):
    mappings = LoRALoader._build_pattern_mappings(MiniMaxH3LoRAMapping.get_mapping())
    keys = {
        f"{prefix}.{module}.lora_{matrix}.default.weight"
        for prefix in ("transformer_blocks.12", "token_refiner.refiner_blocks.1")
        for module in MiniMaxH3LoRAMapping.TARGET_MODULES
        for matrix in "AB"
    }
    normalized = LoRALoader._normalize_peft_adapter_infix({key: mx.zeros((1, 1)) for key in keys})
    for key in normalized:
        matched = [
            (m, LoRALoader._match_pattern(key, m.source_pattern))
            for m in mappings
            if LoRALoader._match_pattern(key, m.source_pattern) is not None
        ]
        assert matched, key
        assert {m.target_path.replace("{block}", str(block)) for m, block in matched} == {key.rsplit(".lora_", 1)[0]}
    path = tmp_path / "turbo.safetensors"
    mx.save_safetensors(
        str(path),
        {
            "transformer_blocks.0.attn.to_q.lora_A.default.weight": mx.zeros((128, 5376)),
            "transformer_blocks.0.attn.to_q.lora_B.default.weight": mx.zeros((7168, 128)),
        },
        {"alpha": "8"},
    )
    assert MiniMaxH3Initializer._peft_alpha_scale(str(path)) == pytest.approx(8 / 128)


@pytest.mark.fast
def test_h3_cli_accepts_mlx_cache_limit():
    from mflux.models.minimax_h3.cli.minimax_h3_generate import _parser

    args = _parser().parse_args(["--model", "minimax-h3", "--prompt", "x", "--mlx-cache-limit-gb", "2.5"])
    assert args.mlx_cache_limit_gb == 2.5
    assert _parser().parse_args(["--model", "minimax-h3", "--prompt", "x"]).mlx_cache_limit_gb is None


@pytest.mark.fast
def test_h3_cli_resolves_prepared_packages_through_base_model():
    from mflux.models.minimax_h3.cli.minimax_h3_generate import _resolve_model

    config, path = _resolve_model("models/minimax-h3-8bit", "minimax-h3-turbo-544p")
    assert config.aliases[0] == "minimax-h3-turbo-544p" and path == "models/minimax-h3-8bit"
    assert config.transformer_overrides.get("turbo_lora")

    config, path = _resolve_model("models/minimax-h3-8bit")
    assert config.aliases[0] == "minimax-h3" and path == "models/minimax-h3-8bit"

    config, path = _resolve_model("minimax-h3-turbo-544p")
    assert config.aliases[0] == "minimax-h3-turbo-544p" and path is None
    assert _parser_accepts_base_model()


def _parser_accepts_base_model() -> bool:
    from mflux.models.minimax_h3.cli.minimax_h3_generate import _parser

    args = _parser().parse_args(["--model", "models/x", "--base-model", "minimax-h3-turbo", "--prompt", "x"])
    return args.base_model == "minimax-h3-turbo"


@pytest.mark.fast
def test_qwen2_tokenizer_workaround_keeps_config_only_special_tokens(tmp_path):
    """MiniMax-H3 declares `<d>`/`</d>` only in tokenizer_config.json; they must get the ids transformers assigns."""
    import json

    from transformers import Qwen2Tokenizer

    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader

    # Byte-level BPE alphabet (GPT-2 / Qwen2 `bytes_to_unicode`): printable bytes map to themselves, the rest shift up.
    printable = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    alphabet = [chr(b) for b in printable]
    shift = 0
    for b in range(256):
        if b not in printable:
            alphabet.append(chr(256 + shift))
            shift += 1
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["<|endoftext|>"] = len(vocab)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "added_tokens_decoder": {str(vocab["<|endoftext|>"]): {"content": "<|endoftext|>", "special": True}},
                "additional_special_tokens": ["<|endoftext|>", "<d>", "</d>"],
            }
        )
    )
    ours = TokenizerLoader._load_qwen2_tokenizer_workaround(tmp_path, Qwen2Tokenizer)
    reference = Qwen2Tokenizer.from_pretrained(str(tmp_path))
    text = "a <d>hi</d>"
    assert ours.convert_tokens_to_ids("<d>") == reference.convert_tokens_to_ids("<d>") == len(vocab)
    assert ours(text, add_special_tokens=False)["input_ids"] == reference(text, add_special_tokens=False)["input_ids"]


@pytest.mark.fast
def test_python_runtime_rejects_a_keyword_the_route_does_not_take():
    """A Wan-shaped keyword used to surface as a bare TypeError once the weights were resident."""
    from types import SimpleNamespace

    from mflux.python_runtime import _RuntimeGenerationExecutor
    from mflux.task_inference import TaskInferenceError

    def generate_video(seed, prompt=None, soundscape=None, music=None, num_frames=None):
        return None

    loaded = SimpleNamespace(
        plan=SimpleNamespace(task="text-to-video"),
        model_config=SimpleNamespace(model_name="MiniMaxAI/MiniMax-H3"),
    )
    reject = _RuntimeGenerationExecutor._reject_unknown_generate_kwargs

    with pytest.raises(TaskInferenceError) as excinfo:
        reject(generate_method=generate_video, generate_kwargs={"fps": 24}, loaded=loaded)
    message = str(excinfo.value)
    assert "'fps'" in message and "MiniMaxAI/MiniMax-H3" in message and "soundscape" in message

    # Accepted keywords pass, and a callable taking **kwargs is never second-guessed.
    reject(generate_method=generate_video, generate_kwargs={"prompt": "x", "soundscape": "y"}, loaded=loaded)
    reject(generate_method=lambda seed, **kwargs: None, generate_kwargs={"anything": 1}, loaded=loaded)


@pytest.mark.fast
def test_capability_rows_publish_the_audio_duration_and_default_contract():
    """A host builds H3 controls from the row alone, without importing ModelConfig or hardcoding the family."""
    from mflux.task_inference import CAPABILITIES_SCHEMA_VERSION, get_model_capabilities

    assert CAPABILITIES_SCHEMA_VERSION == 16
    payload = get_model_capabilities(model="minimax-h3-turbo-544p").to_dict()
    rows = {row["id"]: row for row in payload["capabilities"]}
    assert set(rows) == {"minimax-h3.text-video", "minimax-h3.first-frame"}

    for row in rows.values():
        # The soundtrack: generated with the picture, unlike a restoration row's passthrough.
        assert row["generates_audio"] is True
        assert row["audio_channels"] == 2 and row["audio_sample_rate"] == 32000
        assert row["supports_audio_shift"] is True and row["supports_flow_shift"] is True
        assert row["supports_text_encoder_release"] is True
        # The duration contract: the grid, the real bounds, and the rate the route writes.
        assert (row["min_frames"], row["max_frames"]) == (124, 345)
        assert (row["frame_multiple"], row["frame_remainder"]) == (17, 5)
        assert row["frame_rounding"] == "up" and row["output_fps"] == 24.0
        assert row["supports_fps"] is False
        # Per-entry defaults, so a host seeds its controls from the catalog.
        assert (row["default_width"], row["default_height"]) == (960, 544)
        assert row["default_steps"] == 8
        assert (row["default_flow_shift"], row["default_audio_shift"]) == (12.0, 3.0)
        # The engine's own section labels, addressable and in order.
        sections = row["prompt_sections"]
        assert [s["label"] for s in sections] == [
            "integrated_multimodal_description",
            "overall_soundscape",
            "non_diegetic_music",
        ]
        assert [s["option"] for s in sections] == ["--prompt", "--soundscape", "--music"]
        assert [s["parameter"] for s in sections] == ["prompt", "soundscape", "music"]
        assert [s["role"] for s in sections] == ["picture", "audio", "audio"]
        assert sections[0]["required"] is True and sections[1]["required"] is False

    # The published bounds are the ones the runtime enforces.
    from mflux.models.minimax_h3.latent_creator.h3_layout import valid_frame_counts

    counts = valid_frame_counts()
    row = rows["minimax-h3.text-video"]
    assert counts[0] == row["min_frames"] and counts[-1] == row["max_frames"]
    assert all(count % row["frame_multiple"] == row["frame_remainder"] for count in counts)


@pytest.mark.fast
def test_each_catalog_entry_publishes_a_distinct_label_and_its_own_defaults():
    """The two Turbo entries used to share one label, hiding the 11-versus-34-minute choice."""
    from mflux.task_inference import get_model_capabilities

    seen = {}
    for alias, canvas, steps, shift in (
        ("minimax-h3", (1344, 768), 50, 12.0),
        ("minimax-h3-turbo", (1344, 768), 8, 6.0),
        ("minimax-h3-turbo-544p", (960, 544), 8, 12.0),
    ):
        payload = get_model_capabilities(model=alias).to_dict()
        row = payload["capabilities"][0]
        seen[alias] = payload["label"]
        assert (row["default_width"], row["default_height"]) == canvas
        assert row["default_steps"] == steps and row["default_flow_shift"] == shift
    assert len(set(seen.values())) == 3, seen
    assert seen["minimax-h3-turbo"] != seen["minimax-h3-turbo-544p"]


@pytest.mark.fast
def test_rows_publish_the_precision_and_memory_contract():
    """A host decides whether a run fits from the row, in bytes, without parsing prose."""
    from mflux.task_inference import CAPABILITIES_SCHEMA_VERSION, get_model_capabilities

    assert CAPABILITIES_SCHEMA_VERSION == 16
    for alias in ("minimax-h3-turbo-544p", "minimax-h3-turbo"):
        for row in get_model_capabilities(model=alias).to_dict()["capabilities"]:
            assert row["weight_precision"] == "bf16"
            assert row["recommended_quantize"] == 8
            assert row["validated_quantization_bits"] == [8]
            # Unquantized the weights take 97% of a 128 GiB machine, leaving nothing for the run,
            # which is why q8 is validated rather than optional. Published in bytes so a host never
            # has to guess GB versus GiB: 134.2 GB is 125 GiB, and the difference decides the run.
            assert row["unquantized_weights_bytes"] / 1024**3 > 0.95 * 128
            assert row["unquantized_weights_bytes"] / 1024**3 < 128
            completed = [run for run in row["measured_runs"] if run["outcome"] == "completed"]
            assert {(run["width"], run["height"]) for run in completed} == {(960, 544), (1344, 768)}
            for run in completed:
                assert run["quantize"] == 8 and run["frames"] == 124
                assert 80 * 1024**3 < run["footprint_bytes"] < 128 * 1024**3


@pytest.mark.fast
def test_guidance_and_negative_prompt_are_published_independently():
    """They diverge in both directions, so neither is a usable proxy for the other."""
    from mflux.task_inference import get_model_capabilities

    def flags(alias):
        rows = get_model_capabilities(model=alias).to_dict()["capabilities"]
        return {(row["supports_guidance"], row["supports_negative_prompt"]) for row in rows}

    assert flags("flux2-klein-4b") == {(True, False)}, "distilled Klein: guidance, no negative prompt"
    assert flags("z-image-turbo") == {(False, True)}, "Z-Image Turbo: negative prompt, no guidance"
    assert flags("z-image") == {(True, True)}
    assert flags("minimax-h3-turbo-544p") == {(False, False)}


@pytest.mark.fast
def test_memory_preflight_refuses_a_load_that_cannot_fit(tmp_path, monkeypatch):
    """Unquantized MiniMax-H3 does not fit a 128 GiB machine; it must say so, not be killed mid-load."""
    from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer
    from mflux.utils.runtime_memory import RuntimeMemory

    weight_definition = MiniMaxH3WeightDefinition.for_config(ModelConfig.minimax_h3())
    gib = 1024**3
    monkeypatch.setattr(MiniMaxH3Initializer, "_estimate_resident_weight_bytes", lambda *a, **k: 125 * gib)
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 128 * gib))

    with pytest.raises(MemoryError) as excinfo:
        MiniMaxH3Initializer._preflight_memory(tmp_path, weight_definition, None)
    message = str(excinfo.value)
    assert "125 GiB" in message and "128 GiB" in message and "--quantize 8" in message

    # A machine that can hold it proceeds, and so does a quantized load.
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 512 * gib))
    MiniMaxH3Initializer._preflight_memory(tmp_path, weight_definition, None)
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 128 * gib))
    monkeypatch.setattr(MiniMaxH3Initializer, "_estimate_resident_weight_bytes", lambda *a, **k: 71 * gib)
    MiniMaxH3Initializer._preflight_memory(tmp_path, weight_definition, 8)

    # Unknown physical memory never guesses: a preflight that invents a number refuses runs that fit.
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 0))
    monkeypatch.setattr(MiniMaxH3Initializer, "_estimate_resident_weight_bytes", lambda *a, **k: 900 * gib)
    MiniMaxH3Initializer._preflight_memory(tmp_path, weight_definition, None)


@pytest.mark.fast
def test_flow_shift_is_published_by_concept_not_by_one_family_spelling():
    """Wan spells it --flow-shift and MiniMax-H3 --video-shift, but the mechanism is one.

    Naming the field after either spelling would publish `false` on the family using the other, which
    is a positive claim that the route has no shift control at all.
    """
    from mflux.task_inference import get_model_capabilities

    for alias, expected_default in (
        ("wan2.2-t2v-a14b", 3.0),
        ("wan2.2-ti2v-5b", 5.0),
        ("minimax-h3-turbo-544p", 12.0),
        ("minimax-h3-turbo", 6.0),
    ):
        for row in get_model_capabilities(model=alias).to_dict()["capabilities"]:
            assert row["supports_flow_shift"] is True, alias
            assert row["default_flow_shift"] == expected_default, alias

    # A route with no flow shift says so, and says nothing about a default.
    for row in get_model_capabilities(model="qwen-image").to_dict()["capabilities"]:
        assert row["supports_flow_shift"] is False and row["default_flow_shift"] is None


@pytest.mark.fast
def test_h3_accepts_low_ram_like_every_other_generate_route():
    """`--low-ram` is a shared-parser option; MiniMax-H3 was the one generate route rejecting it."""
    from mflux.cli import mlx_gen
    from mflux.models.minimax_h3.cli.minimax_h3_generate import _parser

    assert _parser().parse_args(["--model", "m", "--prompt", "x", "--low-ram"]).low_ram is True
    assert _parser().parse_args(["--model", "m", "--prompt", "x"]).low_ram is False
    invocation = mlx_gen._resolve_invocation(["--model", "minimax-h3-turbo-544p", "--prompt", "x", "--low-ram"])
    assert "--low-ram" in invocation.argv


@pytest.mark.fast
def test_flow_shift_spelling_is_published_and_matches_the_route():
    """The concept is shared, the spelling is not, so the row carries the spelling a host must emit.

    Pinned against the real parsers and the real generate signatures: this fails if either family
    renames its option or its keyword.
    """
    import inspect

    from mflux.models.minimax_h3.cli.minimax_h3_generate import _parser as h3_parser
    from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3
    from mflux.models.wan.cli.wan_generate import _parser as wan_parser
    from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
    from mflux.task_inference import get_model_capabilities

    cases = (
        ("wan2.2-ti2v-5b", wan_parser, ["--model", "m", "--prompt", "x"], Wan2_2_TI2V),
        ("minimax-h3-turbo-544p", h3_parser, ["--model", "m", "--prompt", "x"], MiniMaxH3),
    )
    for alias, parser_factory, base_argv, variant in cases:
        for row in get_model_capabilities(model=alias).to_dict()["capabilities"]:
            assert row["supports_flow_shift"] is True, alias
            option, parameter = row["flow_shift_option"], row["flow_shift_parameter"]
            # The published option really parses on that route, and lands on the published keyword.
            namespace = parser_factory().parse_args([*base_argv, option, "5.0"])
            assert getattr(namespace, parameter) == 5.0, (alias, option, parameter)
            # And the keyword really exists on the Python entry point a host would call instead.
            assert parameter in inspect.signature(variant.generate_video).parameters, (alias, parameter)

    # The spelling is present exactly when the control is, on every row of every catalog entry.
    for alias in ("wan2.2-ti2v-5b", "minimax-h3", "qwen-image", "z-image-turbo"):
        for row in get_model_capabilities(model=alias).to_dict()["capabilities"]:
            assert (row["flow_shift_option"] is None) == (not row["supports_flow_shift"]), alias
            assert (row["flow_shift_parameter"] is None) == (not row["supports_flow_shift"]), alias


@pytest.mark.fast
def test_universal_options_are_accepted_by_every_generate_parser():
    """`universal_options` is a promise about the whole build, so it is checked against every parser.

    Generate routes take their options from one of three places: the shared `CommandLineParser`, and
    the two hand-rolled parsers (Wan and MiniMax-H3). Covering those three covers every route.
    """
    import importlib

    import toml

    from mflux.task_inference import UNIVERSAL_GENERATE_OPTIONS, get_model_capabilities

    published = set(UNIVERSAL_GENERATE_OPTIONS)
    assert published, "the list is a promise; an empty one is not worth publishing"

    # Every generate console script, read from the packaging metadata so a new route joins this
    # check by existing. Asserting on a parser built here instead would only prove that the shared
    # parser defines the option, not that any route calls it.
    scripts = toml.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    router = "mflux.cli.mlx_gen:main"
    targets = {
        name: target
        for name, target in scripts.items()
        if "generate" in name and "upscale" not in name and target != router
    }
    assert len(targets) >= 15, f"expected the generate surface, found {sorted(targets)}"

    for name, target in sorted(targets.items()):
        module = importlib.import_module(target.split(":")[0])
        help_text = _capture_help(module)
        missing = sorted(option for option in published if option not in help_text)
        assert not missing, f"{name} ({target}) does not accept {missing}"

    # The router declares none of these itself: it forwards what it does not consume, so the promise
    # holds through `mlxgen generate` only if the option actually reaches the backend argv.
    from mflux.cli import mlx_gen

    for alias in ("minimax-h3-turbo-544p", "wan2.2-t2v-a14b", "qwen-image"):
        argv = mlx_gen._resolve_invocation(["--model", alias, "--prompt", "x", *published]).argv
        assert published <= set(argv), f"the router dropped {sorted(published - set(argv))} for {alias}"

    # And the payload publishes it, so a host reads the build instead of the release number.
    payload = get_model_capabilities(model="minimax-h3-turbo-544p").to_dict()
    assert payload["universal_options"] == list(UNIVERSAL_GENERATE_OPTIONS)


def _capture_help(module) -> str:
    """The `--help` text of a CLI module's own parser, however that module builds it."""
    import contextlib
    import io
    import sys

    argv = sys.argv
    buffer = io.StringIO()
    try:
        sys.argv = [module.__name__, "--help"]
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            module.main()
    finally:
        sys.argv = argv
    return buffer.getvalue()


@pytest.mark.fast
def test_packed_sequence_length_is_the_axis_the_peak_follows():
    """Peak tracks packed rows, not the canvas or the frame count separately.

    A 243-frame `960x544` request and a 124-frame `1344x768` request differ by 0.5% in rows, and were
    measured at the same MLX peak. The helper exists so a caller can size a request without
    reverse-engineering the layout.
    """
    from mflux.models.minimax_h3.latent_creator.h3_layout import packed_sequence_length

    assert packed_sequence_length(960, 544, 124) == 19_484
    assert packed_sequence_length(1344, 768, 124) == 37_910
    assert packed_sequence_length(960, 544, 243) == 37_730
    assert packed_sequence_length(960, 544, 345) == 53_370
    assert packed_sequence_length(1344, 768, 345) == 104_166
    # Off-grid requests are sized at the count that will actually run.
    assert packed_sequence_length(960, 544, 130) == packed_sequence_length(960, 544, 141)


@pytest.mark.fast
def test_request_preflight_warns_near_the_limit_and_refuses_the_impossible(capsys, monkeypatch):
    """A request that cannot fit is reported before the run, not discovered by the OS killing it."""
    from types import SimpleNamespace

    from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3
    from mflux.utils.runtime_memory import RuntimeMemory

    gib = 1024**3
    model = MiniMaxH3.__new__(MiniMaxH3)

    def plan(width, height, frames):
        return SimpleNamespace(width=width, height=height, num_frames=frames)

    # Anchored on what is actually resident, so it estimates the loaded model rather than assuming
    # the released weights. At the released weights' measured 75.7 GiB baseline it reproduces both
    # published measurements and the killed run's observed footprint, all within 0.5 GiB.
    loaded = int(75.7 * gib)
    # The cache term is RAM-derived, so pin it to the 8 GiB the runs were measured under. Without
    # this the expectations below only hold on a machine with the same amount of memory.
    monkeypatch.setattr(RuntimeMemory, "resolve_cache_limit_bytes", staticmethod(lambda *a, **k: 8 * gib))

    def estimate(w, h, f):
        return MiniMaxH3.estimate_peak_bytes(w, h, f, resident_bytes=loaded) / gib

    assert abs(estimate(960, 544, 124) - 88.2) < 0.5
    assert abs(estimate(1344, 768, 124) - 92.9) < 0.5
    assert abs(estimate(960, 544, 243) - 92.7) < 0.6

    # A tiny model resident in a few hundred MB must not be sized as if it were the released one:
    # that refused every tiny-config test on a small machine.
    assert MiniMaxH3.estimate_peak_bytes(96, 64, 124, resident_bytes=200 * 1024**2) < 12 * gib
    # The cache term follows the machine, which is why the pin above is needed at all.
    monkeypatch.setattr(RuntimeMemory, "resolve_cache_limit_bytes", staticmethod(lambda *a, **k: 1 * gib))
    assert estimate(960, 544, 124) < 88.2 - 6.0
    monkeypatch.setattr(RuntimeMemory, "resolve_cache_limit_bytes", staticmethod(lambda *a, **k: 8 * gib))

    def resident(byte_count):
        monkeypatch.setattr(
            RuntimeMemory,
            "snapshot",
            staticmethod(
                lambda *a, **k: SimpleNamespace(
                    darwin_physical_footprint_bytes=byte_count, process_rss_bytes=byte_count
                )
            ),
        )

    resident(loaded)
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 128 * gib))
    model._preflight_request(plan(960, 544, 124))
    assert capsys.readouterr().err == "", "a run with headroom says nothing"

    model._preflight_request(plan(960, 544, 345))
    warning = capsys.readouterr().err
    assert "345 frames" in warning and "killed" in warning and "--release-text-encoder" in warning

    # A request larger than the whole machine is refused rather than warned about.
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 64 * gib))
    with pytest.raises(MemoryError, match="cannot fit"):
        model._preflight_request(plan(1344, 768, 345))

    # Unknown physical memory, and an unreadable footprint, never guess.
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 0))
    model._preflight_request(plan(1344, 768, 345))
    monkeypatch.setattr(RuntimeMemory, "total_physical_memory_bytes", staticmethod(lambda: 64 * gib))
    resident(0)
    model._preflight_request(plan(1344, 768, 345))


@pytest.mark.fast
def test_measured_runs_publish_outcomes_and_their_conditions():
    """A killed run is evidence a host needs; its number is a lower bound and must not be called a peak."""
    from mflux.task_inference import CAPABILITIES_SCHEMA_VERSION, get_model_capabilities

    assert CAPABILITIES_SCHEMA_VERSION == 16
    for row in get_model_capabilities(model="minimax-h3-turbo-544p").to_dict()["capabilities"]:
        runs = row["measured_runs"]
        assert len(runs) == 3
        outcomes = [run["outcome"] for run in runs]
        assert outcomes == ["completed", "completed", "killed"]
        for run in runs:
            # Conditions travel with every number, because they move it more than the canvas does.
            assert run["cache_limit_bytes"] == 8 * 1024**3
            assert run["machine_total_ram_bytes"] == 128 * 1024**3
            assert run["quantize"] == 8 and run["steps"] == 8
            assert "peak_bytes" not in run, "a killed run has no peak, only a highest observed value"
        killed = runs[-1]
        assert killed["frames"] == 243 and killed["terminated_at"]
        # The grid bound and the evidence bound are different questions.
        assert row["max_frames"] == 345 and row["max_validated_frames"] == 124
        assert row["peak_bytes_per_packed_row"] == 271_356 and row["peak_bytes_fixed"] == 89_410_000_000
