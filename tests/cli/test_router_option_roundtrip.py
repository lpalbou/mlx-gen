import argparse
import json

import pytest

from mflux.cli import mlx_gen
from mflux.cli.router_options import ROUTER_OPTIONS, ForwardPolicy, reemit_options

WAN_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
QWEN_MODEL = "Qwen/Qwen-Image"


def _wan_v2v_argv(extra: list[str]) -> list[str]:
    return [
        "--model",
        WAN_MODEL,
        "--video-path",
        "input.mp4",
        "--prompt",
        "change the ship silhouette",
        *extra,
    ]


def test_every_router_parser_action_maps_to_exactly_one_descriptor():
    parser = mlx_gen._parser()
    actions = [action for action in parser._actions if not isinstance(action, argparse._HelpAction)]
    descriptor_flags = {option.flags: option for option in ROUTER_OPTIONS}

    assert len(actions) == len(ROUTER_OPTIONS)
    for action in actions:
        key = tuple(action.option_strings)
        assert key in descriptor_flags, f"parser action {key} has no RouterOption descriptor"
        assert action.dest == descriptor_flags[key].dest


def test_every_descriptor_declares_an_explicit_fate():
    for option in ROUTER_OPTIONS:
        if option.policy in {ForwardPolicy.REEMIT_VALUE, ForwardPolicy.REEMIT_FLAG}:
            assert option.emit_flag in option.flags
            assert option.emit_order is not None
        elif option.policy is ForwardPolicy.TRANSFORMED:
            assert option.emitter, f"{option.dest} must name its owning emitter helper"
            assert option.emit_order is None
        else:
            assert option.policy is ForwardPolicy.ROUTER_ONLY
            assert option.emit_order is None


def test_reemit_descriptor_values_roundtrip_from_argv(tmp_path):
    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"png")
    invocation = mlx_gen._resolve_invocation(
        _wan_v2v_argv(
            [
                "--video-strength",
                "0.7",
                "--video-mask-path",
                str(mask_file),
                "--debug",
            ]
        )
    )

    for option in reemit_options():
        if option.route_gated:
            continue  # covered by the dedicated base-model gating test below
        assert option.emit_flag in invocation.argv, f"{option.emit_flag} was consumed but not re-emitted"
    strength_index = invocation.argv.index("--video-strength")
    assert invocation.argv[strength_index + 1] == "0.7"
    mask_index = invocation.argv.index("--video-mask-path")
    assert invocation.argv[mask_index + 1] == str(mask_file)


def test_base_model_reemitted_only_on_accepting_routes():
    gated = mlx_gen._resolve_invocation(
        [
            "--model",
            QWEN_MODEL,
            "--base-model",
            QWEN_MODEL,
            "--prompt",
            "a puffin",
        ]
    )
    assert "--base-model" in gated.argv

    wan = mlx_gen._resolve_invocation(_wan_v2v_argv(["--base-model", WAN_MODEL]))
    assert "--base-model" not in wan.argv


def test_router_only_options_never_reach_backend_argv():
    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            QWEN_MODEL,
            "--task",
            "text-to-image",
            "--family",
            "qwen",
            "--i2i-mode",
            "auto",
            "--prompt",
            "a puffin",
        ]
    )

    for option in ROUTER_OPTIONS:
        if option.policy is ForwardPolicy.ROUTER_ONLY:
            for flag in option.flags:
                assert flag not in invocation.argv, f"router-only {flag} leaked to the backend"


def test_debug_flag_reemitted_and_accepted_by_wan_backend_parser():
    from mflux.models.wan.cli import wan_generate

    invocation = mlx_gen._resolve_invocation(_wan_v2v_argv(["--debug"]))

    assert "--debug" in invocation.argv
    args = wan_generate._parser().parse_args(invocation.argv[1:])
    assert args.debug is True


def test_debug_flag_absent_when_not_requested():
    invocation = mlx_gen._resolve_invocation(_wan_v2v_argv([]))
    assert "--debug" not in invocation.argv


def test_metadata_sourced_values_roundtrip(tmp_path):
    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"png")
    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": WAN_MODEL,
                "video_path": "input.mp4",
                "video_strength": 0.65,
                "video_mask_path": str(mask_file),
                "prompt": "replay",
            }
        )
    )

    invocation = mlx_gen._resolve_invocation(["--config-from-metadata", str(metadata_path)])

    strength_index = invocation.argv.index("--video-strength")
    assert invocation.argv[strength_index + 1] == "0.65"
    mask_index = invocation.argv.index("--video-mask-path")
    assert invocation.argv[mask_index + 1] == str(mask_file)
    model_index = invocation.argv.index("--model")
    assert invocation.argv[model_index + 1] == WAN_MODEL
    # The metadata file itself is still forwarded for backend-side replay of unconsumed options.
    assert "--config-from-metadata" in invocation.argv


def test_out_of_range_metadata_video_strength_fails_at_backend_parse(tmp_path):
    from mflux.models.wan.cli import wan_generate

    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": WAN_MODEL,
                "video_path": "input.mp4",
                "video_strength": 1.5,
                "prompt": "replay",
            }
        )
    )

    invocation = mlx_gen._resolve_invocation(["--config-from-metadata", str(metadata_path)])

    strength_index = invocation.argv.index("--video-strength")
    assert invocation.argv[strength_index + 1] == "1.5"
    with pytest.raises(SystemExit):
        wan_generate._parser().parse_args(invocation.argv[1:])


def test_wan_v2v_full_emitted_block_order(tmp_path):
    mask_file = tmp_path / "mask.png"
    mask_file.write_bytes(b"png")

    invocation = mlx_gen._resolve_invocation(
        [
            "--model",
            WAN_MODEL,
            "--video-path",
            "input.mp4",
            "--video-strength",
            "0.7",
            "--video-mask-path",
            str(mask_file),
            "--debug",
            "--steps",
            "4",
            "--prompt",
            "change the ship silhouette",
        ]
    )

    # Pins the emitted block ordering end to end: consumed re-emissions precede forwarded args.
    assert invocation.argv == [
        "mlxgen-generate-wan",
        "--model",
        WAN_MODEL,
        "--video-path",
        "input.mp4",
        "--video-strength",
        "0.7",
        "--video-mask-path",
        str(mask_file),
        "--debug",
        "--steps",
        "4",
        "--prompt",
        "change the ship silhouette",
    ]


def test_unconsumed_options_pass_through_verbatim():
    invocation = mlx_gen._resolve_invocation(_wan_v2v_argv(["--steps", "4", "--fps", "16", "--low-ram"]))

    steps_index = invocation.argv.index("--steps")
    assert invocation.argv[steps_index + 1] == "4"
    fps_index = invocation.argv.index("--fps")
    assert invocation.argv[fps_index + 1] == "16"
    assert "--low-ram" in invocation.argv
