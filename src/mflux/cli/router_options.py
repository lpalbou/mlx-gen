from dataclasses import dataclass, field
from enum import Enum

from mflux.cli.parser.parsers import image_strength_value


class ForwardPolicy(Enum):
    # Consumed for routing only; must never reach a backend.
    ROUTER_ONLY = "router_only"
    # Consumed by the router parser and re-emitted as `<emit_flag> <value>` when set.
    REEMIT_VALUE = "reemit_value"
    # Consumed store_true flag, re-emitted as a bare `<emit_flag>` when set.
    REEMIT_FLAG = "reemit_flag"
    # Consumed and re-emitted by a dedicated helper (route-dependent argument names or
    # validated transforms); `emitter` names the owning helper for auditability.
    TRANSFORMED = "transformed"


@dataclass(frozen=True)
class RouterOption:
    flags: tuple[str, ...]
    dest: str
    policy: ForwardPolicy
    kwargs: dict = field(default_factory=dict)
    # Metadata sidecar keys that may backfill this option when absent from argv.
    metadata_keys: tuple[str, ...] = ()
    # Canonical spelling used on re-emission (REEMIT_* policies).
    emit_flag: str | None = None
    # Emission position among REEMIT_* options (lower first); None for non-emitting policies.
    emit_order: int | None = None
    # Emission gated on the selected route accepting the flag (base-model style allowlist).
    route_gated: bool = False
    # Emitted value is `route.model_override or args.<dest>` (model only).
    use_route_model_override: bool = False
    # TRANSFORMED: name of the helper that owns re-emission, for the completeness audit.
    emitter: str | None = None


# Single source of truth for every option the `mlxgen generate` router CONSUMES. Everything else
# passes through parse_known_args untouched. Order defines parser construction (help listing).
# NOT covered by this table: options the router INJECTS without consuming (--controlnet-model
# conflict/injection logic lives in _resolve_invocation).
ROUTER_OPTIONS: tuple[RouterOption, ...] = (
    RouterOption(
        flags=("--model", "-m"),
        dest="model",
        policy=ForwardPolicy.REEMIT_VALUE,
        kwargs={"type": str, "help": "Model alias, Hugging Face repo, or local model path."},
        metadata_keys=("model",),
        emit_flag="--model",
        emit_order=0,
        use_route_model_override=True,
    ),
    RouterOption(
        flags=("--base-model",),
        dest="base_model",
        policy=ForwardPolicy.REEMIT_VALUE,
        kwargs={
            "type": str,
            "default": None,
            "help": "Base model hint for custom repositories or local paths.",
        },
        metadata_keys=("base_model",),
        emit_flag="--base-model",
        emit_order=1,
        route_gated=True,
    ),
    RouterOption(
        flags=("--family",),
        dest="family",
        policy=ForwardPolicy.ROUTER_ONLY,
        kwargs={
            "choices": ["qwen", "flux2", "fibo", "z-image", "ernie-image", "wan", "bonsai"],
            "default": None,
            "help": "Override model-family detection for local paths or custom repo names.",
        },
    ),
    RouterOption(
        flags=("--debug",),
        dest="debug",
        policy=ForwardPolicy.REEMIT_FLAG,
        kwargs={
            "action": "store_true",
            "help": "Enable debug logging for internal generation details such as LoRA fusion targets.",
        },
        emit_flag="--debug",
        emit_order=4,
    ),
    RouterOption(
        flags=("--task",),
        dest="task",
        policy=ForwardPolicy.ROUTER_ONLY,
        kwargs={
            "choices": [
                "auto",
                "text-to-image",
                "txt2img",
                "image-to-image",
                "img2img",
                "edit",
                "text-to-video",
                "txt2vid",
                "t2v",
                "image-to-video",
                "img2vid",
                "i2v",
                "video-to-video",
                "vid2vid",
                "v2v",
            ],
            "default": "auto",
            "help": "Override automatic routing. Default: auto.",
        },
    ),
    RouterOption(
        flags=("--i2i-mode",),
        dest="i2i_mode",
        policy=ForwardPolicy.ROUTER_ONLY,
        kwargs={
            "choices": ["auto", "latent", "img2img", "edit", "edit-reference", "multi", "multi-reference"],
            "default": "auto",
            "help": (
                "Internal image-to-image mode. Default: auto. Use latent/img2img for image-strength "
                "variation, edit for instruction/reference edits, or multi-reference for two or more input images."
            ),
        },
    ),
    RouterOption(
        flags=("--image", "--input-image", "-i"),
        dest="images",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "action": "append",
            "default": [],
            "help": (
                "Input image. Use one image for image-to-image or Wan first-frame image-to-video. "
                "Repeat only for multi-reference image-to-image."
            ),
        },
        metadata_keys=("image_paths", "image_path"),
        emitter="_collect_images/route.image_argument",
    ),
    RouterOption(
        flags=("--images", "--input-images", "--image-paths"),
        dest="image_groups",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "nargs": "+",
            "action": "append",
            "default": [],
            "help": "One or more input images for image-to-image reference/edit modes.",
        },
        metadata_keys=("image_paths",),
        emitter="_collect_images/route.image_argument",
    ),
    RouterOption(
        flags=("--image-path",),
        dest="image_path",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "default": None,
            "help": "Compatibility alias for a single input image, including Wan first-frame image-to-video.",
        },
        metadata_keys=("image_path",),
        emitter="_collect_images/route.image_argument",
    ),
    RouterOption(
        flags=("--video", "--input-video"),
        dest="videos",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "action": "append",
            "default": [],
            "help": "Input source video for plain prompt-guided video-to-video routes.",
        },
        metadata_keys=("video_paths", "video_path"),
        emitter="_collect_videos/route.video_argument",
    ),
    RouterOption(
        flags=("--video-path",),
        dest="video_path",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "default": None,
            "help": "Compatibility alias for a single source video on plain video-to-video routes.",
        },
        metadata_keys=("video_path",),
        emitter="_collect_videos/route.video_argument",
    ),
    RouterOption(
        flags=("--video-strength",),
        dest="video_strength",
        policy=ForwardPolicy.REEMIT_VALUE,
        kwargs={
            "type": image_strength_value,
            "default": None,
            "help": (
                "Denoising strength for plain video-to-video routes. Higher values allow larger appearance changes."
            ),
        },
        metadata_keys=("video_strength",),
        emit_flag="--video-strength",
        emit_order=2,
    ),
    RouterOption(
        flags=("--video-mask-path",),
        dest="video_mask_path",
        policy=ForwardPolicy.REEMIT_VALUE,
        kwargs={
            "default": None,
            "help": (
                "Static image mask for masked video-to-video. White marks the region the model may change; "
                "black regions are preserved exactly from the source video."
            ),
        },
        metadata_keys=("video_mask_path",),
        emit_flag="--video-mask-path",
        emit_order=3,
    ),
    RouterOption(
        flags=("--reframe-padding",),
        dest="reframe_padding",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "default": None,
            "help": (
                "Generative reframe request: CSS-style top,right,bottom,left padding such as "
                "'0,25%%,0,25%%'. Supported edit models redraw into the larger canvas; this is not "
                "masked outpainting and does not preserve source pixels exactly."
            ),
        },
        metadata_keys=("reframe_padding",),
        emitter="_reframe_forwarded_argv",
    ),
    RouterOption(
        flags=("--outpaint-padding", "--image-outpaint-padding"),
        dest="outpaint_padding",
        policy=ForwardPolicy.TRANSFORMED,
        kwargs={
            "default": None,
            "help": (
                "Canvas outpaint request: CSS-style top,right,bottom,left padding such as "
                "'0,25%%,0,25%%'. Qwen Image Edit variants use generative canvas expansion with "
                "adaptive source restoration. FLUX.2 strict outpaint requires a base Klein model "
                "and uses source-locked denoising instead of generative reframe."
            ),
        },
        metadata_keys=("outpaint_padding", "image_outpaint_padding"),
        emitter="_outpaint_forwarded_argv",
    ),
)


def add_router_options(parser) -> None:
    for option in ROUTER_OPTIONS:
        kwargs = dict(option.kwargs)
        if option.dest != option.flags[0].lstrip("-").replace("-", "_"):
            kwargs["dest"] = option.dest
        parser.add_argument(*option.flags, **kwargs)


def reemit_options() -> tuple[RouterOption, ...]:
    emitting = [option for option in ROUTER_OPTIONS if option.emit_order is not None]
    return tuple(sorted(emitting, key=lambda option: option.emit_order))
