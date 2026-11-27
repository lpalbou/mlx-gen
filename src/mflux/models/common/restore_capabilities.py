"""Declared capability contract for the restoration families behind ``mlxgen upscale``.

Two families share that command - SeedVR2, which restores images and video and can
upscale, and SwiftVR, which restores video in one step at the source resolution. Before
this module the only way to ask "can this restoration model do images?" was to match on a
family string, and the refusals a family owed the user were nine hardcoded
``parser.error`` calls in one CLI. This module makes both of those things data.

WHY A RECORD AND NOT A PROTOCOL OR AN ABC
-----------------------------------------
The video half of the interface already exists and needs declaring, not inventing:
``SeedVR2.restore_video_to_path`` and ``SwiftVR.restore_video_to_path`` share a name and
an identical keyword-only core (``video_path``, ``resolution``, ``output_path``,
``start_seconds``, ``max_frames``, ``color_correction_mode``, ``drop_audio``,
``export_json_metadata``, ``overwrite``, ``validate_health``, ``restore_metadata``,
``enforce_memory_budget``) with identical defaults, both returning ``Path``. They differ
only in family extras.

The image half cannot be a total interface. SeedVR2 pairs the repo-wide in-memory house
name ``generate_image() -> GeneratedImage`` with ``restore_image_to_path``, the
write-to-disk route this record declares because it mirrors ``restore_video_to_path``;
SwiftVR has neither and must not carry a stub that raises. A ``Protocol`` or ABC covering both would therefore be non-total, and there is no
generic consumer to justify one: ``_run_swiftvr_restore`` holds a ``SwiftVR``,
``_run_video_with_fresh_model`` holds a ``SeedVR2``, and ``upscale_main()`` dispatches to
two genuinely different ``main()`` flows. A Protocol here would be documentation with no
call site.

So the route name is a *field* (:attr:`RestoreCapability.route_method`), exactly the way
``task_inference.GenerationCapability.handler_id`` names a handler with a string. The
static guarantee an ABC would give is replaced by a weight-free test asserting
``hasattr(route_class, capability.route_method)`` for every declared row - which checks
the real classes rather than a declaration.

WHAT THIS MODULE DESCRIBES AND WHAT IT DOES NOT ENFORCE
-------------------------------------------------------
:attr:`RestoreCapability.scale_mode` is descriptive. The authoritative 1x rule lives in
``SwiftVRUtil.output_canvas``, which has the source dimensions and already raises the
message naming ``--resolution 1x`` and ``--model seedvr2-3b``. A second copy here would be
a rule with two sources that can disagree. The record describes; ``output_canvas``
enforces; there is one enforcement site.

The same split holds for the route-level guards. ``SwiftVR._assert_supported_options``,
``_assert_quantization_supported`` and ``_assert_clip_length_supported`` are NOT replaced
by :attr:`RestoreFamilyCapabilities.unsupported_options`. The route is a public Python API
callable without argparse and must fail closed on its own; the record is what the CLI
refuses from. Two layers is intentional.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

# SEEDVR2_HANDLES is re-exported (import-as-self) because 0.30.0 published it from this
# module; the definition moved to restore_dispatch to break an import cycle.
from mflux.models.common.cli.restore_dispatch import (
    SEEDVR2_HANDLES as SEEDVR2_HANDLES,
    SWIFTVR_HANDLES,
    _looks_like_seedvr2_directory,
    _looks_like_swiftvr_directory,
    is_seedvr2_handle,
)
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution

# Bumped when a field is added, renamed or given a new meaning. Mirrors the additive
# convention `task_inference.CAPABILITIES_SCHEMA_VERSION` documents for the generate side.
RESTORE_CAPABILITIES_SCHEMA_VERSION = 2

INPUT_IMAGE = "image"
INPUT_VIDEO = "video"
INPUT_KINDS = (INPUT_IMAGE, INPUT_VIDEO)

# Whether a family can be asked for an output size other than the source size.
SCALE_SCALABLE = "scalable"
SCALE_SOURCE_ONLY = "source-only"

RESTORE_FAMILIES = ("seedvr2", "swiftvr")

# The nine refusals SwiftVR owes the user, in the order the CLI's if-chain reported them
# so that a command violating several still names the same first problem. Every message
# states what cannot be honoured, why, and what to use instead (ADR 0002). Eight are the
# CLI's own wording moved verbatim; the ``--image-path`` reason is corrected below.
#
# ``{value}`` is the only interpolation token, and only ``--quantize`` uses it. Callers
# substitute with ``str.replace``, never ``str.format``: a message that ever contained a
# literal brace would otherwise raise while the user is already in an error path.
_SWIFTVR_UNSUPPORTED_OPTIONS: Mapping[str, str] = MappingProxyType(
    {
        # t = 1 is a legal 4a + 1 length, so the pipeline does not refuse it on
        # arithmetic: it plans as one LAST chunk (b = 0, one latent), the codec replicates
        # the final frame to fill the 4-frame encode slice, the decoder emits four frames
        # and the 3-frame head trim leaves exactly one. It runs to completion. Measured on
        # 2026-08-18 across three subjects against SeedVR2's image route on identical
        # input: at 1:1 the frame is acceptable but soft, and under magnification facial
        # features lose structure (eyes flatten into smudges, fine texture waxes over)
        # because the causal autoencoder state is seeded from a replicated frame instead
        # of real temporal context. SeedVR2 resolves the same detail cleanly, so stills
        # route there. Correlation does not separate the two - SwiftVR scores higher on
        # two of three samples while being visibly worse - which is why the committed
        # proof is a contact sheet rather than a number.
        "--image-path": (
            "SwiftVR restores video only. A single frame is a legal clip for its chunk protocol and "
            "does run, but it restores stills measurably worse than SeedVR2 on the same input: "
            "facial features lose structure under magnification and fine texture smooths over, "
            "because the causal autoencoder state is seeded from a replicated frame rather than "
            "real temporal context. Use --video-path for SwiftVR, or --model seedvr2-3b, which is "
            "the supported route for stills."
        ),
        "--quantize": (
            "--quantize {value} does not apply to SwiftVR, which runs only its bf16 source "
            "route. At 8 bits Wan's q8 sensitivity policy spares every quantizable module in this "
            "architecture, so the run would be labelled quantized while staying bf16; at 4 bits the "
            "quantized condition embedder fails inside the Wan timestep projection. Re-run without "
            "--quantize, or use --model seedvr2-3b, which has quantized packages."
        ),
        "--vae-tiling": (
            "--vae-tiling does not apply to SwiftVR. It replaces the Wan 3D VAE with ReAE, which decodes "
            "one latent frame at a time and has no tiling path."
        ),
        "--steps": (
            "--steps does not apply to SwiftVR: restoration is a single forward pass per chunk at a fixed "
            "timestep, with no sampler to step. Use --model seedvr2-3b if you want a multi-step restore."
        ),
        "--temporal-chunk-size": (
            "--temporal-chunk-size does not apply to SwiftVR. That is SeedVR2's memory-chunking axis; SwiftVR uses "
            "its own fixed FIRST/MIDDLE/LAST protocol with a clip length of 4a + 1."
        ),
        "--temporal-chunk-overlap": (
            "--temporal-chunk-overlap does not apply to SwiftVR. That is SeedVR2's memory-chunking axis; SwiftVR uses "
            "its own fixed FIRST/MIDDLE/LAST protocol with a clip length of 4a + 1."
        ),
        "--softness": (
            "--softness does not apply to SwiftVR. It is a SeedVR2 degradation control with no counterpart "
            "in a one-step restoration."
        ),
        "--stepwise-image-output-dir": (
            "--stepwise-image-output-dir does not apply to SwiftVR: one forward pass per chunk leaves no "
            "denoise trajectory to preview."
        ),
        "--seed": (
            "SwiftVR restoration is deterministic - one forward pass at a fixed timestep with no noise - so "
            "multiple seeds would produce identical files. Pass a single --seed, or omit it."
        ),
    }
)

_NO_UNSUPPORTED_OPTIONS: Mapping[str, str] = MappingProxyType({})


class RestoreCapabilityError(ValueError):
    """A restoration request the declared contract cannot honour.

    Subclasses ``ValueError`` on purpose, mirroring ``TaskInferenceError``: both
    restoration CLIs already funnel ``ValueError`` into ``parser.error``, so raising this
    needs no new handling anywhere.
    """


@dataclass(frozen=True)
class RestoreCapability:
    """One (family, input kind) route a restoration model actually offers.

    Attributes:
        id: Stable identifier, ``"<family>.<input_kind>"``.
        input_kind: :data:`INPUT_IMAGE` or :data:`INPUT_VIDEO`.
        route_method: Attribute name of the method on the route class that serves this
            capability. A string, not a callable, so the record stays importable without
            the route and its weights - see the module docstring.
        scale_mode: :data:`SCALE_SCALABLE` or :data:`SCALE_SOURCE_ONLY`. Descriptive;
            the source-size rule is enforced by ``SwiftVRUtil.output_canvas``.
        max_source_frames: Hard ceiling on source frames for a video row, or ``None``
            when the route streams without one.
        color_correction_modes: Exactly the ``--color-correction`` values the route
            accepts. The route guard enforces the same set post-load; the CLI reads this
            field to refuse anything else at parse time, before any weight load.
    """

    id: str
    input_kind: str
    route_method: str
    scale_mode: str
    max_source_frames: int | None = None
    color_correction_modes: tuple[str, ...] = ("wavelet", "lab", "off")

    def allows_source_frame_count(self, frame_count: int) -> bool:
        """Whether this capability accepts a clip of ``frame_count`` source frames.

        Mirrors ``GenerationCapability.allows_video_count``: an image row accepts no
        frame count at all, and an unbounded video row accepts any positive count.
        """
        if self.input_kind != INPUT_VIDEO:
            return False
        if frame_count < 1:
            return False
        return self.max_source_frames is None or frame_count <= self.max_source_frames

    def to_dict(self) -> dict:
        """JSON-ready row."""
        return {
            "id": self.id,
            "input_kind": self.input_kind,
            "route_method": self.route_method,
            "scale_mode": self.scale_mode,
            "max_source_frames": self.max_source_frames,
            "color_correction_modes": list(self.color_correction_modes),
        }


@dataclass(frozen=True)
class RestoreFamilyCapabilities:
    """What one restoration family offers, and what it refuses with which message.

    Attributes:
        schema_version: :data:`RESTORE_CAPABILITIES_SCHEMA_VERSION` at build time.
        family: ``"seedvr2"`` or ``"swiftvr"``.
        label: Human-readable model name for messages and inspection output.
        model_name: Catalog ``model_name``, or ``None`` for a local checkpoint.
        capabilities: One row per supported input kind. A kind that is absent is a kind
            the family does not offer; :attr:`unsupported_options` says why.
        unsupported_options: Ordered option token to actionable refusal message.
    """

    schema_version: int
    family: str
    label: str
    model_name: str | None
    capabilities: tuple[RestoreCapability, ...]
    unsupported_options: Mapping[str, str]

    def capability_for(self, input_kind: str) -> RestoreCapability | None:
        """The row serving ``input_kind``, or ``None`` when the family has none."""
        for capability in self.capabilities:
            if capability.input_kind == input_kind:
                return capability
        return None

    def to_dict(self) -> dict:
        """JSON-ready payload, shaped like ``ModelCapabilities.to_dict``."""
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "label": self.label,
            "model_name": self.model_name,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
            "unsupported_options": dict(self.unsupported_options),
        }


def require_capability(capabilities: RestoreFamilyCapabilities, input_kind: str) -> RestoreCapability:
    """The row serving ``input_kind``, or the family's own refusal for not having one.

    Raises:
        RestoreCapabilityError: If ``input_kind`` is not a known kind, or the family
            declares no capability for it.
    """
    if input_kind not in INPUT_KINDS:
        raise RestoreCapabilityError(
            f"Unknown restoration input kind {input_kind!r}. Use one of: {', '.join(INPUT_KINDS)}."
        )
    capability = capabilities.capability_for(input_kind)
    if capability is not None:
        return capability
    refusal = capabilities.unsupported_options.get(f"--{input_kind}-path")
    if refusal is not None:
        raise RestoreCapabilityError(refusal)
    raise RestoreCapabilityError(
        f"{capabilities.label} does not restore {input_kind} input. It offers: "
        f"{', '.join(row.input_kind for row in capabilities.capabilities) or 'nothing'}."
    )


def is_restore_family_handle(model: str | None, model_path: str | None) -> bool:
    """Whether ``model`` / ``model_path`` positively names a restoration family.

    Positive matchers only. ``classify_restore_family`` cannot be reused here: it answers
    ``"seedvr2"`` for anything it does not recognise so that SeedVR2's resolver owns the
    unknown-handle error, which is right for dispatch and wrong for inspection.
    """
    return _match_restore_family(model, model_path) is not None


def get_restore_capabilities(
    *,
    model: str | None = None,
    model_path: str | None = None,
    family: str | None = None,
) -> RestoreFamilyCapabilities:
    """The declared contract for a restoration handle.

    Args:
        model: A ``--model`` value, or ``None``.
        model_path: A ``--path`` value, or ``None``.
        family: Explicit family override, the escape hatch for a local checkpoint whose
            handle names nothing. Takes precedence over ``model`` / ``model_path``.

    Raises:
        RestoreCapabilityError: If ``family`` is not a restoration family, or neither
            argument positively names one.
    """
    if family is not None:
        normalized_family = family.strip().lower()
        if normalized_family not in RESTORE_FAMILIES:
            raise RestoreCapabilityError(
                f"Unknown restoration family {family!r}. Use one of: {', '.join(RESTORE_FAMILIES)}."
            )
        if normalized_family == "swiftvr":
            return _swiftvr_capabilities()
        return _seedvr2_capabilities(_seedvr2_variant(model))

    matched = _match_restore_family(model, model_path)
    if matched == "swiftvr":
        return _swiftvr_capabilities()
    if matched == "seedvr2":
        return _seedvr2_capabilities(_seedvr2_variant(model))
    raise RestoreCapabilityError(
        f"{model!r} does not name a restoration model. Use seedvr2, seedvr2-3b, seedvr2-7b, "
        "seedvr2-7b-sharp, swiftvr or swiftvr-5b, or pass --family seedvr2 / --family swiftvr "
        "for a local checkpoint."
    )


def _match_restore_family(model: str | None, model_path: str | None) -> str | None:
    """Positively classify a handle, or ``None`` when it names no restoration family."""
    normalized = model.strip().lower() if model else ""
    if normalized in SWIFTVR_HANDLES:
        return "swiftvr"
    if is_seedvr2_handle(model):
        return "seedvr2"

    # A local checkpoint is identified by its contents, which is far stronger evidence
    # than a path substring: no other family in the catalog ships reae.safetensors.
    for candidate in (model_path, model):
        if _looks_like_swiftvr_directory(candidate):
            return "swiftvr"
        if _looks_like_seedvr2_directory(candidate):
            return "seedvr2"
    return None


def _seedvr2_variant(model: str | None) -> str:
    """Which SeedVR2 variant a handle names, for the label only.

    ``seedvr2-7b`` and ``seedvr2-7b-sharp`` share ``model_name``
    ``"ByteDance-Seed/SeedVR2-7B"``, so the alias is the only trustworthy discriminator -
    the same rule ``SeedVR2VideoRestorePlan._variant_name`` already applies. An unprovable handle degrades
    to ``"3b"``, the resolver's own default, which is a label difference and never a
    capability difference: all three variants declare identical rows.
    """
    normalized = ConfigResolution.identifying_name(model.strip()).lower() if model else ""
    if "7b-sharp" in normalized or "7b_sharp" in normalized:
        return "7b-sharp"
    if "7b" in normalized:
        return "7b"
    return "3b"


@lru_cache(maxsize=None)
def _seedvr2_capabilities(variant: str) -> RestoreFamilyCapabilities:
    """SeedVR2's rows: images and video, both scalable, nothing family-level refused.

    ``unsupported_options`` is empty, and that is the honest declaration rather than an
    omission. SeedVR2's CLI rules (``--start-seconds`` / ``--max-frames`` are image-only,
    ``--vae-tiling`` is not supported for video, chunk-size bounds) are argument-shape
    validation for a particular request, not statements about what the family can do.
    """
    labels = {"3b": "SeedVR2 3B", "7b": "SeedVR2 7B", "7b-sharp": "SeedVR2 7B Sharp"}
    configs = {
        "3b": ModelConfig.seedvr2_3b,
        "7b": ModelConfig.seedvr2_7b,
        "7b-sharp": ModelConfig.seedvr2_7b_sharp,
    }
    model_config = configs.get(variant, ModelConfig.seedvr2_3b)()
    return RestoreFamilyCapabilities(
        schema_version=RESTORE_CAPABILITIES_SCHEMA_VERSION,
        family="seedvr2",
        label=labels.get(variant, "SeedVR2"),
        model_name=model_config.model_name,
        capabilities=(
            RestoreCapability(
                # Declares the write-to-disk route, mirroring the video row, so a caller
                # selects a route by input kind alone. ``generate_image`` remains the
                # in-memory house entry point that this route adapts.
                id="seedvr2.image",
                input_kind=INPUT_IMAGE,
                route_method="restore_image_to_path",
                scale_mode=SCALE_SCALABLE,
            ),
            RestoreCapability(
                id="seedvr2.video",
                input_kind=INPUT_VIDEO,
                route_method="restore_video_to_path",
                scale_mode=SCALE_SCALABLE,
                # The route streams chunk by chunk and carries no rotary ceiling, so
                # there is no frame count it refuses on length alone.
                max_source_frames=None,
            ),
        ),
        unsupported_options=_NO_UNSUPPORTED_OPTIONS,
    )


@lru_cache(maxsize=None)
def _swiftvr_capabilities() -> RestoreFamilyCapabilities:
    """SwiftVR's row: video only, source resolution only, with nine refusals.

    There is deliberately no image row. The chunk arithmetic for ``t = 1`` closes exactly
    (see the ``--image-path`` message), but nothing has measured the output, and ADR 0001
    does not accept an operating point on arithmetic alone. Flipping this later is a row
    addition plus deleting one entry from :data:`_SWIFTVR_UNSUPPORTED_OPTIONS`; the CLI
    needs no edit because its refusals are driven from this record.
    """
    # Imported here, not at module scope: `mflux.models.swiftvr` pulls the route, mlx and
    # numpy, and this module is an inspection surface that must stay cheap to import.
    from mflux.models.swiftvr.variants.upscale.swiftvr_util import SwiftVRUtil

    model_config = ModelConfig.swiftvr()
    rope_max_seq_len = int((model_config.transformer_overrides or {}).get("rope_max_seq_len", 1024))
    return RestoreFamilyCapabilities(
        schema_version=RESTORE_CAPABILITIES_SCHEMA_VERSION,
        family="swiftvr",
        label="SwiftVR 5B",
        model_name=model_config.model_name,
        capabilities=(
            RestoreCapability(
                id="swiftvr.video",
                input_kind=INPUT_VIDEO,
                route_method="restore_video_to_path",
                scale_mode=SCALE_SOURCE_ONLY,
                # Derived, never copied: the ceiling moves with the catalog's rotary
                # table, and a literal here would start lying the day that changes.
                max_source_frames=SwiftVRUtil.max_supported_source_frames(rope_max_seq_len),
                # The route guard refuses everything else; declaring more would let a
                # host offer a flag that only fails after the 5B weight load.
                color_correction_modes=("off",),
            ),
        ),
        unsupported_options=_SWIFTVR_UNSUPPORTED_OPTIONS,
    )
