# Proposed: publish the duration and sampling fields on the other model families

## Metadata

- Created: 2026-09-06
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none.
- ADR impact: none; the fields already exist and follow the additive convention.

## Context

Item 0120 added `min_frames`, `max_frames`, `frame_multiple`, `frame_remainder`, `frame_rounding`,
`output_fps`, `supports_video_shift` and the `default_*` block to `GenerationCapability` and
populated them for MiniMax-H3. Every other family still publishes the falsy defaults, so a host
reading `output_fps: null` on a Wan row cannot tell "this route has no fixed rate" from "nobody
filled this in yet".

## Current code reality

- The fields default to `None`/`False` and are serialized by `GenerationCapability.to_dict`.
- Wan has real values for most of them: a `4n + 1` frame contract, per-entry canvas and step
  defaults in `transformer_overrides`, and `--flow-shift` as its shift control.
- Wan's shift control is named `--flow-shift`, not `--video-shift`; a Wan row should either publish
  a `supports_flow_shift` sibling or the field should be generalised. Decide before populating,
  because the name is part of the published contract.
- SeedVR2 and SwiftVR already carry their own frame fields on `RestorationCapability`; the generation
  fields are the gap.

## Also unfilled after 0.35.0

The precision and memory fields added in the same pass (`weight_precision`,
`unquantized_weights_bytes`, `recommended_quantize`, `validated_quantization_bits`, `measured_peak`)
are likewise MiniMax-H3-only. Wan and SeedVR2 have measured figures in their docs that could fill
them. Until they do, a null there reads as unknown rather than as absent.

## Scope

1. Decide the shift-field naming across families (per-control names versus one generalised field).
2. Populate the duration and default fields for Wan T2V/I2V/TI2V and Bernini from their configs.
3. Leave image families' frame fields null, which is correct for them, and say so in `docs/api.md`.

## Non-goals

A schema bump: the fields already exist as of 13.

## Validation

A test per family asserting the published defaults match the entry's own configuration, so the row
cannot drift from the catalog.
