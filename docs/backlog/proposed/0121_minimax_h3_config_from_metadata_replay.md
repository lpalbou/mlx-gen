# Proposed: `--config-from-metadata` replay for MiniMax-H3

## Metadata

- Created: 2026-09-06
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none.
- ADR impact: none.

## Context

Every route built on the shared CLI parser accepts `--config-from-metadata`, which reloads a saved
run's settings from its metadata JSON and regenerates it. MiniMax-H3 declares its own parser and does
not, so an H3 clip is the one video artifact in the catalog that cannot be replayed from its own
sidecar. Raised by the BlackPixel host contract review (item 0120) as a missed issue.

## Current code reality

- `src/mflux/models/minimax_h3/cli/minimax_h3_generate.py` builds a standalone `argparse` parser and
  has no `--config-from-metadata`; `src/mflux/cli/parser/parsers.py` implements the flag and its
  metadata merge for the routes that use the shared parser (see `namespace.base_model` and the
  seeding around `prior_gen_metadata`).
- H3 metadata already records what a replay needs: `steps`, `video_shift`, `audio_shift`,
  `num_inference_steps`, `frames`, `requested_frames`, `width`, `height`, `seed`, `prompt`,
  `image_path`, `lora_paths`, `lora_scales`, plus the `audio_*` block.
- The prompt is stored composed (all three sections in one string), so a naive replay that also
  re-sends `--soundscape`/`--music` would now be refused by the duplicate-section guard added in
  0120. Replay must pass the composed prompt alone.

## Scope

1. Accept `--config-from-metadata` on the H3 CLI, seeding the same fields the shared parser seeds,
   with explicit CLI arguments winning over the file.
2. Replay the composed prompt as `--prompt` without re-deriving the section flags.
3. A round-trip test: generate with a tiny config, replay from the metadata, assert the resolved plan
   matches (canvas, frames, steps, shifts, seed).

## Non-goals

Refactoring the H3 CLI onto the shared parser wholesale; the two differ in enough places
(`--soundscape`, `--music`, `--audio-shift`, no `--fps`, no `--guidance`) that the merge is its own
piece of work.

## Validation

A replayed run reproduces the original clip byte-for-byte at the same seed.
