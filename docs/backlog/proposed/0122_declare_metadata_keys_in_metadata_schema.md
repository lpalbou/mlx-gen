# Proposed: declare runtime metadata keys in `metadata_schema.py`

## Metadata

- Created: 2026-09-06
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: none.
- ADR impact: a declared metadata surface would be a candidate ADR once the shape settles.

## Context

`src/mflux/utils/metadata_schema.py` is a stub: a `VERSION` constant and nothing else. Every metadata
key a run writes - the `audio_*` block MiniMax-H3 added, `video_health`, `runtime_memory`,
`requested_frames`, the Wan `svi_*` and `context_image_paths` families - exists only as a literal at
its write site. A host consuming metadata has no declared surface to read against, and no way to tell
a key that is part of the contract from one that happens to be present.

Raised by the BlackPixel host contract review (item 0120), which noted the `audio_*` keys
specifically. Declaring only those would be inconsistent, since nothing else is declared either;
this item is the general version.

## Current code reality

- `metadata_schema.py` is 6 lines, `VERSION = 2`, no key declarations.
- Keys are written across `src/mflux/utils/video_util.py`, `generated_video.py`, `generated_image.py`
  and each variant's `extra_metadata`.
- `mlxgen capabilities` publishes the route surface but says nothing about the metadata surface.

## Scope

1. Enumerate the keys actually written today, per artifact kind, from the write sites.
2. Declare them with type and meaning, and mark which are guaranteed versus best-effort (the
   `audio_*` block is only written when the route generates audio; `video_health` only with
   validation enabled).
3. A test that fails when a write site introduces a key the schema does not declare.

## Non-goals

Changing any key's name or value, and versioning the metadata format beyond the existing `VERSION`.

## Validation

The declared set matches what a real run of each artifact kind writes, enforced by the drift test.
