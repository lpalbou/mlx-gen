# Proposed: publish each route's accepted options, derived from its parser

## Metadata

- Created: 2026-09-06
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md).
- ADR impact: none. A derived list is runtime truth; a hand-maintained one would not be.

## Context

The capability schema declares some options and not others, so an embedding host still needs
per-family knowledge to know which flags a route accepts. Raised twice by the BlackPixel host: a
"Low RAM mode" toggle sent `--low-ram` to MiniMax-H3, which rejected it at argparse. Item 0120
closed that specific case by accepting `--low-ram` on that route, but the general problem stands for
`--solver` (Wan-only) and for every option nobody has thought about yet.

Adding one `supports_<option>` boolean per case does not scale and mis-declares routes that were
never considered: `supports_flow_shift` had to be renamed from `supports_video_shift` in 0.35.0 for
exactly that reason, because a field named after one family's spelling read `false` on the family
that used the other.

## Current code reality

- `GenerationCapability` publishes per-option booleans, added case by case.
- `_Route` (`src/mflux/cli/mlx_gen.py`) binds `target_main`, not a parser, so the router cannot
  currently reach a backend's option surface.
- 4 of ~13 generate backends already expose a module-level parser factory
  (`wan_generate.py`, `minimax_h3_generate.py`, plus the bonsai and ernie pattern). The rest build
  `CommandLineParser` inside `main()`; extracting a factory is mechanical and non-behavioral.
- The honest surface is the router's options plus the backend's, because `ROUTER_OPTIONS` consumes
  and renames some (`--image` becomes the route's own image argument) and `_route_accepts_base_model`
  gates `--base-model` per route.
- `_route_for_plan` covers 12 handler ids, not one per backend, and 4 of them already expose a parser
  factory, so the extraction is 8 modules rather than the whole catalog.
- Layering blocks it today: the route table lives in `src/mflux/cli/mlx_gen.py`, which imports
  `task_inference`, so a capability row cannot reach a route's parser without moving that table to a
  neutral module first. Do that before the extraction.
- 0.35.0 ships a payload-level `universal_options` naming the options every generate route accepts.
  That is the seed of this item: per-route `accepted_options` supersedes it, and `--solver` discovery
  arrives with it.

## Scope

1. Give `_Route` a `parser_factory`, and extract a module-level `_parser()` in the backends that
   lack one, changing no behavior.
2. Publish `accepted_options: tuple[str, ...]` on each row, derived at capability time from the
   router table plus that parser.
3. A test asserting every published name actually parses on its route, so the list cannot drift.

## Non-goals

Migrating the existing per-option booleans onto a descriptor list. They answer a different question
(`accepted_options` says "may I send this flag"; `supports_flow_shift` says "is this the same control
I have elsewhere") and both are wanted.

## Validation

For every route, the published set parses and any option outside it is rejected, checked by the
drift test rather than by hand.
