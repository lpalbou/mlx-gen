# Runtime contract hardening backlog track

## Status
Completed

## Purpose
Preserve the completed 2026-06-29 adversarial validation band for runtime correctness, progress
semantics, capability truthfulness, and quality-sensitive model math.

## Items
- `../../completed/0067_progress_event_contract_hardening.md`: hardened progress tasks and
  terminal event truthfulness across image and video routes.
- `../../completed/0068_qwen_control_route_hardening.md`: fixed the Qwen control no-LoRA
  crash and aligned control-route negative-prompt handling with the base Qwen contract.
- `../../completed/0069_zimage_cfg_and_inpaint_repair.md`: corrected Z-Image CFG math and
  refreshed the published native-inpaint proof surface.
- `../../completed/0070_canonical_capability_identity_for_variant_routes.md`: closed the
  variant-route spoofing leaks for local/custom and remote-looking prepared ids.

## Reading order
1. `../../completed/0067_progress_event_contract_hardening.md`
2. `../../completed/0068_qwen_control_route_hardening.md`
3. `../../completed/0069_zimage_cfg_and_inpaint_repair.md`
4. `../../completed/0070_canonical_capability_identity_for_variant_routes.md`

## Governing ADRs
None identified after review.

## Scope
Runtime callback/task semantics, route-level crash fixes, capability exposure truthfulness, and
quality-preserving validation tied to the confirmed 2026-06-29 probes.

## Non-goals
Do not broaden this track into generic memory-reduction work, model-family expansion, or speculative
quality tuning unrelated to the confirmed bugs.

## Notes for future agents
Use the real validation artifacts under `validation_outputs/runtime_contracts_2026_06_29/` and the
written report at `docs/assets/validation/runtime-contracts-2026-06-29/runtime_contracts_report.md`
as the baseline evidence for this band.
