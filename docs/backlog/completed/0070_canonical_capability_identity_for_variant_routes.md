# Completed: Canonical capability identity for variant routes

## Metadata
- Created: 2026-06-29
- Status: Completed
- Completed: 2026-06-29

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
The first confirmed spoof probe showed that a custom local path whose name merely contained
`qwen-image-8bit` could incorrectly advertise base Qwen control and control-inpaint support when
combined with `--base-model qwen-image --family qwen`. A broader adversarial review then confirmed
the same raw-path poisoning pattern on Qwen edit, Z-Image Turbo, FLUX.2 Klein base, and FIBO Edit
variant predicates, plus a second authority leak where shared `family=` overrides can lie about a
known canonical model family.

## Historical pre-fix reality
- `src/mflux/task_inference.py` currently resolves a model config when possible, but then builds
  one fuzzy `model_key` string that mixes canonical identity with raw path tokens and does not
  preserve resolution provenance in the shared identity layer.
- Variant-sensitive predicates such as `_supports_qwen_base_control()`, `_is_qwen_edit()`,
  `_is_qwen_edit_plus()`, `_is_z_image_turbo()`, `_is_flux2_klein_base()`, and `_is_fibo_edit()`
  still trust raw-path substrings inside that composite key.
- Shared `family=` overrides can still force false capability contracts for known canonical models
  or unresolved local paths unless identity resolution fail-closes the conflict.
- Confirmed spoof outcomes already include:
  - `qwen-image-8bit-custom` + base `qwen-image` incorrectly exposes `qwen.control*`
  - `qwen-image-edit-custom` + base `qwen-image` incorrectly exposes edit-only capabilities
  - `z-image-turbo-custom` + base `z-image` incorrectly exposes `z-image.inpaint`
  - `flux2-klein-base-custom` + base `flux2-klein-4b` incorrectly exposes `flux2.outpaint`
  - `fibo-edit-custom` + base `fibo` incorrectly collapses unified capabilities to `[]`

## Problem
Capability exposure and route selection are not truthful for local/custom model paths or forced
family overrides. Once the runtime has enough evidence to resolve or reject identity, raw path
naming and family-only overrides should not be able to mutate variant-sensitive capabilities.

## What we want to do
Make variant-sensitive capability and route selection depend on a provenance-aware shared identity
layer once the runtime has enough information to resolve it, and fall back conservatively when
exact identity cannot be proven.

## Why
The capabilities surface is part of the public contract. If it lies, users can select unsupported
routes, get the wrong defaults, or lose supported routes for the wrong reason.

## Requirements
- Variant-sensitive capabilities must be shaped from canonical identity when `ModelConfig` is
  resolved.
- Identity provenance must stay in the shared routing-resolution layer, not in the reusable
  `ModelConfig` descriptor.
- Raw model-path substrings may help infer family only when canonical identity is absent; they must
  not override resolved variant identity.
- Local path naming alone must never unlock Qwen control, Qwen edit, Z-Image Turbo, FLUX.2 base,
  or FIBO Edit-specific routing.
- Shared `family=` overrides must fail closed when they conflict with a resolved canonical model or
  when they are not enough to prove a concrete backend config.
- Ambiguous custom paths should fail closed or advertise only conservative capabilities.

## Suggested implementation
Separate family inference from variant/capability proof in a provenance-aware identity object. Keep
fuzzy path heuristics only for family discovery, gate variant-sensitive routes from trusted
aliases/canonical names or explicit base-model evidence, and keep exact validated exceptions such
as `qwen.control*` on a narrow allowlist.

## Scope
- Shared identity resolution and variant-sensitive predicates in `task_inference.py`.
- Provenance-aware config-resolution handoff into route planning and capabilities.
- Focused regression tests for exact supported identity versus spoofed local/custom names.
- A small public capabilities and routing proof set for the previously reproduced spoof cases.

## Non-goals
- Do not turn this item into a full model-registry migration by default.
- Do not broaden supported variant routes for unofficial or merely similar-looking local models.
- Do not rewrite unrelated family predicates unless the same poisoning pattern is proven there.

## Dependencies and related tasks
- Completed [0020 generation capability contract](0020_generation_capability_contract.md)
- Proposed [0058 model profile registry authority](../proposed/0058_model_profile_registry_authority.md)
- Completed [0068 Qwen control route hardening](0068_qwen_control_route_hardening.md)

## Expected outcomes
- Spoofed local names no longer advertise or select unsupported variant-sensitive routes.
- Exact supported identities continue to expose the correct route surface.
- The public capabilities contract becomes more truthful without a broad registry migration.

## Validation
- Focused task-inference or CLI-router regression tests for Qwen, Z-Image, FLUX.2, and FIBO spoof
  cases.
- One public `mlxgen capabilities` proof showing previously reproduced spoof cases are corrected.
- Sanity checks that the real supported prepared packages still expose the expected routes.

## Progress checklist
- [x] Separate family heuristics from variant/capability proof.
- [x] Replace raw-path variant poisoning with provenance-aware canonical identity logic.
- [x] Fail closed on conflicting or family-only shared identity overrides.
- [x] Add spoof and supported-identity regression coverage across affected families.
- [x] Re-run public capabilities proofs one case at a time.

## Guidance for the implementing agent
Prefer the narrowest truthful identity rule that can be defended from current code reality. Keep
broader registry authority work in item 0058 unless this fix proves it is unavoidable.

## Completion report

- Date: 2026-06-29
- Original path: `docs/backlog/planned/runtime_contracts/0070_canonical_capability_identity_for_variant_routes.md`
- Final path: `docs/backlog/completed/0070_canonical_capability_identity_for_variant_routes.md`
- Summary: Variant-sensitive routing is now gated by provenance-aware identity instead of raw path
  poisoning. Spoofed local/custom paths and remote-looking `AbstractFramework/...-custom` ids stay
  conservative, while exact supported prepared ids keep their expected route surface.
- Implementation: `config_resolution.py` now trusts only exact official prepared repo-name patterns
  instead of every `AbstractFramework/<anything>` handle, and `task_inference.py` no longer lets
  pre-resolved derived `ModelConfig` objects silently upgrade an untrusted id into a trusted
  variant identity on the CLI/public planning path.
- Validation: `tests/resolution/test_config_resolution.py`, `tests/test_task_inference.py`, and the
  focused CLI capabilities regression in `tests/cli/test_mlx_gen_router.py` passed. The public
  artifact `validation_outputs/runtime_contracts_2026_06_29/capabilities_spoof_proof.json` shows
  both positive exact-package controls and corrected spoof negatives.
- Evidence report: [runtime_contracts_report.md](../../assets/validation/runtime-contracts-2026-06-29/runtime_contracts_report.md)
- Residual risk: the broader model-registry-authority design stays deferred to proposed item 0058.
  This fix is intentionally narrow: it hardens truthful current routes without turning the whole
  runtime into a registry migration.
