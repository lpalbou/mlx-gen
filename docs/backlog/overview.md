# Backlog Overview

## Project summary

MLX-Gen is an independent Apple Silicon image and video generation package derived from
[mflux](https://github.com/filipstrand/mflux). The backlog tracks model integration work,
compatibility fixes, release-readiness work, and follow-up investigations that should survive
outside chat history.

## Counts

| State | Count |
| --- | ---: |
| Planned | 14 |
| Proposed | 16 |
| Completed | 53 |
| Deprecated | 1 |
| Recurrent | 1 |

Counts are item files (recounted 2026-07-15), including topic-track items under
`planned/memory/`; the completed `planned/runtime_contracts/` track holds only its index.

## Completed runtime contract hardening band

The 2026-06-29 serial validation pass converted several previously theoretical concerns into
concrete shipped bugs with preserved repro commands and artifacts. Items 0067-0070 are now
completed with focused regressions, one-at-a-time real proofs, and a shared validation report.

1. Finished [progress event contract hardening](completed/0067_progress_event_contract_hardening.md)
   so masked image routes, SeedVR2 upscale/restore routes, and terminal success semantics are
   truthful for subscribers.
2. Finished [Qwen control route hardening](completed/0068_qwen_control_route_hardening.md)
   so public control-inpaint no longer crashes on the no-LoRA path and base control routes honor
   the base-Qwen blank-negative contract.
3. Finished [Z-Image CFG and inpaint repair](completed/0069_zimage_cfg_and_inpaint_repair.md)
   so guided denoise math is correct and native inpaint has refreshed same-case evidence after the
   formula fix.
4. Finished [canonical capability identity for variant routes](completed/0070_canonical_capability_identity_for_variant_routes.md)
   so spoofed local paths and remote-looking prepared ids cannot advertise unsupported
   variant-sensitive routes.

## Completed Python runtime multi-output surface

MLX-Gen now owns the shared Python multi-output execution contract for the unified
`mlxgen generate` families. Completed item 0071 records the route-resolved runtime wrapper,
overwrite/collision behavior, and the reuse-vs-reload evidence across image and video routes.

1. Finished [Python runtime serial multi-output reuse](completed/0071_python_runtime_serial_multi_output_reuse.md)
   so embedding apps can call `load_generation_model(...).generate_outputs(...)` instead of
   rebuilding seed loops, output naming, and per-seed save handling around direct model classes.

## Completed video-edit boundary, public route, and reference-proof groundwork

The 2026-07-03 video-edit pass now has three concrete outcomes: the public boundary is coherent,
the first plain public Wan `video-to-video` route is shipped, and the bounded upstream VACE
reference proof is preserved. The remaining result is intentionally split: plain source-video
editing is now real, but the local `Wan2.1-VACE-1.3B` MPS proof was still not good enough to
promote richer VACE conditioning into the public runtime yet.

1. Finished [reader-first video workflow boundary and generative video-edit contract](completed/0072_reader_first_video_workflow_boundary_and_generative_video_edit_contract.md)
   so the current CLI, docs, and ADR surfaces agree that prompt-guided source-video editing
   belongs to `mlxgen generate` and starts with plain `video-to-video`.
2. Finished [Wan VACE reference validation harness and bounded source cases](completed/0073_wan_vace_reference_validation_harness_and_bounded_source_cases.md)
   so future agents have preserved local artifacts, wall time, and memory evidence instead of
   generic “VACE exists upstream” claims. The bounded portrait and ship cases ran, but they did
   not pass a release-quality visual bar on this host.
3. Finished [Wan plain generative video-to-video route](completed/0074_wan_plain_generative_video_to_video_route.md)
   so `mlxgen generate` now owns a bounded public Wan source-video edit route on `Wan2.2-T2V-A14B`
   with truthful config gating, `unipc`-only public solver support, focused tests, and a preserved
   ship-edit proof bundle.
4. Finished [Wan masked video-to-video via latent compositing](completed/0076_wan_masked_video_to_video_latent_compositing.md)
   so `--video-mask-path` locks preserved regions to the source at the measured H.264 floor
   (drift 1.7 vs 14.9 unmasked on the conference proof), with UniPC scheduler-state compositing,
   fail-closed contracts, replayable metadata, and an in-repo proof bundle under
   `docs/assets/examples/conference-masked-v2v/`.
5. Finished [Wan V2V fps resampling and audio copy-through](completed/0077_wan_v2v_fps_resampling_and_audio_copy_through.md)
   so video-to-video keeps real-time speed on any source fps (decode-time resampling with a
   drift-based skip tolerance that leaves matching-fps runs bit-identical) and copies source
   audio onto the output best-effort with the outcome recorded in metadata; the proof's first
   run exposed and fixed a `-shortest` mux bug that dropped trailing frames of short clips,
   pinned by a regression test, with the model-backed proof under
   `validation_outputs/fps_audio_proof_2026_07_05/`.
6. Finished [router descriptors, Wan decomposition phase 1, MaskUtil, metadata schema version](completed/0078_router_descriptors_wan_decomposition_mask_util_schema_version.md)
   so the `mlxgen generate` option surface is single-sourced with completeness and round-trip
   tests (fixing the silently dropped `--debug`, late metadata `video_strength` validation, an
   `mflux-completions` crash present at HEAD, and the empty z-image completion), the Wan
   runtime's validation head and twin metadata blocks are deduplicated behind
   `WanVideoRequest`/`_to_video_shared_kwargs`, user-mask loading is centralized in `MaskUtil`
   with a reference-faithful resampling policy, and image/video metadata carries
   `metadata_schema_version: 1`.
7. Finished [Wan V2V motion-fidelity ladder and control run](completed/0079_wan_v2v_motion_fidelity_ladder.md)
   so the strength-vs-gesture-preservation trade-off is measured (gesture-timing r 0.86/0.90 at
   strength 0.5/0.6, 0.73 at 0.7, 0.20 at the 0.8 default; edit landed at every strength) and
   documented with a copy-pasteable motion-preserving recipe in `docs/wan-video.md`, a paired
   Lightning-point control showing prompt language recovers the class of motion but not its
   timing, and a committed proof mirror under `docs/assets/validation/motion-ladder-2026-07-05/`.
8. Finished [Wan2.1-VACE-1.3B native MLX port](completed/0080_wan_vace_1_3b_native_port.md)
   so `wan-vace` ships reference-image object injection and learned masked source-video
   editing natively (parity proven stage-by-stage against the diffusers reference: mask
   preparation and UniPC bit-exact, transformer at the measured fp32 noise floor), with
   capability proofs that carry their own controls - a masked object replacement in the
   default generate mode (in-mask change 63.8 against a codec-floor background, 1027 s;
   repaint mode restyles in place, 239 s; the upstream pipeline on un-blanked inputs fails
   to replace and re-renders the whole frame, 2161 s CPU) and a reference-injection identity
   ablation (segmented subject transfers; same seed without the reference gives an unrelated
   ship) - in a committed proof mirror under `docs/assets/validation/wan-vace-2026-07-06/`.

## Completed masked-edit expansion band

The 2026-07-15 masked-edit surface audit (one adversarial subagent) identified base-Qwen native
inpaint as the strongest unsurfaced route and the Z-Image non-turbo gate as proof-blocked.
Both shipped the same day with a consolidated canonical docs page.

1. Finished [masked edit expansion: native base-Qwen and Z-Image non-turbo](completed/0082_masked_edit_expansion_qwen_zimage.md)
   so `--mask-path` works natively on trusted base Qwen rows (diffusers
   `QwenImageInpaintPipeline` port with the internal 0.85 warm start and `effective_steps`
   runtime-truth metadata; the exact validated 8bit row keeps control-inpaint) and on
   non-turbo Z-Image rows, with one masked route per row enforced by construction, plan-time
   maskless rejection, re-mirrored completions, the canonical `docs/masked-editing.md` page,
   and a visual-smoke proof bundle in `docs/assets/validation/masked-edit-2026-07-15/`.
   Control-inpaint row broadening was superseded by the native route (decision of record in
   the item); Z-Image ControlNet-inpaint stayed proposed in item 0045, refreshed with
   upstream sizing facts.
2. Promoted the smoke rows to graded matrix rows the same day: the masked-edit 5x5 matrix
   (`docs/assets/validation/masked-edit-matrix-2026-07-15/`, registry profile
   `masked_edit_matrix_5x5_2026_07_15`) scores Klein 4B q8 rows PASS across all four
   well-posed cases, the `qwen.base-inpaint` q4/2512-q8 rows PARTIAL (documented warm-start
   recolor limitation), and the Z-Image non-turbo q8 row mixed (seed-reproducible
   arm-retexture geometry failure), with an unscored partial-object-removal limitation
   demonstration folded into the masked-editing docs as a mask-design rule.

## Completed FLUX.2 Klein masked edit

The 2026-07-15 pass ported the upstream diffusers `Flux2KleinInpaintPipeline` semantics onto
the existing FLUX.2 Klein family so `--mask-path` works on distilled and base Klein rows
through unified `mlxgen generate`, with optional masked-area reference images on the backend
command and Python API.

1. Finished [FLUX.2 Klein masked edit / inpaint](completed/0081_flux2_klein_masked_edit.md)
   so the new `flux2.inpaint` capability composites unmasked latents from the re-noised
   clean source every step while the clean source rides along as conditioning tokens
   (t=10, references at t=20+), with torch-parity bilinear mask downsampling onto the
   packed latent grid, adversarial-subagent review (one major guidance-default finding
   fixed pre-smoke), and model-backed q8 smoke proofs (distilled 4-step guidance-1, base
   8-step CFG guidance-4, and a reference-conditioned plaid-fill case) preserved locally in
   `validation_outputs/flux2-klein-inpaint-smoke/`. Published visual-QA proof rows remain a
   follow-up before any exact package claims a validated masked-edit row.

## Completed audit hardening band

The 2026-06-27 code-only audit and two adversarial confirmation passes inserted a new hardening
band ahead of model-expansion work. Items 0051-0057 are now completed. The residual architecture
follow-up remains proposed as item 0058. Items 0059-0061 are now quantitatively completed; items
0062-0064 remain planned until exact-quality SeedVR2 video, startup, and retention memory
statistics prove completion.

1. Finished [safe Torch checkpoint loading](completed/0051_safe_checkpoint_loading.md) before treating
   arbitrary local or remote `.pt` / `.pth` model artifacts as safe to load.
2. Finished [fail-closed PyPI publishing](completed/0052_release_publish_fail_closed.md) before running
   local release tagging or GitHub release creation.
3. Finished [prepare model-class config resolution](completed/0053_prepare_model_class_registry_resolution.md)
   before relying on `mlxgen prepare` for custom/local model paths with `--base-model`.
4. Finished [inference defaults and step validation](completed/0054_inference_defaults_and_step_validation.md)
   to stop invalid step counts early and then remove CLI/Python default drift.
5. Finished [low-RAM repeat-generation safety](completed/0055_low_ram_repeat_generation_safety.md)
   before advertising `--low-ram` as safe for all multi-seed or prompt-file runs.
6. Finished [Wan/VLM/BF16 performance hardening](completed/0056_wan_vlm_bf16_performance_hardening.md)
   as the first concrete performance and precision pass.
7. Finished [package dependency and dist hygiene](completed/0057_package_dependency_and_dist_hygiene.md)
   as release-cleanup work after the blockers above.

## Active memory validation band

The 2026-06-27 adversarial memory pass implemented several concrete memory changes. Items 0059-0061
are now completed with real process-isolated quantitative evidence. Items 0062-0064 are still
planned because the available reports did not prove a quality-preserving SeedVR2 video memory
reduction, startup/first-step peak reduction, or retention memory reduction.

The SeedVR2 1280px image follow-up now has real 3B and 7B stats. The fixed image `--low-ram` path
is stable and pixel-identical but only reduces retained physical footprint at metadata time.
Explicit tuned `--vae-tiling` reduces MLX peak sharply, with a measured non-zero pixel delta, so it
remains an opt-in memory-pressure path rather than a default quality-preserving optimization.

1. Finished [runtime memory telemetry and manifests](completed/0060_runtime_memory_telemetry_and_manifests.md)
   with sampled process peaks, MLX peaks, schema evidence, parent physical-footprint sampling, and
   telemetry overhead.
2. Finished [prompt materialization for low-RAM release](completed/0061_prompt_materialization_for_low_ram_release.md)
   with prompt-to-denoise peak RSS statistics and exact output parity on affected families.
3. Reopened [SeedVR2 chunk-bounded video noise](planned/memory/0062_seedvr2_chunk_bounded_noise.md)
   after same-video review showed the bounded `absolute_latent_frame` path changes restored pixels
   and the 7B evidence was a frame-count scaling run, not a normal 1:1 baseline/candidate proof.
   The current package default is back to `seedvr2_noise_mode=global`; the first 2026-06-28 normal
   proof was rejected because it used unsafe temporal chunking, and the corrected
   `seedvr2_temporal_quality_repair_20260628` proof demonstrates normal 149-frame 3B/7B `29/8`
   restores but does not close a memory-reduction task.
4. Validate and finish [component-wise model loading memory policy](planned/memory/0063_componentwise_model_loading_memory_policy.md)
   with startup/first-step memory statistics; keep broader initializer streaming in proposed item
   0065 until startup remains a measured blocker.
5. Validate and finish [generation retention cleanup](planned/memory/0064_generation_retention_cleanup.md)
   with stepwise/debug and hidden-state-retention memory statistics plus quality/performance checks.

## Topic tracks

- [Runtime contract hardening](planned/runtime_contracts/README.md): completed June 29 closure band
  for items 0067-0070, with the shared report in
  `docs/assets/validation/runtime-contracts-2026-06-29/runtime_contracts_report.md`.
- [Memory validation](planned/memory/README.md): items 0062-0064 keep the remaining exact-quality
  memory validation work grouped separately from runtime contract repair.

## Next recommended work

The highest-priority work is now the remaining memory-validation band. The runtime-contract
hardening band is closed and documented.

1. Finish the remaining memory validation track: find and test an exact-quality SeedVR2 video
   memory reduction for item 0062 only if a larger still-supported profile proves the global-noise
   term is materially relevant; otherwise re-scope it as a low-value research item for future
   enlarged-video profiles. In parallel, run phase-isolated startup/first-step profiles for item
   0063 and retention-isolating profiles for item 0064, record statistics under
   `validation_outputs/memory/`, and only then close those remaining items.
2. Finish
   [Wan prompt adherence parity validation](planned/0015_wan_prompt_adherence_parity_validation.md)
   before treating T2V/I2V prompt or motion behavior as quality-proven; explicitly match official
   Wan negative prompts and A14B guidance pairs in Diffusers-vs-MLX runs.
3. Finish the remaining
   [first-class I2I modes and outpaint/reframe UX](planned/0019_first_class_i2i_modes_and_outpaint_reframe.md)
   work where it still adds capability beyond the now-completed Qwen and FLUX.2 route surface.
   Route ownership is much clearer now; the next useful work is UX and contract cleanup, not
   another speculative route split.
4. Finish
   [Wan2.2 TI2V-5B math and behavior parity](planned/0035_wan_ti2v5b_math_and_behavior_parity.md)
   after the current image-route work is closed. The official Wan source and local Diffusers audit
   found no tensor mismatch in the existing TI2V-5B fixtures, but public quality claims still need
   the broader behavior proof and the explicit `--flow-shift 3` route.
5. Keep proposed
   [SeedVR2 enlarged-video safe-profile certification](proposed/0048_seedvr2_enlarged_video_safe_profile_certification.md)
   as the remaining SeedVR2 proof question after release `0.18.20`. The bounded June 21 public
   proof already answers the `3B` versus `7B` comparison on the accepted slice; what remains is
   whether any enlarged SeedVR2 video recipe should graduate from explicit unsafe override to a
   documented safe public profile.
6. Preserve
   [Z-Image ControlNet follow-up](proposed/0045_zimage_controlnet_followup.md) as the next image
   follow-up after native Z-Image inpaint and the completed Qwen parity work rather than
   broadening current items again.
7. Keep proposed
   [LightX2V Wan distilled-model loader support](proposed/0041_lightx2v_wan_distilled_model_loader_support.md)
   scoped as the next Wan acceleration follow-up, not the current one. Completed
   [item 0040](completed/0040_lightx2v_wan_4step_acceleration_profiles.md) now provides the exact
   LightX2V Lightning 4-step A14B fast path on the current runtime, so 0041 should only advance if
   native distilled checkpoints still offer clearer user value than the explicit LoRA recipe.

Keep proposed
[component-wise weight streaming migration](proposed/0065_componentwise_weight_streaming_migration.md)
as the next startup-memory follow-up after real process-memory profiling. The Z-Image cache-policy
profile now shows startup/first-step peak still dominates even when retained cache falls sharply.
Do not promote it until one specific family has a measured startup peak that justifies the
family-specific migration risk. The 2026-06-29 size audit suggests the first prepared-package
candidate should be Z-Image or FLUX.2 rather than SeedVR2, while the SeedVR2 1280px image profile
still points instead to VAE spatial encode peak, so 0065 should not be promoted solely for that
case.

Completed
[image finalization memory peak and metadata rewrite](completed/0066_image_finalization_memory_peak_and_metadata_rewrite.md).
The current generated-image default path is now a one-pass save with zero reopen passes and zero
runtime-memory snapshots, while opt-in embedded metadata still preserves EXIF plus PNG XMP/IPTC in
one save call. The June 30, 2026 `4096x4096` save-phase probe measured peak sampled RSS
`0.394 GB -> 0.192 GB` (`-51.2316%`) and peak Darwin physical footprint `0.373 GB -> 0.171 GB`
(`-54.1496%`) versus the legacy three-pass simulation, so this no longer belongs in proposed
memory follow-up state.

8. Finish the prepared-package residue in
   [FLUX.2 Klein base source validation and contact sheets](planned/0036_flux2_klein_base_source_validation_and_contact_sheets.md).
   Source-model base `4B/9B` now have starship proof, and the exact q8 base-4B outpaint row is
   now validated through the LoRA route-completion bundle, but the broader prepared base package
   contact-sheet surface is still narrower than the source-model profile.
9. Keep proposed
   [Wan VACE video editing and control](proposed/0039_wan_vace_video_editing_and_control.md) and
   [Wan VACE conditioning expansion after plain video-to-video](proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md)
   as the later richer-conditioning follow-up, not the current runtime baseline. The shipped plain
   V2V route means the remaining bar is now strictly about whether richer conditioning is good
   enough to justify expansion. The July 3 local VACE proof bundle is useful because it stops
   overclaiming: the path runs, but the bounded `1.3B` MPS results were not good enough to promote yet.
10. Keep the
   [FLUX.2-dev multi-angle LoRA support](planned/0034_flux2_dev_multi_angle_lora_support.md)
   item parked even after 0007 completion. The lovis multi-angle adapter still targets
   `black-forest-labs/FLUX.2-dev`, not FLUX.2 Klein, so there is no reason to expand the runtime
   surface until the current supported image families have exact LoRA proofs.
12. Validate and finish
   [Wan A14B boundary memory recovery and full-size validation](planned/0013_wan_a14b_boundary_memory_recovery.md)
   after the full-size I2V retry captures memory, exit-code, metadata, and output evidence across
   the high-noise to low-noise denoiser boundary. Remaining memory items 0062-0064 must provide quantitative
   evidence before this can depend on those memory claims.
13. Finish the [Wan quantization and motion parity](planned/0002_wan_quantization_motion_parity.md)
   residuals: TI2V-5B now has clean source/BF16/q8 evidence at 1280x704, 17 frames, 20 steps, but
   the TI2V-5B memory result is storage/MLX-footprint focused rather than a full-process physical
   peak reduction; full-duration validation, I2V-A14B mixed q8 quality, q4 policy, and exact-setting
   card claims still need to stay tied to passed settings.
14. Use the completed
   [edit model prepared-package capability contact sheets](completed/0026_edit_model_prepared_capability_contact_sheets.md)
   as the current release gate for image-edit quality claims: FLUX.2 Klein source/q8 and Qwen Edit
   2509 source/q8 passed the standardized sequence. Qwen Edit 2511 has newer source/q8/q4 proof in
   [item 0029](completed/0029_qwen_image_edit_2511_base_parity.md). FIBO Edit remains unsupported
   through unified `mlxgen generate`. Use the [release validation registry](completed/0028_release_validation_registry.md)
   for machine-readable package status.
15. Keep [FIBO Edit Diffusers parity](planned/0027_fibo_edit_diffusers_parity_release_quality.md)
   and [FIBO Edit unified validation](planned/0024_fibo_edit_unified_i2i_validation.md) deferred.
   FIBO Edit remains unsupported through unified `mlxgen generate`; do not schedule more work here
   ahead of outpaint/reframe, LoRA strictness, or video quality work unless a specific product need
   changes the priority.
16. Keep Bonsai binary 1-bit deferred in
   [proposed item 0004](proposed/0004_bonsai_binary_1bit_runtime_support.md) until stock MLX can
   execute the required 1-bit packed affine matmul or an ADR accepts a custom kernel path.
17. Investigate [Wan q8 performance](planned/0005_wan_q8_performance_investigation.md) only after
   integrity-gated outputs are healthy enough for timing claims; current public docs describe mixed
   q8/BF16 as model-size and measured-profile footprint focused, not speed-improving.
18. Continue the [model integration roadmap](planned/0001_model_integration_roadmap.md) in priority
   order, starting with automated publication audits, supported q4/q8 validation, and
   gated-derivative hygiene.
19. Keep the second-family video bucket explicit. Proposed
   [HunyuanVideo-1.5 second-family spike](proposed/0044_hunyuanvideo15_second_family_spike.md)
   is now the strongest concrete non-Wan candidate for Apple Silicon bounded by a public `480p`
   step-distilled path, while proposed
   [LTX family conditioning and LoRA spike](proposed/0010_ltx2_conditioning_lora_spike.md) should
   stay narrowed to `LTX-Video` first rather than broad `LTX-2.3` audio-video scope.
20. Keep the image/edit watchlist honest. Proposed
   [next-generation image/edit watchlist](proposed/0011_next_generation_image_edit_watchlist.md)
   now includes Ideogram 4, Ovis-Image, PRXPixel, and DreamLite, but those remain watchlist-only
   until license, runtime shape, or direct MLX value becomes clearer. Proposed
   [Boogu image family support](proposed/0049_boogu_image_family_support.md) is now the focused
   version of that watch state: interesting enough to preserve because of bilingual text and
   unified generation/editing, but not yet strong enough to outrank the active Qwen, Z-Image, and
   Wan items. Proposed [Krea 2 Turbo integration](proposed/0050_krea2_turbo_integration.md) is an
   adjacent low-priority watch item: technically credible, but held back mainly by its
   materially restrictive custom license.
21. Continue ERNIE-Image/Turbo after completed
   [ERNIE Image Turbo LoRA runtime support](completed/0037_ernie_image_turbo_lora_runtime_support.md):
   the latent img2img proof is now accepted, so the remaining follow-up is stronger Diffusers
   parity coverage and non-turbo validation.
22. Continue Wan2.2 after the first TI2V-5B and A14B T2V/I2V milestones: add q8/q4 validation,
   stronger quality/performance checks, and remaining cancel APIs. SeedVR2 has a validated
   `mlxgen upscale` command, official 3B/7B source loading, and q8/q4 `mlxgen prepare` package
   support.
23. Keep Bonsai LoRA fail-closed and low priority; revisit it only through
   [proposed item 0038](proposed/0038_bonsai_packed_lora_runtime_support.md). The current packed
   runtime does not expose replaceable linear targets for standard LoRA injection, and the first
   public “Bonsai LoRA” candidate inspected used unrelated SDXL UNet keys.
## Planned ledger

| ID | Item | Area | Priority | Status |
| --- | --- | --- | --- | --- |
| 0001 | [Model integration roadmap](planned/0001_model_integration_roadmap.md) | Models, routing, quantization, UX | P0-P3 | Planned |
| 0002 | [Wan quantization and motion parity](planned/0002_wan_quantization_motion_parity.md) | Video, quantization, Diffusers parity | P0 | Planned |
| 0005 | [Wan q8 performance investigation](planned/0005_wan_q8_performance_investigation.md) | Video, performance, quantization | P1 | Planned |
| 0013 | [Wan A14B boundary memory recovery and full-size validation](planned/0013_wan_a14b_boundary_memory_recovery.md) | Video, memory, progress, validation | P0 | Planned |
| 0015 | [Wan prompt adherence parity validation](planned/0015_wan_prompt_adherence_parity_validation.md) | Video, Diffusers parity, prompt adherence | P0 | Planned |
| 0019 | [First-class I2I modes and outpaint/reframe UX](planned/0019_first_class_i2i_modes_and_outpaint_reframe.md) | Image routing, generative reframe, outpaint capability | P0 | Planned |
| 0024 | [FIBO Edit unified I2I validation](planned/0024_fibo_edit_unified_i2i_validation.md) | Image routing, FIBO Edit, validation | P3 deferred | Planned |
| 0027 | [FIBO Edit Diffusers parity and release-quality validation](planned/0027_fibo_edit_diffusers_parity_release_quality.md) | Image edit, FIBO Edit, Diffusers parity | P3 deferred | Planned |
| 0034 | [FLUX.2-dev multi-angle LoRA support](planned/0034_flux2_dev_multi_angle_lora_support.md) | LoRA, FLUX.2-dev, validation | P0 | Planned |
| 0035 | [Wan2.2 TI2V-5B math and behavior parity](planned/0035_wan_ti2v5b_math_and_behavior_parity.md) | Video, Wan TI2V-5B, official/Diffusers parity | P0 | Planned |
| 0036 | [FLUX.2 Klein base source validation and contact sheets](planned/0036_flux2_klein_base_source_validation_and_contact_sheets.md) | FLUX.2 base, validation, docs | P0 | Planned |
| 0062 | [SeedVR2 chunk-bounded video noise](planned/memory/0062_seedvr2_chunk_bounded_noise.md) | Video restoration, SeedVR2, memory | P0 | Reopened; exact-quality memory reduction pending |
| 0063 | [Component-wise model loading memory policy](planned/memory/0063_componentwise_model_loading_memory_policy.md) | Memory, CLI, model loading | P0 | Quantitative validation pending |
| 0064 | [Generation retention cleanup](planned/memory/0064_generation_retention_cleanup.md) | Memory, hidden states, stepwise output | P0 | Quantitative validation pending |

## Proposed ledger

| ID | Item | Area | Promotion criteria |
| --- | --- | --- | --- |
| 0004 | [Bonsai binary 1-bit runtime support](proposed/0004_bonsai_binary_1bit_runtime_support.md) | T2I, low-bit runtime | Promote after ternary works and MLX 1-bit packed affine runtime support is proven or accepted by ADR. |
| 0006 | [Wan I2V prompt motion validation](proposed/0006_wan_i2v_prompt_motion_validation.md) | Video, I2V quality | Promote only if planned item 0015 shows an I2V-specific motion or prompt-adherence gap that needs a separate fix. |
| 0009 | [Video second-family selection](proposed/0009_video_second_family_selection.md) | Video model roadmap | Promote after Wan stabilization leaves room for the next video backend. |
| 0010 | [LTX family conditioning and LoRA spike](proposed/0010_ltx2_conditioning_lora_spike.md) | Video, LTX, LoRA | Promote if the LTX family becomes the selected second video family or a local spike proves feasibility. |
| 0011 | [Next-generation image/edit watchlist](proposed/0011_next_generation_image_edit_watchlist.md) | Image/edit roadmap | Promote when a watched model becomes locally cacheable, licensed, and useful enough for implementation. |
| 0038 | [Bonsai packed-runtime LoRA support](proposed/0038_bonsai_packed_lora_runtime_support.md) | Bonsai, LoRA, packed runtime architecture | Promote only if MLX-Gen adopts an unpacked Bonsai LoRA route, a packed-kernel LoRA path, or a real public Bonsai-compatible adapter family. |
| 0039 | [Wan VACE video editing and control](proposed/0039_wan_vace_video_editing_and_control.md) | Video editing, Wan, VACE | Promote after current Wan parity work settles and one official Wan VACE or Wan video-to-video route is selected for bounded smoke validation. |
| 0041 | [LightX2V Wan distilled-model loader support](proposed/0041_lightx2v_wan_distilled_model_loader_support.md) | Video, Wan, LightX2V, native distilled checkpoints | Promote after completed item 0040 and one exact distilled A14B file set is audited deeply enough to size the scheduler-plus-loader delta. |
| 0044 | [HunyuanVideo-1.5 second-family spike](proposed/0044_hunyuanvideo15_second_family_spike.md) | Video, HunyuanVideo, second-family selection | Promote after a bounded upstream Diffusers smoke and a license/value comparison against Wan, SeedVR2, and LTX. |
| 0045 | [Z-Image ControlNet follow-up](proposed/0045_zimage_controlnet_followup.md) | Image edit, Z-Image, ControlNet | Promote after completed item 0043 is accepted as the native-inpaint baseline and one public ControlNet weight family is audited cleanly. |
| 0048 | [SeedVR2 enlarged-video safe-profile certification](proposed/0048_seedvr2_enlarged_video_safe_profile_certification.md) | Video restoration, SeedVR2, validation | Promote after the accepted `0.18.20` proof bundle is stable, completed item 0046 is no longer the blocking follow-up, and there is a concrete reason to revisit the safe enlarged-video boundary. |
| 0049 | [Boogu image family support](proposed/0049_boogu_image_family_support.md) | Image model roadmap, Boogu, text rendering, editing | Promote only if independent evidence or repeatable local proof shows a meaningful win over current Qwen/Z-Image/FLUX routes and the non-fp8 Boogu path looks credible on Apple Silicon. |
| 0050 | [Krea 2 Turbo integration](proposed/0050_krea2_turbo_integration.md) | Image model roadmap, Krea 2, licensing, fast text-to-image | Promote only if MLX-Gen explicitly accepts the restrictive Krea license class and a bounded upstream smoke shows clear value over current fast image routes. |
| 0058 | [Model profile registry authority](proposed/0058_model_profile_registry_authority.md) | Architecture, model identity, defaults, loader policy | Promote after an ADR spike proves one lightweight registry can own model family identity/defaults without import-cycle or startup-cost regressions. |
| 0065 | [Component-wise weight streaming migration](proposed/0065_componentwise_weight_streaming_migration.md) | Memory, startup, weight loading | Promote when profiling shows startup peak remains a practical blocker after remaining item 0063, or when one specific family needs the reduction for a supported profile. |
| 0075 | [Wan VACE conditioning expansion after plain video-to-video](proposed/0075_wan_vace_conditioning_expansion_after_plain_video_to_video.md) | Video editing, Wan, richer conditioning | Promote only after the plain public `video-to-video` route is shipped or strongly proven and a new exact VACE proof beats the bounded July 3 `1.3B` MPS results. |
## Completed ledger

| ID | Item | Area | Completed | Outcome |
| --- | --- | --- | --- | --- |
| 0003 | [Bonsai ternary FLUX.2 support](completed/0003_bonsai_ternary_flux2_support.md) | T2I, FLUX.2, low-bit packed MLX | 2026-05-27 | Added Bonsai ternary 2-bit routing, packed transformer loading, q4 Qwen3 text-encoder loading, binary 1-bit runtime gating, docs, and local quality/speed validation against FLUX.2 Klein 4B q8. |
| 0007 | [LoRA capability matrix and strict application](completed/0007_lora_capability_matrix_and_strict_application.md) | LoRA, routing, validation | 2026-06-22 | Finished the route-level LoRA contract, strict loader/runtime rejection path, exact validation-profile surfacing, and current production-supported Qwen/Z-Image/FLUX.2/ERNIE/Wan proof rows. |
| 0008 | [Qwen edit parity expansion](completed/0008_qwen_edit_parity_expansion.md) | Qwen edit, inpaint, structured control | 2026-06-22 | Completed the current production-grade Qwen surface: route matrix, masked edit, structured control, base control-inpaint, and exact 2511 multi-reference/reframe/outpaint route proofs. |
| 0012 | [Wan2.2 A14B T2V/I2V support](completed/0012_wan_a14b_t2v_i2v_support.md) | Video, Wan A14B, Diffusers parity | 2026-05-31 | Added T2V-A14B and I2V-A14B configs, dynamic two-transformer loading, Wan2.1-style VAE support, boundary routing, optional `--guidance-2`, fail-closed model identity checks, docs, tests, and MP4 smoke validation. |
| 0014 | [Shared progress callbacks for image and video pipelines](completed/0014_shared_progress_callbacks.md) | Callbacks, Python API, image/video progress | 2026-06-03 | Added one shared `ProgressEvent`, image progress subscriptions, Wan shared step-based progress events, CLI denoise-step progress, docs, and focused tests. |
| 0016 | [Wan video integrity release gate](completed/0016_wan_video_integrity_release_gate.md) | Video, numerical integrity, release validation | 2026-06-06 | Added fail-closed decoded-frame/video-health gates, saved-video health metadata, Wan failure diagnostics, diagnostic opt-in behavior, and targeted T2V/I2V A14B q8 release-validation artifacts. |
| 0018 | [Taskless generation routing](completed/0018_taskless_generation_routing.md) | Routing, Python API, task validation | 2026-06-04 | Added public task inference, taskless T2I/I2I/T2V/I2V routing, FLUX.2 single-image I2I behavior, Wan A14B early fixed-task validation, generated-card/doc updates, and focused validation. |
| 0020 | [Generation capability contract and route planning](completed/0020_generation_capability_contract.md) | Routing, Python API, capabilities, I2I modes | 2026-06-04 | Added typed generation capabilities, public generation plans, `mlxgen capabilities`, `--i2i-mode`, metadata-aware route planning, local-path base-model hints, early option rejection, edit/I2I progress labeling, docs, and focused validation. |
| 0021 | [Wan I2V source aspect-ratio preservation](completed/0021_wan_i2v_source_aspect_ratio.md) | Video, I2V, geometry | 2026-06-04 | Wan image-to-video now resolves output dimensions from the source image ratio and model spatial multiples before conditioning, records requested/source/resolved dimensions in metadata, and has TI2V-5B plus A14B local proof outputs. |
| 0022 | [I2I source aspect-ratio policy](completed/0022_i2i_source_aspect_ratio_policy.md) | Image routing, I2I geometry, Python API | 2026-06-04 | Ordinary latent, edit/reference, and multi-reference I2I now default to source-aspect canvas resolution, expose canvas policy in capabilities and metadata, retain explicit exact-resize, and have focused tests plus a post-fix I2I compatibility matrix. |
| 0023 | [I2I capability validation matrix](completed/0023_i2i_capability_validation_matrix.md) | Image routing, I2I validation, release gate | 2026-06-05 | Produced a historical installed-model I2I proof matrix, but a 2026-06-05 correction report marks the matrix insufficient as comparable complex-edit proof; completed items 0025 and 0026 supersede it for standardized release validation. |
| 0025 | [Standardized I2I sequence validation](completed/0025_standardized_i2i_sequence_validation.md) | Image routing, I2I validation, release proof | 2026-06-05 | Ran a fixed source/prompt/seed 30-row matrix: FLUX.2 Klein q4/q8 passed all applicable steps; Qwen Edit 2511 4-bit failed complex crash/composition; Qwen Edit 2511 8-bit passed crash but not full composition; Qwen Image 2512 latent lost spaceship identity; Z-Image Turbo and ERNIE Turbo passed latent cinematic rows. |
| 0026 | [Edit model prepared-package capability contact sheets](completed/0026_edit_model_prepared_capability_contact_sheets.md) | Image edit, q8 validation, contact sheets | 2026-06-05 | Rebuilt source-handle and per-variant proof matrices: FLUX.2 Klein 4B/9B source/q8/q4 passed; Qwen Edit 2509 source/q8 passed and q4 is partial on multi-reference; the Qwen Edit 2511 rows were superseded by completed item 0029; FIBO Edit source validation failed and remains unsupported through unified generation. |
| 0028 | [Release validation registry for I2I evidence](completed/0028_release_validation_registry.md) | Image routing, validation API, release evidence | 2026-06-05 | Added route-separated release-validation metadata, public Python lookup helpers, `mlxgen validation`, clearer capabilities wording, exact-command manifest coverage for q4 matrix rows, and regenerated the legacy summary contact sheet as the clear 5x4 matrix. |
| 0029 | [Qwen Image Edit 2511 base parity](completed/0029_qwen_image_edit_2511_base_parity.md) | Image edit, Qwen 2511, Diffusers parity | 2026-06-06 | Fixed Qwen FlowMatch dynamic-shift scheduler parity and validated Qwen Image Edit 2511 source/q8/q4 on the focused pencil, crash, and multi-reference composition profile. |
| 0030 | [SeedVR2 upscale smoke, metadata, and quality defaults](completed/0030_seedvr2_upscale_smoke_and_metadata.md) | Upscale, SeedVR2, metadata | 2026-06-07 | Validated the SeedVR2 3B q8 upscaler on a small real image, fixed non-16-multiple output metadata, defaulted SeedVR2 to untiled VAE processing for image quality, added `--vae-tiling`, and added fast regression tests for final output dimensions, source-image metadata, and CLI routing. |
| 0031 | [SeedVR2 official ByteDance checkpoint support](completed/0031_seedvr2_official_bytedance_checkpoint_support.md) | Upscale, SeedVR2, official checkpoints | 2026-06-07 | Added direct official `ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B` `.pth` loading, switched SeedVR2 aliases to official sources, added q8/q4 `mlxgen prepare` support, generated reusable package cards, and validated source/q8/q4 5x upscale profiles. |
| 0032 | [SeedVR2 video restoration and upscaling](completed/0032_seedvr2_video_restoration_upscaling.md) | Video restoration, upscale, SeedVR2 | 2026-06-21 | Added bounded and streamed `--video-path` restore under `mlxgen upscale`, official temporal clip handling, source-FPS preservation, explicit host-safe video guardrails, accepted June 21 five-second `1x 29/8` and `2x 29/8` proof bundles, and the release-quality visual validation surface that item 0046 then completed with audio copy-through. |
| 0046 | [SeedVR2 video audio copy-through](completed/0046_seedvr2_video_audio_copythrough.md) | Video restoration, audio, remux | 2026-06-21 | Added shared post-write audio copy-through for SeedVR2 video restore, made source-audio preservation the default saved-output contract with explicit `--drop-audio` opt-out, kept fail-closed metadata, and published a real-source Air France `25s–35s` proof bundle. |
| 0043 | [Z-Image native inpaint](completed/0043_zimage_native_inpaint.md) | Image edit, Z-Image, mask-based inpaint | 2026-06-21 | Added exact `z-image.inpaint` routing on `AbstractFramework/z-image-turbo-8bit`, fail-closed `--mask-path` validation, native latent-mask blending, and an accepted same-prompt same-seed engine-thruster proof against the old latent route. |
| 0033 | [Video LoRA support for T2V and I2V](completed/0033_video_lora_for_t2v_i2v.md) | Video, LoRA, Wan2.2 | 2026-06-11 | Added Wan-specific LoRA mapping and explicit role routing, then validated all current Wan q8 public rows with model-backed A/B artifacts: TI2V-5B text-to-video, TI2V-5B first-frame image-to-video, T2V-A14B text-to-video, and I2V-A14B first-frame image-to-video. |
| 0040 | [LightX2V Wan 4-step acceleration profiles](completed/0040_lightx2v_wan_4step_acceleration_profiles.md) | Video, Wan, LightX2V, fast-path validation | 2026-06-12 | Validated the explicit LightX2V Lightning 4-step A14B fast path on q8 T2V and I2V with same-seed no-LoRA vs paired-LoRA A/B contact sheets, route-level validation profiles, and documented exact commands using `steps=4`, `flow_shift=5.0`, `guidance=1.0`, and `guidance_2=1.0`. |
| 0042 | [GitHub Actions Node 24 migration](completed/0042_github_actions_node24_migration.md) | CI, release automation, GitHub Actions | 2026-06-14 | Finished the Node 24 cleanup by replacing `softprops/action-gh-release@v2` with a GitHub CLI release step, validating PR checks and a non-publishing release rehearsal, and preserving the final PR as a reusable migration example. |
| 0037 | [ERNIE Image Turbo LoRA runtime support](completed/0037_ernie_image_turbo_lora_runtime_support.md) | ERNIE, LoRA, routing, validation | 2026-06-11 | Added ERNIE transformer LoRA mapping, public-route LoRA support, exact q8 text-to-image validation with an anime-style adapter, and kept latent img2img plus Bonsai packed-runtime work explicitly separate. |
| 0051 | [Safe Torch checkpoint loading](completed/0051_safe_checkpoint_loading.md) | Security, weights, model loading | 2026-06-27 | Switched Torch checkpoint loading to `weights_only=True`, rejected empty/mixed non-tensor payloads, preserved official SeedVR2 tensor/state-dict loading, and preserved BF16 tensors as MLX BF16. |
| 0052 | [Fail closed on PyPI publish failures](completed/0052_release_publish_fail_closed.md) | Release, PyPI, integrity | 2026-06-27 | Made local PyPI failures fatal before tag/release creation, preserved duplicate-version idempotency, added ReleaseManager coverage, and reordered GitHub Actions so PyPI succeeds before GitHub Release publication. |
| 0053 | [Resolve prepare model class from model config](completed/0053_prepare_model_class_registry_resolution.md) | Prepare, routing, model config | 2026-06-27 | Made `mlxgen prepare` select backends from resolved `ModelConfig`, covered explicit-base custom Qwen paths, and replaced the hidden Flux fallback with an explicit `--base-model` recovery error. |
| 0054 | [Centralize inference defaults and step validation](completed/0054_inference_defaults_and_step_validation.md) | Config, CLI, Python API | 2026-06-27 | Added early CLI/common-config/scheduler step validation and introduced a shared inference-step resolver used by CLI defaults, common Config, and confirmed direct-API drift points. |
| 0055 | [Low-RAM repeat-generation safety](completed/0055_low_ram_repeat_generation_safety.md) | Memory, CLI, callbacks | 2026-06-27 | Added parser-level fail-closed guards for low-RAM repeat cases that would need released encoders, covering multi-seed prompt-file and FIBO-family multi-seed combinations. |
| 0056 | [Wan, VLM, and BF16 performance hardening](completed/0056_wan_vlm_bf16_performance_hardening.md) | Video, VLM, precision | 2026-06-27 | Added Wan lazy batched-save routing, moved Qwen3 top-p sampling to MLX, vectorized image-token replacement, preserved Torch BF16 as MLX BF16, and split full Wan VAE streaming proof into follow-up item 0059, now completed. |
| 0057 | [Package dependency and distribution hygiene](completed/0057_package_dependency_and_dist_hygiene.md) | Packaging, build, dependencies | 2026-06-27 | Fixed dist artifact cleanup/extraction for `mlx_gen-*`, moved `twine` to optional dev/release extras, refreshed the lockfile, and verified `make build` against actual artifacts. |
| 0059 | [Wan VAE streaming and memory measurement](completed/0059_wan_vae_streaming_memory_measurement.md) | Video, Wan, memory, measurement | 2026-06-27 | Completed real eager-versus-streamed Wan TI2V validation: median MLX peak fell 29.1%, metadata physical footprint fell 68.9%, saved MP4 output was exact, and startup peak remains tracked separately by 0063/0065. |
| 0060 | [Runtime memory telemetry and manifests](completed/0060_runtime_memory_telemetry_and_manifests.md) | Memory, telemetry, manifests | 2026-06-28 | Completed real Z-Image telemetry-overhead validation: exact image parity, sampled RSS overhead +0.0019%, sampled Darwin physical-footprint overhead +0.3370%, wall overhead +1.0814%, and metadata physical footprint agreed with parent sampling. |
| 0061 | [Prompt materialization for low-RAM release](completed/0061_prompt_materialization_for_low_ram_release.md) | Memory, prompt encoders, low-RAM | 2026-06-28 | Completed real Flux2 and ERNIE prompt-materialization validation with exact image parity; peak sampled RSS fell 7.25% for ERNIE and 2.17% for Flux2, while MLX peak stayed effectively flat. |
| 0066 | [Image finalization memory peak and metadata rewrite](completed/0066_image_finalization_memory_peak_and_metadata_rewrite.md) | Memory, image finalization, metadata embedding | 2026-06-30 | Completed the default-save contract rewrite: default image save is now one-pass and metadata-light, embedded metadata remains opt-in and preserves EXIF/XMP/IPTC, and the dedicated `4096x4096` save-phase probe measured peak sampled RSS `-51.2316%` and peak Darwin physical footprint `-54.1496%` versus the legacy three-pass simulation. |
| 0067 | [Progress event contract hardening](completed/0067_progress_event_contract_hardening.md) | Progress, callbacks, image/video task semantics | 2026-06-29 | Fixed task labeling and terminal progress semantics across masked image routes and SeedVR2 image/video restore so subscribers only see `complete` after artifact-ready success. |
| 0068 | [Qwen control route hardening](completed/0068_qwen_control_route_hardening.md) | Qwen, control-inpaint, prompt contract | 2026-06-29 | Fixed the public base-Qwen control and control-inpaint no-LoRA crash, preserved base-Qwen blank-negative behavior, and refreshed the one-at-a-time control-inpaint proof. |
| 0069 | [Z-Image CFG and inpaint repair](completed/0069_zimage_cfg_and_inpaint_repair.md) | Z-Image, CFG, native inpaint quality | 2026-06-29 | Corrected Z-Image CFG math, reran the native inpaint engine case on the same prompt/source/mask/seed settings, and kept the route-level quality claim tied to the documented proof row. |
| 0070 | [Canonical capability identity for variant routes](completed/0070_canonical_capability_identity_for_variant_routes.md) | Routing, capabilities, model identity | 2026-06-29 | Hardened capability and route identity so spoofed local/custom names and remote-looking prepared ids cannot unlock unsupported Qwen, Z-Image, FLUX.2 base, or FIBO variant-sensitive routes. |
| 0071 | [Python runtime serial multi-output reuse](completed/0071_python_runtime_serial_multi_output_reuse.md) | Python API, runtime loading, multi-output execution | 2026-06-30 | Added route-resolved Python runtime loading plus `generate_output(...)` / `generate_outputs(...)`, overwrite-safe save handling, and published reuse-vs-reload validation across Qwen masked edit, FLUX.2 multi-reference, Wan A14B I2V, and large Z-Image generation. |
| 0072 | [Reader-first video workflow boundary and generative video-edit contract](completed/0072_reader_first_video_workflow_boundary_and_generative_video_edit_contract.md) | CLI help, docs, ADR alignment, validation discovery | 2026-07-03 | Aligned the current video workflow boundary across CLI help, docs, ADR 0006, and backlog sequencing so future source-video editing starts from a truthful public surface instead of hidden or ambiguous terminology. |
| 0073 | [Wan VACE reference validation harness and bounded source cases](completed/0073_wan_vace_reference_validation_harness_and_bounded_source_cases.md) | Video editing, Wan VACE, reference proof, memory | 2026-07-03 | Added a repeatable upstream VACE probe with preserved artifacts and memory metrics, then proved that the bounded `Wan2.1-VACE-1.3B` MPS path runs locally but did not meet a release-quality visual bar on the portrait and ship cases. |
| 0074 | [Wan plain generative video-to-video route](completed/0074_wan_plain_generative_video_to_video_route.md) | Video editing, public task surface, Wan runtime | 2026-07-03 | Shipped the bounded public Wan `video-to-video` route on `Wan2.2-T2V-A14B`, kept non-V2V Wan rows fail-closed, required `unipc` on the public V2V surface, aligned source-video conditioning with float32 warm-start prep, and preserved a model-backed ship-edit proof bundle. |
| 0076 | [Wan masked video-to-video via latent compositing](completed/0076_wan_masked_video_to_video_latent_compositing.md) | Video editing, masked conditioning, Wan runtime, planner roles | 2026-07-04 | Shipped `--video-mask-path` masked video-to-video on the existing A14B route with per-step latent plus UniPC-state compositing, exact background preservation at the measured H.264 floor (drift 1.7 vs 14.9 unmasked), typed planner role, replayable metadata, and an in-repo conference proof bundle. |
| 0077 | [Wan V2V fps resampling and audio copy-through](completed/0077_wan_v2v_fps_resampling_and_audio_copy_through.md) | Video editing, fps timeline, audio | 2026-07-05 | Made Wan A14B video-to-video resample the source onto the requested fps timeline at real-time speed and copy source audio onto the output best-effort, with a published conference proof bundle and documented contract. |
| 0078 | [Router descriptors, Wan decomposition phase 1, MaskUtil, metadata schema version](completed/0078_router_descriptors_wan_decomposition_mask_util_schema_version.md) | Router contract, Wan structure, masks, metadata | 2026-07-05 | Replaced ad-hoc router forwarding with a descriptor table locked by round-trip tests (fixing dropped `--debug`, late strength validation, and broken completions), started Wan runtime decomposition, centralized user-mask loading in `MaskUtil`, and introduced `metadata_schema_version` 1. |
| 0079 | [Wan V2V motion-fidelity ladder and control run](completed/0079_wan_v2v_motion_fidelity_ladder.md) | Video editing, motion preservation, measured docs | 2026-07-05 | Measured the strength-vs-gesture-preservation trade-off (r 0.86-0.90 at strength 0.5-0.6 vs 0.20 at the 0.8 default) with a paired Lightning control run, published the proof mirror, and documented a motion-preserving restyle recipe. |
| 0080 | [Wan2.1-VACE-1.3B native MLX port](completed/0080_wan_vace_1_3b_native_port.md) | Video editing, VACE conditioning, Wan2.1 | 2026-07-06 | Shipped `wan-vace` reference-image object injection and learned masked source-video editing, parity-proven stage-by-stage against the diffusers reference, with controlled capability proofs and a committed validation mirror. |
| 0083 | [Bug: --failure-diagnostics advertised as common](completed/0083_bug_declared_cli_option_coverage_failure_diagnostics.md) | CLI help contract, router | 2026-07-15 (resolved pre-0.19.0) | Root-level 2026-06-30 bug report moved into the backlog; the router help now scopes the flag to Wan routes, docs state the boundary, and a focused test pins the scoped help text. |
| 0084 | [Bug: image finalization memory spike](completed/0084_bug_image_finalization_memory_contract.md) | Memory, image save path | 2026-07-15 (resolved 2026-06-30) | Root-level bug report moved into the backlog; authoritative completion record is item 0066 (one-pass metadata-light default save, measured -51%/-54% save-phase peaks). |
| 0085 | [Bug: machine-readable CLI runtime contract](completed/0085_bug_machine_readable_cli_runtime_contract.md) | CLI events, embedding apps | 2026-07-15 (resolved in 0.19.0) | Root-level bug report moved into the backlog; `--json-events` shipped the JSONL runtime stream with authoritative identity, artifact-ready terminal semantics, diagnostics paths, and documented schema. |
| 0081 | [FLUX.2 Klein masked edit / inpaint](completed/0081_flux2_klein_masked_edit.md) | Image edit, FLUX.2 Klein, mask-based inpaint | 2026-07-15 | Ported the diffusers `Flux2KleinInpaintPipeline` semantics onto the Klein family: new `flux2.inpaint` capability with per-step source compositing and clean-source conditioning tokens, torch-parity bilinear mask downsampling, optional masked-area reference images on the backend/Python surface, adversarial review, and local q8 smoke proofs (published visual-QA rows remain follow-up). |
| 0082 | [Masked edit expansion: native base-Qwen and Z-Image non-turbo](completed/0082_masked_edit_expansion_qwen_zimage.md) | Image edit, Qwen base, Z-Image, mask routing | 2026-07-15 | Shipped native `qwen.base-inpaint` (diffusers `QwenImageInpaintPipeline` port, internal 0.85 warm start, `effective_steps` metadata, one masked route per row) and non-turbo `z-image.inpaint`, plan-time maskless rejection, the canonical `docs/masked-editing.md` page, and a published visual-smoke proof bundle with preservation measurements. |

## Deprecated ledger

| ID | Item | Deprecated | Reason |
| --- | --- | --- | --- |
| 0047 | [SeedVR2 7B quality and safe video scale revalidation](deprecated/0047_seedvr2_7b_quality_and_safe_video_scale_revalidation.md) | 2026-06-21 | Release `0.18.20` closed the bounded 7B quality revalidation through completed item 0032; the only remaining question was split into proposed item 0048. |

## Recurrent ledger

| ID | Item | Area | Cadence |
| --- | --- | --- | --- |
| 0017 | [Backlog release-state hygiene](recurrent/0017_backlog_release_hygiene.md) | Backlog, release state, follow-up triage | After each release, large validation run, or priority-changing item move. |

## Process

- New backlog items must use a unique four-digit global prefix.
- Planned items need current code reality, scope, non-goals, validation, and ADR status.
- New model routes must follow [ADR 0001](../adr/0001_runtime_smoke_validation_for_model_routes.md):
  do not mark support as working or release-ready without a model-backed smoke command and output
  evidence, or an intentional unsupported-state failure.
- Model, backend, architecture, task, and quantization resolution must follow
  [ADR 0002](../adr/0002_no_silent_automatic_fallbacks.md): ambiguous or unsupported requests fail
  closed unless fallback behavior is explicitly requested by the caller.
- When implementation completes, move the item to `docs/backlog/completed/` and append completion
  evidence instead of deleting the planning history.
- If a backlog item establishes lasting architecture policy, create or update an ADR before
  closure.

## Planning notes

- Created the backlog system on 2026-05-25 while triaging local and online model integration
  candidates for MLX-Gen and AbstractVision.
- Refined the model roadmap on 2026-05-25 after checking Hugging Face model sizes, licenses,
  local cache state, and current T2I/I2I/T2V/I2V popularity.
- Added 2026-05-25 AbstractFramework publication audit note: checked Qwen, FLUX.2 Klein/Base,
  Z-Image, and Z-Image-Turbo q4/q8 repos are complete; future work should automate this audit.
- Added 2026-05-25 collection follow-up: several q8 and non-turbo Z-Image repos still need to be
  added to the Hugging Face `AbstractFramework / mlx-gen` collection once collection write
  permission is available.
- Added 2026-06-25 Krea watch item: `krea/Krea-2-Turbo` looks technically feasible enough for a
  future MLX spike, but its custom Krea 2 Community License is materially restrictive
  (revenue-capped, revocable, redistribution-constrained), so the item stays proposed and low
  priority.
- Added 2026-06-27 code-only audit hardening band after two adversarial subagents confirmed the
  unsafe Torch checkpoint loader, fail-open local release path, prepare class/config split, step
  validation/default drift, low-RAM repeat-generation hazard, Wan/VLM/BF16 performance issues, and
  package hygiene follow-ups.
- Added 2026-05-25 ERNIE follow-up: ERNIE Image Turbo text-to-image support exists with BF16,
  q8, q4, and optional Prompt Enhancer validation; remaining work is parity coverage and
  non-turbo ERNIE-Image.
- Added 2026-05-26 Wan follow-up: initial Wan2.2 TI2V text-to-video and first-frame
  image-to-video support exists with MP4 output, focused tests, and opt-in full-model parity
  fixtures for the Wan transformer, VAE encoder/decoder, prompt embeddings, scheduler replay, and
  a tiny latent-only CFG denoise loop; remaining work is decoded-video quality validation,
  quantized validation, cancellation APIs, and broader full-generation Diffusers parity. Shared
  progress callbacks were completed later in item 0014.
- Added 2026-05-27 Wan focused item: fixed the q8 prepare blocker caused by unconditional LoRA
  kwargs, prepared and smoke-tested a q8 Wan folder locally, analyzed the user's three 121-frame
  videos for motion, and split remaining q8 quality/q4 policy work into planned item 0002.
- Added 2026-05-27 Wan q8 note: a same-settings 704x384, 25-frame, 12-step comparison found q8
  visually close to BF16/source but slower in that smoke run; peak runtime memory was not captured.
- Added 2026-05-27 Bonsai planning split: ternary 2-bit is planned as a FLUX.2-compatible packed
  loader; binary 1-bit is proposed/deferred because the local MLX runtime did not support the
  required 1-bit packed affine matmul probe.
- Added 2026-05-27 Wan performance item: q8 prepared output was visually close in the short
  comparison but took 217.4s versus 95.48s for source BF16, so a dedicated timing/memory
  investigation now tracks that abnormal slowdown.
- Added 2026-06-04 TI2V-5B clean memory clarification: upstream stores FP32 transformer/VAE plus a
  BF16 text encoder, while MLX-Gen loads transformer/VAE weights at BF16 runtime precision.
  Prepared BF16 reduces storage/download size but not runtime memory; mixed q8/BF16 reduces storage
  and MLX model/allocator footprint but did not reduce full-process physical peak in the 1280x704
  clean profile.
- Added 2026-06-04 Wan memory-metric cleanup: docs now distinguish storage, Wan MLX model bytes,
  MLX active-after-generation bytes, full-process physical footprint, max RSS, and MLX allocator
  peak. Low-RAM mode now requests transformer-block cache boundaries in addition to step/decode
  cleanup.
- Added 2026-06-04 Wan memory-profile correction: TI2V-5B `~103 GiB` physical peaks are historical
  1280x704 normal-cache validation rows and are not comparable to the A14B 384px-class low-RAM
  table as model-size evidence. A14B metric JSONs are now tracked under
  `docs/assets/quantization/wan-a14b-lowram/`. Runtime cleanup was tightened with explicit CFG
  prediction materialization, VAE temporal-slice cache clearing, and indexed prepared-shard loading.
- Added 2026-06-09 Wan TI2V-5B parity follow-up: M5 Max practical clips now favor A14B at
  `480x240`, 25 steps, 101 frames, and 20 fps over TI2V-5B at `832x480` and `1280x704` for the
  starship-takeoff prompt. Planned item 0035 tracks a source-model math and behavior comparison
  against official Wan plus local Diffusers/Transformers before drawing model-quality conclusions.
- Added 2026-06-10 FLUX.2 Klein base validation follow-up: source-model base `4B/9B` now have
  cropped-starship text/edit/strict-outpaint proof and a source-only validation profile. Planned
  item 0036 keeps prepared base q8/q4 starship proof separate until those package rows are
  actually generated and reviewed.
- Added 2026-06-12 Qwen and Wan expansion follow-up: promoted Qwen structured control/inpaint
  work into planned item 0008, added Wan VACE as proposed item 0039, split LightX2V fast-video
  work into planned item 0040 for LoRA-based 4-step A14B acceleration and proposed item 0041 for
  native distilled-model loader support.
- Added 2026-06-15 opportunity refresh: updated the video-roadmap proposals to track the current
  LTX family rather than only older LTX-2.3 framing, refreshed the image-edit watchlist with
  Step1X-Edit, JoyAI, OmniGen2, and current HiDream/FLUX Kontext reality, tightened Wan VACE to
  stay VACE-scoped despite newer Wan Animate/S2V releases, and added item 0043 for native
  Z-Image inpaint.
- Added 2026-06-21 post-0.18.20 SeedVR2 hygiene: promoted item 0046 to active implementation
  because the shipped restore route already had explicit accepted proof and the audio gap was the
  next concrete delivery task; deprecated mixed-scope item 0047 because the bounded 7B-quality
  revalidation was complete; and added proposed item 0048 for the narrower question of whether
  enlarged SeedVR2 video should ever become a safe documented public profile.
- Added 2026-06-21 Boogu watch follow-up: created proposed item 0049 after a focused public-source
  audit found Boogu interesting for bilingual text rendering and unified generation/editing, but
  still too early and too custom to outrank the active Qwen, Z-Image, and Wan work.
- Added 2026-06-21 audio-copy closure: completed item 0046 after the shared post-write SeedVR2
  audio copy-through path, focused automated coverage, and the published Air France `25s–35s`
  proof bundle landed.
- Added 2026-06-12 post-0.18.17 hygiene: the release succeeded, LightX2V/Wan proof assets and
  public LoRA docs were refreshed, and planned item 0042 now tracks the GitHub Actions Node 20
  deprecation warning observed in release run `27440684820` before the Node 24 default switch.
- Completed 2026-06-12 GitHub Actions Node 24 migration: PR `#4` upgraded the affected actions,
  release rehearsal `27443742691` passed without the earlier Node 20 deprecation warnings, and PR
  CI run `27443720109` exposed only pre-existing repository lint drift unrelated to the migration.
- Added 2026-06-13 follow-up: release `0.18.18` still emitted one Node 20 warning from
  `softprops/action-gh-release@v2`, so item 0042 was reopened and narrowed to a final GitHub
  Release publication cleanup pass using the `gh` CLI as the reusable migration strategy.
- Completed 2026-06-14 GitHub Actions Node 24 migration follow-up: PR `#6` replaced
  `softprops/action-gh-release@v2` with a `gh release create` / `gh release upload --clobber`
  step, PR checks passed, and release rehearsal `27464789730` succeeded on the branch.
- Completed 2026-05-27 Bonsai ternary 2-bit support: `mlxgen generate` can run
  `prism-ml/bonsai-image-ternary-4B-mlx-2bit` directly, binary 1-bit now fails with an explicit
  unsupported-runtime message, and a local validation panel compares Bonsai ternary with FLUX.2
  Klein 4B q8.
- Rechecked 2026-05-27 stock MLX 0.31.2 for Bonsai binary 1-bit. `bits=1, group_size=128`
  quantization still fails while 2/3/4/5/6/8 succeed, so proposed item 0004 remains deferred and
  the Bonsai ternary release can proceed.
- Added 2026-05-30 Wan A14B priority item after confirming local T2V-A14B cache and then the full
  I2V-A14B cache. A14B is tracked separately from 5B quantization because it needs two-transformer
  boundary routing, scalar timesteps, `flow_shift=3.0`, and a Wan2.1-style VAE.
- Added 2026-05-30 ADR 0001 after the A14B route exposed a validation gap: new model routes now
  require model-backed smoke proof before support or release-readiness claims. The A14B T2V route
  has source-checkpoint MP4 smoke evidence; I2V-A14B has source-conditioned MP4 smoke evidence
  from the complete local I2V snapshot.
- Added 2026-05-30 ADR 0002 after Wan model resolution exposed silent default-config fallback risk:
  model identity, backend, architecture, task, and quantization resolution now fail closed unless a
  caller explicitly opts into fallback behavior.
- Completed 2026-05-31 Wan2.2 A14B T2V/I2V initial support. Remaining Wan work is tracked in the
  quantization, motion parity, and q8 performance items rather than in the A14B wiring item.
- Added 2026-06-02 Wan A14B boundary-memory item after a full-size I2V run appeared to stop at
  `denoise step 6/20`, which maps exactly to the high-noise/low-noise denoiser boundary for
  I2V-A14B at 20 steps.
- Added 2026-06-02 Wan T2V-A14B q8 validation: full q8 collapsed to near-black/static video, so
  Wan q8 now keeps `condition_embedder.*` and `proj_out` BF16. The rebuilt
  `models/wan2.2-t2v-a14b-diffusers-8bit` folder is about 40 GiB and passes the preserved
  384x224, 17-frame contact-sheet check, but short-run peak RSS did not improve versus BF16.
- Added 2026-06-02 Wan memory/speed follow-up: component-wise prepared-q8 loading produced a
  byte-identical sample but did not materially reduce the 19.5 GiB prepared-q8 peak RSS. The video
  conversion path now builds PIL frames one at a time to avoid full-video NumPy temporaries, which
  should matter most for 81/121-frame outputs.
- Added 2026-06-02 Wan package-level memory follow-up: RSS alone under-reported source-package
  memory pressure. MLX peak memory for the same 384x224 A14B T2V sample was 32.99 GiB from the
  upstream source package, 33.11 GiB from the BF16 prepared package, and 20.84 GiB from the mixed
  q8/BF16 prepared package. The BF16 prepared output was byte-identical to source; the mixed
  package is the first measured usage-memory reduction.
- Completed 2026-06-03 shared progress callbacks: image and video generation now use one
  lightweight `ProgressEvent`, image callers can subscribe through
  `model.callbacks.subscribe_progress(...)`, Wan still supports direct `progress_callback`, and
  CLI video progress advances by denoising step. Focused validation passed with 11 progress-related
  tests.
- Added 2026-06-04 first-class I2I mode and outpaint/reframe item after reviewing current routing
  semantics and FLUX Fill support. Public tasks should remain media-direction based, while
  latent img2img, edit-conditioned I2I, multi-reference I2I, inpainting, and outpainting become
  explicit internal modes or workflow options.
- Added 2026-06-04 I2I source aspect-ratio item after the Wan I2V canvas policy exposed the same
  geometry risk for ordinary image-to-image: explicit mismatched dimensions should be size targets
  for latent/edit/multi-reference I2I, not silent source-stretch instructions.
- Added 2026-06-03 Wan prompt-adherence parity item after source review confirmed MLX-Gen's Wan
  negative-prompt and A14B guidance defaults match official Wan recommended config semantics, while
  raw Diffusers omitted-argument defaults differ. No heavy validation was run because a long Wan
  generation was active.
- Added 2026-06-03 Wan video integrity release gate after a full-size T2V-A14B mixed q8/BF16 run
  completed after about 13h15m but encoded an all-black MP4 from non-finite decoded values. The q8
  A14B card must stay validation-sized until exact full-size integrity and quality validation passes;
  early tensor-health checks, CLI completion semantics, default video-health save validation, and
  failure manifests are now implemented.
- Ran 2026-06-04 post-release backlog hygiene after `mlx-gen` 0.18.9: promoted LoRA strictness to
  planned work because loader-level best-effort behavior conflicts with ADR 0002 and public docs,
  narrowed Wan A14B boundary-memory work to the remaining full-size I2V retry evidence, kept the
  Wan video integrity gate planned for release artifacts/full-size revalidation, and added recurrent
  backlog release-state hygiene.
- Completed 2026-06-04 taskless generation routing: `mlxgen generate` now infers common T2I/I2I/T2V/I2V
  tasks from model plus image inputs, keeps `--task` as an explicit override, rejects contradictory
  task/image shapes early, exposes public Python `resolve_task`/`infer_task`, and updates normal Wan
  docs and generated model-card usage snippets to omit unnecessary `--task`.
- Added 2026-06-04 generation capability contract item after image-edit routing review showed that
  `edit` must become an internal I2I mode, `--image-strength` must be restricted to latent img2img,
  and apps need a public capability/plan API before launching expensive generations.
- Completed 2026-06-04 generation capability contract: added typed capability descriptors,
  `GenerationPlan`, public `get_model_capabilities(...)` / `resolve_generation_plan(...)`,
  `mlxgen capabilities`, `--i2i-mode`, metadata-aware latent route replay, local-path
  `--base-model` handling, early option rejection, image edit progress labeling, docs, and
  lightweight route-plan artifacts. Focused resolver/router/progress validation passed with 86
  tests, then reviewer-driven metadata/base-model/mask/family-conflict coverage expanded the
  focused suite to 91 passing tests.
- Updated 2026-06-04 pre-0.18.10 backlog release state: current code has the taskless/capability
  planner in completed items 0018 and 0020, with docs now explicitly distinguishing latent
  img2img, edit/reference I2I, multi-reference I2I, `--image-strength`, and Wan video size rules.
  First-class outpaint/reframe remains planned in item 0019, and full-size Wan A14B q8 readiness
  remains gated by item 0016 rather than claimed in this release.
- Added 2026-06-04 TI2V-5B clean quantization validation: source and prepared BF16 are
  byte-identical at 1280x704, 17 frames, 20 steps with `--negative-prompt ""`; the mixed q8/BF16
  package stays visually in-family with mean frame MAE 1.66. Public docs now link the MP4s, contact
  sheet, and metrics. Residual Wan work remains full-duration validation, q4 policy, A14B/I2V
  exact-setting checks, and release-gated full-size claims.
- Completed 2026-06-04 Wan I2V source aspect-ratio preservation: image-to-video dimensions now
  resolve from the input image ratio before conditioning, metadata records requested/source/resolved
  sizes, and local TI2V-5B plus A14B proof videos confirm square source images no longer stretch
  into a requested 16:9 canvas.
- Completed 2026-06-04 I2I source aspect-ratio policy and follow-up compatibility hardening:
  ordinary latent, edit/reference, and multi-reference image-to-image now use `source-aspect` by
  default with `exact-resize` as an explicit opt-in. Base FIBO no longer exposes unvalidated I2I,
  base Qwen Image requires explicit latent `--image-strength`, and local route tests plus a real
  generated I2I matrix cover runnable FLUX.2, Qwen Edit, Qwen latent, Z-Image Turbo, and ERNIE
  packages.
- Added 2026-06-05 I2I capability validation matrix after a partial validation sheet showed that
  one Qwen edit crash proof failed visually and that several advertised model/mode pairs were not
  covered in one complete evidence table. Item 0023 now gates image edit release-readiness on a
  model, capability, source, prompt, result, and visual-status table.
- Corrected 2026-06-05 I2I validation state after manual review: completed item 0023 should not be
  cited as complete complex-edit proof because Qwen hard-landing/crash behavior was not proven and
  older rows mixed source/reference assets.
- Completed 2026-06-05 standardized I2I sequence validation in item 0025: all 30 commands ran
  successfully, but visual QA found FLUX.2 Klein q4/q8 is the only tested family that passed the
  full B/C/D/E sequence; Qwen Edit 2511 remains unreliable for complex crash composition, Qwen
  Image 2512 latent I2I lost spaceship identity at the tested strength, and FIBO Edit was still
  excluded at the time of that matrix pending planned item 0024.
- Advanced 2026-06-05 FIBO Edit repair under planned item 0024: completed item 0026 and the
  follow-up Diffusers baseline showed that FIBO Edit did not produce acceptable images in the
  current validation environment. Unified capability discovery now exposes no FIBO Edit route;
  base FIBO remains text-to-image only. Planned item 0027 tracks Diffusers parity, source-route
  visual recovery, and any future re-enablement criteria.
- Completed 2026-06-06 Qwen Image Edit 2511 parity: MLX-Gen now uses the model-config FlowMatch
  dynamic-shift scheduler contract for Qwen Image/Edit, and Qwen Image Edit 2511 source/q8/q4 pass
  the focused pencil sketch, hard-landing edit, and multi-reference composition proof profile.
- Completed 2026-06-07 SeedVR2 upscaler smoke: the cached 3B q8 route produced valid
  `640x384` and `960x576` scale-factor upscales from a `320x192` source, and a `426x256`
  target-short-edge restoration smoke. SeedVR2 now records final output dimensions and source-image
  metadata correctly, wires the CLI `--metadata` flag to JSON sidecar export, defaults to untiled
  VAE processing for image quality, and exposes `--vae-tiling` as an explicit memory-saving opt-in.
- Completed 2026-06-07 SeedVR2 official package support: `seedvr2`/`seedvr2-3b` now route to the
  official `ByteDance-Seed/SeedVR2-3B` source model, `seedvr2-7b` routes to
  `ByteDance-Seed/SeedVR2-7B`, `mlxgen prepare` can write reusable SeedVR2 q8/q4 packages, and the
  source/q8/q4 5x profiles are documented with storage, timing, memory, and contact-sheet evidence.
- Ran 2026-06-07 backlog priority refinement after the 0.18.13 release: FIBO Edit parity and
  unified validation remain planned but deferred, first-class outpaint/reframe is now the top
  image-routing priority, LoRA strictness now includes task-direction capability metadata, SeedVR2
  video restoration/upscaling was added as proposed item 0032, and T2V/I2V video LoRA was split
  into item 0033.
- Refined 2026-06-07 item 0019 after the FLUX.1 support concern: initial implementation and
  validation starts with generative reframe and canvas outpaint for FLUX.2, then Qwen Image Edit
  2511, while Z-Image/ERNIE remain lower-confidence candidates and native fill/inpaint outpaint
  waits for a proven fill/mask backend.
- Ran 2026-06-08 LoRA backlog refinement after the 0.18.14 release. Item 0007 now treats LoRA as
  a task/mode capability, requires fail-closed runtime loading, requires strict scale counts,
  blocks unsupported-family LoRA before model load, and separately gates `mlxgen prepare
  --lora-paths` for q4/q8 packages. Item 0033 remains video-only work and requires explicit Wan
  target roles plus MP4 A/B proof before each exact row is promoted.
- Added 2026-06-08 FLUX.2-dev multi-angle LoRA item 0034 after checking the downloaded
  `lovis93/Flux-2-Multi-Angles-LoRA-v2` model card and local matrix shapes. The adapter targets
  `black-forest-labs/FLUX.2-dev`, so it is not valid FLUX.2 Klein proof. Current code rejects the
  adapter for Klein and keeps first-class FLUX.2-dev support as a separate planned item.
- Updated 2026-06-11 Qwen Image Edit 2509 LoRA parity under item 0007: MLX-Gen had a real 2509
  modulation-key mapping gap for `img_mod.1` / `txt_mod.1` `default.weight` tensors, which is now
  fixed in the Qwen LoRA mapping. The exact `AbstractFramework/qwen-image-edit-2509-8bit`
  `qwen.edit` row is now validated with the stacked `lightx2v/Qwen-Image-Lightning` plus
  `dx8152/Qwen-Edit-2509-Multiple-angles` path on the proper Lightning profile (`8` steps,
  `guidance 1`).
- Updated 2026-06-11 original Qwen Image Edit LoRA follow-up under item 0007:
  `AbstractFramework/qwen-image-edit-8bit` now has an accepted exact q8 single-image edit proof
  using the local `ghibli_style_qwen_v3.safetensors` adapter. The route matches `1680/1680`
  tensors, applies `840` targets, and is now treated as validated for `qwen.edit`.
- Updated 2026-06-11 Qwen Image 2512 q8 LoRA parity under item 0007: the public
  `prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA` adapter exposed a missing
  `diffusion_model.transformer_blocks.*.img_mod.1` / `txt_mod.1` mapping family. After fixing that
  Qwen modulation-key gap, `AbstractFramework/qwen-image-2512-8bit` now has an exact validated
  `qwen.text` row with a visible pixel-art A/B proof.
- Updated 2026-06-11 base Qwen Image q8 LoRA follow-up under item 0007: the local
  `AbstractFramework/qwen-image-8bit` package is now complete, and the exact-base
  `flymy-ai/qwen-image-realism-lora` adapter loads cleanly (`480/480` matched keys, `240` targets
  applied). The later June 22 route-completion pass closed the remaining adherence gap with a
  separate exact latent proof on the same public q8 package.
- Added 2026-06-22 Qwen/LoRA completion note: item 0008 now ships the first-class Qwen route
  matrix and the production-grade Qwen route family, and item 0007 closes the remaining current
  image LoRA gaps with accepted proof rows for base Qwen latent, Qwen 2511 multi-reference,
  reframe, and outpaint, FLUX.2 Klein 9B multi-reference, FLUX.2 Klein base 4B outpaint, plus the
  earlier same-day Z-Image and ERNIE latent rows. FLUX.2 latent stayed out of the public support
  set because the observed effect was still too marginal for an honest production claim.
- Ran 2026-06-11 post-release backlog refinement: promoted item 0008 from proposed to planned
  because official Qwen control/inpaint pipelines and public control weights make structured Qwen
  control a concrete next-step inside an already-supported family, added proposed item 0039 for
  Wan VACE video editing/control, and raised SeedVR2 video restoration from a distant idea to a
  near-term follow-up because the official SeedVR2 models are published as video-to-video
  restoration checkpoints.
