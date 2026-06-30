# Runtime Contracts Validation Report

- Date: 2026-06-29
- Scope: backlog items `0067` through `0070`
- Evidence root: `validation_outputs/runtime_contracts_2026_06_29/`

## What changed

- `0067`: hardened progress semantics so image and video routes emit truthful public tasks, `after_loop()` is no longer public success, and terminal progress is explicit through `complete()` / `failed()`.
- `0068`: fixed the no-LoRA Qwen control/control-inpaint metadata crash and preserved the base-Qwen negative-prompt contract on control routes.
- `0069`: corrected Z-Image CFG math and reran the published native-inpaint engine proof on the same source, mask, prompt, seed, and step count.
- `0070`: separated trusted exact prepared identities from fuzzy family inference, then fixed the CLI/public-capability handoff so pre-resolved configs cannot silently upgrade spoofed ids into trusted variant routes.

## Real proofs

- Qwen control-inpaint success proof:
  `validation_outputs/runtime_contracts_2026_06_29/qwen_control_inpaint_q8_4step_fix.{png,metadata.json,progress.json,stats.json}`
  Result: `image-to-image`, phases `start -> denoise x4 -> complete`, wall `9.60s`, generation `9.17s`, peak RSS `34.07 GB`.

- Z-Image Turbo same-case native-inpaint rerun:
  `validation_outputs/runtime_contracts_2026_06_29/zimage_turbo_inpaint_engine_samecase_fix.{png,metadata.json,progress.json,stats.json}`
  Result: same published source/mask/prompt/seed/steps as the original accepted engine proof, phases `start -> denoise x9 -> complete`, wall `5.59s`, generation `5.33s`, peak RSS `18.09 GB`.
  Historical comparison target:
  `docs/assets/validation/zimage-inpaint-2026-06-21/zimage_inpaint_stats_m5max.json`
  Comparison note: peak RSS stayed at the same scale (`18.09 GB` vs historical `18.11 GB`). This rerun is correctness evidence, not a claim of generalized speed improvement.

- Z-Image base guided latent sanity proof:
  `validation_outputs/runtime_contracts_2026_06_29/zimage_base_q8_latent_guided_4step_fix.{png,metadata.json,progress.json,stats.json}`
  Result: base `AbstractFramework/z-image-8bit` guided latent route completed on the real prepared package, task `image-to-image`, wall `1.84s`, generation `1.61s`, peak RSS `11.55 GB`.

- FLUX.2 Klein base edit proof:
  `validation_outputs/runtime_contracts_2026_06_29/flux2_klein_base4b_q8_edit_4step_fix.{png,metadata.json,progress.json,stats.json}`
  Result: `image-to-image`, phases `start -> denoise x4 -> complete`, wall `2.18s`, generation `2.12s`, peak RSS `8.17 GB`.

- SeedVR2 image-upscale proof:
  `validation_outputs/runtime_contracts_2026_06_29/seedvr2_3b_q8_image_2x_fix.{png,metadata.json,progress.json,stats.json}`
  Result: `image-to-image`, phases `start -> denoise -> complete`, wall `0.63s`, generation `0.60s`, peak RSS `4.84 GB`.

- SeedVR2 streamed video-restore proof:
  `validation_outputs/runtime_contracts_2026_06_29/seedvr2_3b_q8_streamed_480x240_41f_fix.{mp4,metadata.json,progress.json,stats.json}`
  Result: real `restore_video_to_path()` run, `41` frames at `480x240`, `29/8` chunking, `2` streamed chunks, task `video-to-video`, phases `start -> denoise -> denoise -> complete`, wall `24.31s`, generation `23.83s`, peak RSS `5.15 GB`.

- Public CLI capabilities/spoof proof:
  `validation_outputs/runtime_contracts_2026_06_29/capabilities_spoof_proof.{json,commands.txt}`
  Result: `all_cases_match_expected=true`; trusted exact prepared ids kept their expected route surface, remote-looking spoof ids stayed conservative, and explicit-base local spoof ids also stayed conservative.

## Focused regression tests

- `uv run python -m pytest tests/resolution/test_config_resolution.py -q`
  Result: `32 passed`

- `uv run python -m pytest tests/test_task_inference.py -q`
  Result: `43 passed`

- `uv run python -m pytest tests/cli/test_mlx_gen_router.py -q -k 'capabilities_command_reports_model_modes or capabilities_command_accepts_base_model_for_local_paths or capabilities_command_remote_like_spoofs_stay_conservative or family_override_flux2_local_path_requires_base_model or family_override_flux2_local_path_forwards_base_model or qwen_backend_omitted_negative_prompt_stays_none or qwen_backend_explicit_empty_negative_prompt_is_preserved or qwen_backend_forwards_mask_path_to_control_inpaint or qwen_backend_forwards_controlnet_strength_to_control_inpaint'`
  Result: `9 passed`

- `uv run python -m pytest tests/callbacks/test_progress_callbacks.py -q`
  Result: `11 passed`

- `uv run python -m pytest tests/image_generation/test_masked_generation_routes.py -q`
  Result: `8 passed`

- `uv run python -m pytest tests/seedvr2/test_seedvr2_progress.py -q`
  Result: `5 passed`

- `uv run python -m pytest tests/seedvr2/test_seedvr2_video_chunking.py -q -k 'requires_audio_copy_by_default or cleans_final_file_on_postwrite_validation_failure'`
  Result: `2 passed`

## Conclusion

- The runtime-contract band is closed on code, focused tests, and preserved real-case evidence.
- No memory optimization that changes model math was introduced for these fixes. The changes are contract/correctness repairs, plus one Z-Image CFG math correction backed by a focused math regression and refreshed real reruns.
- Remaining documented caution: progress callback exceptions still propagate by design, so handlers must stay small and defensive. The public docs now state that behavior explicitly.
