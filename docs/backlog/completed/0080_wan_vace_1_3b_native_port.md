# 0080 - Wan2.1-VACE-1.3B Native MLX Port

- Status: completed (2026-07-06)
- Scope: native VACE conditioning (reference images + learned masks) on
  `Wan-AI/Wan2.1-VACE-1.3B-diffusers`; closes the runtime half of proposed items 0039/0075
  for the 1.3B rung
- Why Wan2.1: VACE is a Wan2.1-generation release - the only official Wan-AI VACE
  checkpoints are Wan2.1-VACE-1.3B/14B; no official Wan2.2 VACE exists (the community
  `alibaba-pai/Wan2.2-VACE-Fun-A14B` is a 64 GB-class PAI fine-tune, a possible later rung).
  This is the first Wan2.1 model in the runtime; it reuses the wan21 VAE and UMT5 stack the
  A14B route already ships.
- Proof: `docs/assets/validation/wan-vace-2026-07-06/` (committed mirror: README with exact
  commands, labeled request/inputs/output panels with controls - masked edit + upstream
  same-inputs A/B, reference-injection identity ablation, the full-scene-reference failure
  mode - corrected-case MP4s + sidecars, masks/references, bf16 + fp32 parity comparison
  JSONs); working bundle in `validation_outputs/wan_vace_mlx_2026_07_06/` and parity tensors
  in `validation_outputs/wan_vace_parity/`
- Validation: 4 adversarial subagents - design attack (pre-implementation; caught a
  planner-crashing task value, the strength auto-injection, and the multi-ref ordering
  reversal), parity verification (verdict: parity PROVEN - mask preparation and UniPC
  scheduler bit-exact vs diffusers, transformer deltas at the model's measured intrinsic
  fp32 noise floor via a noise-injection probe, CFG-loop amplification matching the analytic
  model), code verification, and a result-quality judge; full no-weights band green
  (1289 passed), lint clean

## What shipped

1. Model config `wan2.1-vace-1.3b` (single Wan2.1 transformer, 30 layers, dim 1536, ffn 8960,
   wan21 16ch VAE, UMT5-XXL shared with A14B, UniPC flow shift 3.0, defaults 832x480/81f/
   30 steps/guidance 5.0/fps 16; `task: text-to-video` deliberately - VACE supports pure T2V,
   and the planner builds the video-video capability from `supports_video_to_video`).
2. Transformer: `WanTransformer` gains optional `vace_layers`/`vace_in_channels`
   (vace patch embedding + 15 `WanVACETransformerBlock`s chained on the control stream,
   emitting per-layer hints added after each mapped main block; strict guard that control is
   provided exactly when configured). Weight mapping/definition extended for the 439
   `vace_*` checkpoint keys (verified against the safetensors index).
3. Runtime `WanVace` (subclasses the Wan family runtime; overrides generate_video):
   conditioning preprocessing ported exactly from the diffusers pipeline - binarized
   inactive/reactive VAE encode (mode), raw-mask 8x8 rearrange + nearest-exact temporal
   resample to 64 channels, white-canvas letterboxed reference images with the reference
   prepend order preserved (last reference lands first), main noise gains one latent frame
   per reference and drops them before decode. Rejects `video_strength`, `guidance_2`,
   `image_path`, non-unipc solvers.
4. CLI: `mlxgen-generate-wan` gains `--reference-image` (repeatable) and
   `--conditioning-scale`, routes VACE configs to the new runtime, validates VACE-only flags
   against non-VACE models (and vice versa) before weight load, replays both fields from
   metadata, and no longer injects the 0.8 strength default on VACE. Router passes
   `--reference-image` through unconsumed; the planner reports
   `supports_video_strength=False` for VACE video-video (capability data only, no schema
   change).
5. Parity harness: `tools/wan_vace_parity_export.py` (diffusers CPU fp32 stage exports on a
   48x80x9f case with one reference) + `tools/wan_vace_parity_compare.py` (MLX side).
   The mask-channel math is additionally pinned bit-exact in `tests/wan/test_wan_vace.py`
   across random time-varying masks x 0/1/2 references.
6. Measured proof cases (corrected twice - 2026-07-07 - after adversarial vision review
   rejected the first attempts; six subagents ran the second cycle):
   - Masked edit, GENERATE mode (the fix): a vision judge ruled every earlier output
     "repaint-only" (silhouette IoU 0.73-0.88 vs ~0.39 for real geometry change); the
     engineer agent traced the root cause to the official VACE inpainting convention -
     ali-vilab gray-fills the editable region before encoding (UserGuide "gray areas
     represent missing video part"; maintainer-confirmed in issue #107), while we fed the
     source pixels into the reactive branch, which the model reads as "repaint". Runtime now
     gray-fills by default (`--vace-masked-region generate`), keeping `repaint` as the
     restyle-in-place mode. Result after a second vision-judged iteration (tight corridor
     mask instead of the over-wide union mask, bulk-explicit prompt, 24 steps, guidance
     5.0): the ship is REPLACED - silhouette IoU vs source 0.16-0.20 (repaint band
     0.73-0.88), outgrowth 0.41-0.64, in-mask change 63.8, background 3.8-3.9 (codec floor)
     at 1027 s / 12.1 GiB; repaint mode measures IoU 0.80-0.88 / in-mask 14-17. Upstream A/B
     on un-blanked inputs fails exactly like repaint while drifting the whole frame (2161 s
     torch CPU). Runtime forensics: 7/7 plumbing checks passed.
   - Reference-injection identity proof by same-seed control: with a subject SEGMENTED onto
     white, the generated ship IS the reference subject (independent vision judge: all
     decisive features match at every frame, down to the nose accent and hull light);
     without the reference (same seed and prompt), an unrelated ship. First-attempt
     pitfalls kept as documented guidance: a source frame passed as reference suppresses
     edits, a background crop yields only style-level identity, a full-scene reference
     loses the subject.
   - Flag-free defaults run (832x480x81f/30 steps): 6948.6 s, 31.7 GiB peak - default cost
     measured, not extrapolated.

## Follow-ups (honest gaps)

- Multi-reference and person/identity injection cases not yet proven (single-object case
  only; the same-seed no-reference identity ablation is done and in the bundle).
- Reference segmentation is manual today (the bundle used a luminance flood-fill); a
  documented helper or guidance for background removal would improve first-use success.
- Soft/per-frame masks: the CLI mask contract is one static binarized image; the reference
  pipeline accepts continuous per-frame masks. Extend only with a proof case.
- q8 quantization policy for `vace_blocks.*.proj_in/proj_out` undecided; the port ships BF16.
- Conditioning latents are recomputed per seed (no keyed cache yet, unlike A14B V2V).
- Control conditions (pose/depth videos) remain out of scope until a dependency-light
  preprocessing story exists (Depth Pro is in-repo and is the natural first control).
