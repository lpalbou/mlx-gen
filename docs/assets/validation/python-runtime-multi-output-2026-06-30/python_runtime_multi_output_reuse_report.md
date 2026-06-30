# Python Runtime Multi-Output Reuse Validation

- Date: 2026-06-30
- Scope: unified Python `mlxgen` runtime helpers for `mlxgen generate` families
- Evidence root: `validation_outputs/python_runtime_multi_output_20260630/`

## What was validated

- Public Python multi-output execution uses `load_generation_model(...).generate_outputs(...)`.
- The contract is serial singular reuse, not tensor batching: one loaded model instance runs one
  seed at a time.
- `overwrite=False` preserves existing targets by resolving a unique final path before save.
- SeedVR2 remains outside this Python wrapper; the proof here covers unified `generate` families
  only.

## Focused contract tests

- `uv run pytest tests/test_python_runtime.py -q`
  Result: `11 passed`

These focused tests now cover:

- duplicate-seed rejection;
- reuse of one loaded model instance across several seeds;
- image and video save contracts;
- `overwrite=False` preservation for image and video outputs;
- per-seed collision suffixing on multi-output image runs.

## Real reuse-vs-reload proofs

Each case ran twice:

- `reuse`: one `load_generation_model(...)` followed by `generate_outputs(...)` for two seeds;
- `reload`: a fresh `load_generation_model(...)` per seed, then `generate_output(...)`.

### Qwen masked edit

- Model: `AbstractFramework/qwen-image-edit-2511-8bit`
- Route: Qwen masked edit on one source image with one generated binary mask
- Profile: `768x432`, `20` steps, guidance `4`, seeds `4201` and `4202`
- Result: exact per-seed image parity
- Wall time: reuse `393.84s`, reload `397.01s`, improvement `0.80%`
- Peak RSS delta: reuse `1.35%` lower
- Peak physical delta: reuse `0.18%` higher
- Save tail: about `0.16s` to `0.19s` per image after `save`

Conclusion: the Python wrapper is correct on the real masked-edit route, but this route does not
show a meaningful speed win from serial reuse on the measured machine. It does avoid a memory
regression.

### FLUX.2 multi-reference edit

- Model: `AbstractFramework/flux.2-klein-9b-8bit`
- Route: FLUX.2 multi-reference edit
- Profile: `432x240`, `20` steps, guidance `1.0`, seeds `8614` and `8615`
- Result: exact per-seed image parity
- Wall time: reuse `35.83s`, reload `43.34s`, improvement `17.34%`
- Peak RSS delta: reuse `1.73%` lower
- Peak physical delta: reuse `1.50%` lower
- Save tail: about `0.05s` to `0.06s` per image after `save`

Conclusion: serial reuse is materially beneficial on the FLUX.2 edit path and keeps quality exact.

### Wan A14B image-to-video, recurring short profile

- Model: `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit`
- Route: Wan A14B image-to-video
- Profile: `448x256` requested, resolved `336x336`, `17` frames, `8` steps, guidance `4 / 3`,
  seeds `6601` and `6602`
- Result: exact per-seed video parity by frame comparison
- Wall time: reuse `192.27s`, reload `299.49s`, improvement `35.80%`
- Peak RSS delta: reuse `10.22%` lower
- Peak physical delta: reuse `0.36%` lower
- Save tail: about `0.53s` to `0.88s` per video after `save`

Conclusion: this is the strongest current proof that the Python wrapper is valuable for a real
embedding-app warm-worker path.

### Large-resolution image finalization

- Model: `AbstractFramework/z-image-turbo-8bit`
- Route: Z-Image Turbo text-to-image
- Profile: `1024x1024`, `8` steps, seeds `7401` and `7402`
- Result: exact per-seed image parity
- Wall time: reuse `80.58s`, reload `82.04s`, improvement `1.78%`
- Peak RSS delta: reuse `2.11%` lower
- Peak physical delta: reuse `5.51%` lower
- Save tail: about `0.11s` to `0.13s` per image after `save`

Conclusion: large-output image finalization does not show a meaningful compute win from reuse, but
the shared wrapper does not degrade quality or memory.

## Overall conclusion

- The Python multi-output wrapper is real and correct on conditioning-heavy routes.
- It is most valuable where model reload and route setup are expensive relative to save time,
  especially Wan A14B image-to-video and FLUX.2 edit.
- It should not be oversold as a universal speedup. The measured Qwen masked-edit and large-image
  cases were essentially flat on wall time, even though they stayed quality-exact and memory-safe.
- SeedVR2/upscale should stay documented as a separate Python surface until MLX-Gen intentionally
  expands the runtime contract beyond unified `generate` families.
