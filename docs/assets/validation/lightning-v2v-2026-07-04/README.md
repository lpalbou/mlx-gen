# Lightning V2V Validation Matrix (2026-07-04)

Question under proof: can Wan video-to-video run with Wan2.2 Lightning LoRA adapters, and which?

Answer: yes, through the public unipc route, with the `lightx2v/Wan2.2-Lightning` T2V-A14B
4-step high/low-noise pairs. The on-grid recipe is `--steps 4 --video-strength 0.75`
(3 effective steps at timesteps 937/833/625, exactly the 4-step distillation grid,
high-noise LoRA engaged for the first step) with `--guidance 1 --guidance-2 1 --flow-shift 5
--solver unipc`.

## Matrix runs (all q8 A14B, `--low-ram --metadata`, sequential, peak RSS <= 14.9 GB)

| Run | Adapter | Clip / size | Seed | Generation | Edit result | Drift vs source* |
| --- | --- | --- | --- | ---: | --- | ---: |
| 1 (prior session) | Seko-V1.1 | conference 480x832x25f | 8602 | 157 s | man -> woman OK | 31.2 |
| 2 | Seko-V1.1 | conference 480x832x25f | 9001 | 625 s | man -> woman OK | 31.8 |
| 3 | Seko-V1.1 | ship 448x256x17f | 4242 | 145 s | ship -> smuggler ship, bright circular reactor OK | n/a (different clip) |
| 4 | Seko-V2.0 | conference 480x832x25f | 8602 | 358 s | man -> woman OK | 26.1 (tracks source closer than V1.1) |
| 5 | Seko-V1.1 + `--video-mask-path` | conference 480x832x25f | 8602 | 407 s | man -> woman OK, background locked | **preserved 1.89 / edited 30.5** |

*Mean per-pixel delta vs source in the mask-preserved region (codec re-encode floor: 1.92).
Reference points: 20-step CFG-on plain V2V = 14.9; 20-step masked V2V = 1.73.

Generation time varies with system load (157-625 s across identical configs tonight); the
consistent invariant is 3 transformer forwards vs 28 for the 20-step CFG-on baseline
(1432-1718 s), i.e. roughly 3.5x-9x wall-clock.

## What is validated, and what is not

- VALIDATED (bounded: 2 seeds, 2 clips, 2 resolutions, 2 adapter versions, masked combo):
  `Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1` and `...-Seko-V2.0` pairs with explicit
  `--lora-target-roles high_noise_transformer low_noise_transformer`.
- NOT APPLICABLE: `Wan2.2-I2V-A14B-...` Lightning adapters (public V2V runs on the T2V route)
  and TI2V-5B Lightning (V2V is A14B-only).
- PRESENT UPSTREAM BUT UNTESTED here: `Wan2.2-T2V-A14B-4steps-lora-250928` and
  `...-250928-dyno`.

## Known trade-offs (unchanged from the assessment)

- Guidance 1 disables CFG: negative prompts are inert on every Lightning run.
- Unmasked Lightning V2V re-synthesizes the scene more than the 20-step CFG-on baseline
  (drift 26-32 vs 15-17). The masked combo removes that problem for the preserved region
  entirely (1.89, at the codec floor) while keeping the speedup - it is the recommended
  fast recipe when the background must survive.
- Strength lattice at steps 4: 0.75-0.99 -> 3 effective steps (high-noise engaged);
  1.0 -> 4 steps; 0.7 -> 2 steps and silently skips the high-noise LoRA.

## Included artifact

[lightning_v2v_matrix_comparison.png](lightning_v2v_matrix_comparison.png) - frame-12 side-by-side
of the source, both unmasked Lightning adapters, and the masked + Lightning run, with banner zooms
showing the preserved region. The full run bundle (five MP4s, contact sheets, and metadata
sidecars) is preserved locally on the validation host; each run's exact settings are recorded in
its metadata sidecar and summarized in the matrix table above.

## Exact reproduce command (masked + Lightning, the recommended fast recipe)

The source clip was generated with the validated Lightning T2V recipe (exact command below); the
mask is a static white rounded rectangle over the speaker.

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit \
  --prompt "A realistic wide shot of a man giving a talk on a conference stage. He is fully visible from head to feet, standing next to a wooden podium, wearing a dark blue suit with a white shirt. Behind him a large bright blue presentation screen and a vertical conference banner. Warm stage lighting, clean stage floor, stable fixed camera, he speaks and gestures naturally with one hand." \
  --width 480 --height 832 --frames 25 --steps 4 --guidance 1 --guidance-2 1 \
  --flow-shift 5 --solver euler --fps 16 --seed 8601 \
  --lora-paths "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
               "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
  --lora-target-roles high_noise_transformer low_noise_transformer \
  --low-ram --metadata \
  --output source_clip.mp4
```

With those two inputs:

```sh
mlxgen generate \
  --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit \
  --video-path source_clip.mp4 \
  --video-mask-path person_mask.png \
  --prompt "A realistic wide shot of a woman giving a talk on a conference stage. Keep the exact same stage, wooden podium with microphones, large bright presentation screen with the same logo, conference banner, warm stage lighting, camera framing, and natural speaking gestures. She is fully visible from head to feet, has shoulder-length dark hair tied back, and wears the exact same dark blue suit with a white shirt." \
  --width 480 --height 832 --frames 25 --steps 4 --guidance 1 --guidance-2 1 --flow-shift 5 \
  --video-strength 0.75 --solver unipc --fps 16 --seed 8602 \
  --lora-paths "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
               "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
  --lora-target-roles high_noise_transformer low_noise_transformer \
  --low-ram --metadata \
  --output masked_lightning_v2v.mp4
```
