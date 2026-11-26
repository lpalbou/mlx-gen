# Video Generation: Wan2.2, VACE, Bernini-R

Open this file for silent video work through `mlxgen generate`: text-to-video, image-to-video,
prompt-guided video-to-video, and reference-image video. Video with a generated soundtrack is in
[minimax-h3.md](minimax-h3.md); restoring an existing video is in [restoration.md](restoration.md).

Every video run is expensive. Agree on model, size, frames, and steps with the user first, and
start from a small profile.

## Models

| Model | Ids | Routes | Size multiple | Defaults |
| --- | --- | --- | ---: | --- |
| Wan2.2 TI2V-5B | `wan2.2-ti2v-5b`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, `AbstractFramework/wan2.2-ti2v-5b-diffusers-{8bit,bf16}` | text-to-video; image-to-video with one image | 32 px | 50 steps, guidance 5, 24 fps, flow shift 5.0 |
| Wan2.2 T2V-A14B | `wan2.2-t2v-a14b`, `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `AbstractFramework/wan2.2-t2v-a14b-diffusers-{8bit,bf16}` | text-to-video; video-to-video with one source video. Rejects images | 16 px | 81 frames, 40 steps, guidance 4 and 3, 16 fps, flow shift 3.0 |
| Wan2.2 I2V-A14B | `wan2.2-i2v-a14b`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers`, `AbstractFramework/wan2.2-i2v-a14b-diffusers-{8bit,bf16}` | image-to-video; requires exactly one image | 16 px | 81 frames, 40 steps, 16 fps, flow shift 3.0 |
| Wan2.1 VACE 1.3B | `wan-vace`, `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | reference-image object injection; learned masked video edit | | |
| Bernini-R 1.3B (experimental) | `bernini-r-1.3b` | reference-to-video, reference video edit, video edit | 16 px | 848x480, 81 frames, 40 steps, 16 fps, flow shift 5 |

Single shots default to 81 frames across the Wan family, and 81 is the recommended shot length;
TI2V-5B supports up to 121 (pass `--frames 121` explicitly). Longer single shots drift back toward
the first frame: chain shots instead of raising `--frames`.

Wan 8-bit packages are a disk saving only; they run at BF16-class memory.

## Size, length, and cost

- Duration is `frames / fps`; there is no duration option. `--frames 81 --fps 16` is about 5.06 s,
  `--frames 121 --fps 24` about 5.04 s.
- Frame counts must be `4n + 1`; other values are adjusted to `4 * floor(frames / 4) + 1`.
- Width and height are adjusted up to the model multiple: on TI2V-5B, `1280x720` becomes
  `1280x736` and `432x240` becomes `448x256`.
- Image-to-video keeps the source image's aspect ratio; `--width`/`--height` are a size target.
  `--canvas-policy exact-resize` forces the canvas, and `--resize-mode resize|crop|pad` maps the
  source onto it.
- The sidecar (`--metadata`) records requested, source, and resolved sizes. With `--json-events`
  the `start` event carries the resolved `width`/`height`.

| Model | Quality size | Lower-cost sizes |
| --- | --- | --- |
| TI2V-5B | `1280x704`, `704x1280` | `832x480`, `480x832` (add `--flow-shift 3`); smaller is a routing check only |
| T2V-A14B, I2V-A14B | `1280x720`, `720x1280` | `832x480`, `480x832`, `480x240`, `240x480`, `448x256`, `432x240` |

Measured reference points on an Apple M5 Max:

| Profile | Time | Peak memory |
| --- | --- | --- |
| A14B T2V, `480x240`, 101 frames, 20 fps, 20 to 25 steps | about 30 minutes | |
| TI2V-5B, `832x480`, 101 frames, 20 fps, 25 steps | about 12 minutes | |
| TI2V-5B, `1280x704`, 101 frames, 25 steps | about 35 minutes | |
| A14B T2V, `384x224`, 33 frames, 12 steps, 8 fps, `--low-ram` | about 2.5 minutes | 33.0 GiB physical |
| A14B I2V, `384x384`, 33 frames, 12 steps, 8 fps, `--low-ram` | about 4 minutes | 33.7 GiB physical |
| TI2V-5B 8-bit, `1280x704`, 17 frames, 20 steps | | 103.7 GiB physical |

For five-second prompt iteration, A14B at `480x240` gave stronger results than TI2V-5B at
`832x480` in the documented comparison. Use tiny canvases only to check that a command routes and
writes an MP4.

## Text-to-video and image-to-video

```sh
mlxgen download --model "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
mlxgen generate \
  --model "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit" \
  --prompt "<PROMPT>" \
  --width 480 --height 240 --frames 101 --fps 20 --steps 20 \
  --seed 42 --low-ram --metadata \
  --output "<VIDEO.mp4>"
```

For image-to-video use an I2V-A14B or TI2V-5B id and add `--image "<FIRST_FRAME.png>"`.

- `--guidance` is the text guidance; `--guidance-2` is the A14B low-noise stage. With both
  omitted, model defaults apply (4 and 3 on T2V-A14B); with only `--guidance` set, the low-noise
  stage follows it. `--guidance-2` is rejected on single-transformer models.
- Wan applies its official negative prompt when none is passed. Pass `--negative-prompt ""` to run
  without one (simple abstract scenes can turn into noisy texture with the default).
- Prompts are cut at 512 UMT5 tokens; a warning names the counts and the sidecar records
  `prompt_tokens` and `prompt_truncated`.
- For image-to-video, describe concrete motion and name the moving parts; put failure modes such
  as "static still image" or "sudden scene cut" in the negative prompt.
- `--low-ram` lowers the peak at some cost in speed. `--solver unipc|euler` selects the solver.
- Success: exit status 0 and an MP4 that passed the post-save health check. On failure the CLI
  writes `<output stem>.failure.json` next to the intended output (error, seed, prompt, size,
  frames, steps); `--failure-diagnostics` adds memory and tensor-health details.

## Prompt-guided video-to-video (T2V-A14B)

One source clip plus one prompt: the clip anchors composition and camera path, and the content is
regenerated under the prompt. It is not restoration and not frame-accurate preservation.

```sh
mlxgen generate \
  --model "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit" \
  --video-path "<SOURCE.mp4>" \
  --prompt "<WHAT THE CLIP SHOULD BECOME>" \
  --width 448 --height 256 --frames 17 --fps 10 --steps 5 \
  --video-strength 0.7 --solver unipc \
  --seed 42 --low-ram --metadata \
  --output "<EDITED.mp4>"
```

That is a bounded route check; for quality use the A14B defaults (`832x480` or `1280x720`, 81
frames, 40 steps).

- Only the T2V-A14B route, and only `--solver unipc`. TI2V-5B and I2V-A14B reject a source video.
- `--video-strength` is in `(0, 1]`, default `0.8`; the run denoises
  `floor(steps x video_strength)` steps. At `0.8` subject gestures are re-synthesized; `0.5` to
  `0.6` at 20 steps preserved source gesture timing in the measured runs. Below about `0.7` the
  high-noise stage is skipped, so `--guidance-2` carries the guidance.
- The source is stretched to the requested canvas (match the aspect ratio) and resampled to
  `--fps`; `--frames 17 --fps 16` consumes the first 1.06 s of the source.
- Source audio is copied onto the output best-effort; the sidecar records `audio_copied`,
  `audio_copy_mode`, and `audio_copy_reason`.
- `--video-mask-path "<MASK.png>"`: one static mask. White may change; black is locked to the
  source at every step. Draw it over the union of a moving subject's positions. An all-black mask
  is rejected.

## VACE (`wan-vace`)

- Reference-image object injection: `--reference-image "<OBJECT.png>"` (repeatable) plus a prompt
  and no source video. Segment the subject onto a plain white background; a crop with background
  gives only a style-level match.
- Learned masked edit: `--video-path` plus `--video-mask-path`. `--vace-masked-region generate`
  (default) synthesizes new structure in the white region; `repaint` keeps structure and changes
  style or color.
- `--conditioning-scale` (default `1.0`) replaces `--video-strength`, which VACE rejects.

## Bernini-R 1.3B (experimental)

Use it for diagnosis and exploration, not production output: its routes run, but they are not
promoted, and a successful exit does not mean a successful edit.

```sh
mlxgen download --model bernini-r-1.3b
mlxgen generate \
  --model bernini-r-1.3b \
  --reference-image "<SUBJECT.png>" \
  --reference-image "<GARMENT.png>" \
  --prompt "Bring the subject from image0 to life wearing the garment from image1" \
  --width 320 --height 192 --frames 17 --fps 16 --steps 20 \
  --max-condition-size 256 \
  --seed 42 --low-ram --metadata \
  --output "<REFERENCED.mp4>"
```

- One to eight `--reference-image` values, addressed in the prompt as `image0`, `image1`, ... in
  command-line order. References only: reference-to-video. One `--video` plus references:
  reference video edit. One `--video` alone: prompt-guided video edit.
- BF16 only: omit `--quantize`. It also rejects LoRA, `--video-strength`, `--video-mask-path`,
  `--guidance-2`, non-UniPC solvers, and ordinary `--image` inputs.
- With a source video, `--width`/`--height` are a source-aspect area target capped by
  `--max-condition-size` (multiple of 16 up to 1280, default 848) and `--frames` is a maximum;
  read `output_frames` and the resolved canvas from the sidecar.
- The tested low-RAM shapes peaked at 9.45 GB; keep `--low-ram`.
- `bernini-r-1.3b-bf16` (15.6 GiB download) and `bernini-r-1.3b` (26.9 GiB) pin different
  repositories with identical output: download and generate with the same alias.

## LoRA on Wan

Validated rows are the 8-bit packages. A14B has two transformers, so pass each adapter file as its
own `--lora-paths` argument and assign roles:

```sh
mlxgen download --model lightx2v/Wan2.2-Lightning --all-files
# add to an A14B text-to-video command:
#   --lora-paths "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
#                "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
#   --lora-target-roles high_noise_transformer low_noise_transformer
```

TI2V-5B uses the single role `transformer`. The fast video-to-video recipe with that pair is
`--steps 4 --video-strength 0.75 --guidance 1 --guidance-2 1 --flow-shift 5 --solver unipc`; it is
fast but does not preserve source motion. See [lora.md](lora.md).

## Repeat runs and applications

- Identical prompts are served from an on-disk prompt-embedding cache
  (`~/Library/Caches/mflux/prompt_embeds/wan`, 64 entries); `--no-prompt-cache` disables it. A new
  prompt takes 1.2 to 3.9 s to encode (UMT5 in float32 on the GPU, read one block at a time, about
  1.2 GiB while it runs). `--keep-text-encoder` keeps the encoder's structure between generations
  in one process; its weights are read for every new prompt.
- For a given prompt and seed, Wan output differs slightly from 0.40.1's (exact float32
  embeddings since 0.40.2), and cache entries written by 0.40.1 are not reused. To reproduce a
  0.40.1 clip exactly, install `mlx-gen==0.40.1`. See [Wan prompt encoding](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/wan-text-encoder.md).
- `--no-validate-health` skips the post-save re-decode (recorded as `health_check: "skipped"`).
- Several seeds in one command reuse the loaded model and write one MP4 per seed.
- Experimental A14B image-to-video continuation controls (`--last-image`, `--context-frames`,
  `--svi-anchor-image`) are described in the Wan video guide.

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/wan-video.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/bernini.md>,
<https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md>.
