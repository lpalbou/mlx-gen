# MiniMax-H3 Video With Audio

MLX-Gen runs [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) natively on Apple Silicon:
one text prompt produces a 24 fps video clip **and** a synchronized stereo soundtrack, saved as a
single MP4 with an AAC audio track. The lightx2v [Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
adapters run the same model in 8 transformer evaluations instead of 50, which is the practical
way to use it locally. Use this page for setup, prompting, sizing, runtime cost, and current
limits; use [API and CLI](api.md#minimax-h3-video-with-audio) for the option table.

## What You Get

- **Text-to-video with audio** (`text-to-video` public task): 5 to 15 seconds at 24 fps, stereo
  audio at 32 kHz, any aspect ratio between 1:4 and 4:1 on a 32-pixel grid.
- **Three catalog entries** that share the same weights and differ only in speed defaults:

| Alias | What it runs | Default canvas | Default steps | Flow shifts (video / audio) |
| --- | --- | --- | --- | --- |
| `minimax-h3` | Base model, no adapter | `1344x768` | `50` | `12` / `3` |
| `minimax-h3-turbo` | lightx2v 8-step Turbo adapter trained at 768p | `1344x768` | `8` | `6` / `3` |
| `minimax-h3-turbo-544p` | lightx2v 8-step Turbo adapter trained at 544p, mixed aspect ratios | `960x544` | `8` | `12` / `3` |

The `--steps` value counts transformer evaluations; the schedulers use one more grid point
internally. All three entries resolve to the `MiniMaxAI/MiniMax-H3` repository and the Turbo
entries add their adapter automatically.

## Requirements

MiniMax-H3 is large: a 27B-parameter transformer, a 32B-parameter Qwen3-VL conditioner (of which
MLX-Gen loads the 50 layers the model conditions on), a 2.4B-parameter video VAE and an audio VAE.

| Resource | Requirement |
| --- | --- |
| Disk | About 140 GB for the model snapshot (`transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, tokenizer and configs) plus 1.4 GB per Turbo adapter. |
| Memory | Run with `--quantize 8`. The measured `960x544`, 124-frame Turbo run peaks at 80 GB of MLX memory and a 103 GB process footprint on an Apple M5 Max, so plan on a 128 GB Mac. |
| Download | `mlxgen download --model minimax-h3-turbo-544p` fetches the snapshot subset MLX-Gen needs and the matching Turbo adapter. Generation never downloads. |

`--quantize 8` quantizes the transformer and conditioner at load time, one shard at a time, so
the load never holds a full BF16 copy next to the q8 copy. The first load reads 133 GB of shards
from disk (about 4.5 minutes on the reference machine); repeated loads from the OS page cache
take about 16 seconds. A prepared q8 package (`mlxgen prepare`) needs another 62 GB of disk.

## Quick Start

```sh
mlxgen download --model minimax-h3-turbo-544p

mlxgen generate \
  --model minimax-h3-turbo-544p \
  --prompt "[Shot 1] Cinematic medium shot, static camera, shallow depth of field. A red fox trots through fresh powder snow in a quiet birch forest at dawn; its breath steams in the cold blue light. Halfway through the clip it stops, ears twitching toward the camera, then bounds forward, kicking up a spray of snow that catches the low sun." \
  --soundscape "Soft rhythmic crunch of paws in dry snow, a faint steady wind through bare branches, and two distant crow caws near the end." \
  --music "Sparse, gentle piano notes with long reverb, slow tempo, contemplative and cold." \
  --seed 42 \
  --quantize 8 \
  --output fox.mp4 \
  --metadata
```

The command writes `fox.mp4` (124 frames, `960x544`, 24 fps, stereo AAC) and `fox.metadata.json`.
Switch to `--model minimax-h3-turbo` for the 768p adapter and canvas, or to `--model minimax-h3`
with `--steps 50` for the base schedule.

## Prompting

MiniMax-H3 is trained on a structured prompt with three labelled sections separated by blank
lines:

```text
integrated_multimodal_description: <what happens on screen, shot by shot>

overall_soundscape: <diegetic sound: what the scene itself sounds like>

non_diegetic_music: <score, or "No non-diegetic music; natural sound only.">
```

`--prompt` fills the first section, `--soundscape` the second, and `--music` the third. A prompt
that already contains the section labels (for example one written with MiniMax's
[prompting guides](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/docs)) is passed through
verbatim, so `--prompt-file` works for complete structured prompts. The prompt is tokenized as-is,
without a chat template or special tokens; it may be long, and detailed shot descriptions help.

Describe motion with timing ("halfway through the clip"), name the sounds you expect on impact
events, and state when a shot has no score. The model is guidance-distilled: there is no negative
prompt and no guidance scale.

## Sizing And Duration

- Frames follow the video VAE's `17n + 5` rule; `--frames` rounds up to the next valid count.
  `124` frames is 5.17 s, `141` is 5.88 s, `362` is the 15-second maximum.
- Width and height must be multiples of 32. The default 16:9 canvas is `1344x768`; the 544p
  adapter defaults to `960x544`. Portrait and square canvases follow the same 768-pixel short edge
  (`768x1344`, `768x768`) when you omit the size.
- Audio length follows the video: 40 audio latents per second, trimmed to the clip duration.

## Measured Runtime

Apple M5 Max, 128 GB, `--quantize 8`, 124 frames, MLX 0.31:

| Route | Canvas | Steps | `generate_video` wall time (denoise + both decodes) | Peak MLX memory |
| --- | --- | --- | --- | --- |
| `minimax-h3-turbo-544p` | `960x544` | 8 | 12.5 to 16 min over four clips (750, 765, 932, 966 s; later clips in one process ran slower as the machine warmed up) | 80 GB |
| `minimax-h3` (base) | `960x544` | 50 | 78 min (4691 s) | 78 GB |
| `minimax-h3-turbo` | `1344x768` | 8 | 36 min (2144 s; about 4.5 min per transformer evaluation) | 84 GB |

Loading is separate: the first load after a download reads 133 GB of shards (about 4.5 min), and
a warm page cache brings that down to about 16 s. The 768p step is dominated by attention over a
37,800-row packed sequence; the 544p canvas is the practical iteration setting on Apple Silicon
today, and the 768p canvas is the quality setting for a final render.

For scale, the same fox description through Wan2.2 TI2V-5B (q8, `832x480`, 121 frames at 24 fps,
50 steps, silent) took 23 min on the same machine at a 25 GB MLX peak; its sheet is included below.

## Output Contract

- The MP4 carries the video stream plus one stereo AAC track (32 kHz, 192 kb/s). Pass
  `--no-audio` to skip the audio decode and write a silent clip.
- If the audio cannot be muxed (no `ffmpeg` on `PATH` and the PyAV fallback fails), the video is
  still written and the track is saved next to it as `<output>.wav`; the metadata records why.
- Metadata records `video_shift`, `audio_shift`, `num_inference_steps`, `text_tokens`,
  `duration_seconds`, and the `audio_*` fields (`audio_present`, `audio_source: "generated"`,
  `audio_channels`, `audio_sample_rate`, `audio_duration_seconds`, `audio_muxed`, `audio_codec`,
  `audio_mux_mode`).
- `mlxgen capabilities --model minimax-h3-turbo` lists the single `minimax-h3.text-video` row with
  `supports_frames`, `supports_lora`, `dimension_multiple: 32`, and `supports_negative_prompt: false`.

## Python

```python
from mflux.models.common.config import ModelConfig
from mflux.models.minimax_h3.variants import MiniMaxH3

model = MiniMaxH3(model_config=ModelConfig.minimax_h3_turbo_544p(), quantize=8)
video = model.generate_video(
    seed=42,
    prompt="A red fox trots through fresh snow in a birch forest at dawn.",
    soundscape="Soft crunch of paws in dry snow, faint wind, two distant crow caws.",
    music="Sparse piano, slow tempo.",
)
video.save("fox.mp4", export_json_metadata=True)
waveform, rate = video.audio.waveform, video.audio.sample_rate  # (2, samples) float32, 32000
```

`generate_video` also accepts `width`, `height`, `num_frames`, `num_inference_steps`,
`video_shift`, `audio_shift`, `generate_audio`, and `progress_callback` (phases `start`,
`denoising`, `decode`, `complete`). The unified runtime resolves `--model minimax-h3*` to this
class through `load_generation_model(...)`.

## Adapters

The Turbo entries load their lightx2v adapter from `lightx2v/Minimax-h3-Turbo` with the effective
scale the adapter was trained with (alpha 8 at rank 128). Pass your own `--lora-paths` to replace
the automatic adapter; PEFT-format adapters that target the diffusers transformer module names
(`transformer_blocks.N.attn.to_q`, `ff.net.0.proj`, `ff.net.2`, and the token refiner) load
directly. The 4-step 768p Turbo adapter (`minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors`)
works with `--steps 4 --video-shift 6`.

## Current Limits

- **First-frame conditioning (FL2VA) and reference-to-video (Ref2VA) are not available yet.**
  `--image-path` is rejected; the keyframe tokens need the Qwen3-VL vision tower, which is tracked in
  [backlog 0118](backlog/planned/0118_minimax_h3_first_frame_conditioning_vision_tower.md).
- No published MLX-Gen q8 package yet; `--quantize 8` quantizes at load time.
- The 768p canvas is slow on Apple Silicon (about 5 minutes per step); use the 544p adapter to
  iterate.
- MiniMax-H3 is released under the MiniMax H3 Community License, which restricts use in some
  territories. Read the license on the model card before you download or distribute weights or
  outputs.

## Verification

Every component is a direct port of the diffusers 0.40 reference implementation and was checked
against it with real weights: the packed layout and per-row timestep plan are bit-exact, the
rectified-flow schedules match `torch.linspace` bit-for-bit on 600 grids, the transformer
(one real block), the Qwen3-VL conditioner, the video VAE encoder/decoder and the audio VAE
encoder/decoder all match at fp32 rounding noise, and the q8 transformer block stays within
1.3e-2 relative RMS of the fp32 reference. The included contact sheets below are the model-backed
proof for the shipped routes.

## Contact Sheets

Each sheet shows eight evenly spaced frames of one clip with the generated track's waveform and
spectrogram on the right; the MP4 next to each sheet in `docs/assets/minimax-h3/` is the playable
proof, and the `prompt_*.txt` files hold the complete structured prompts. All clips:
`minimax-h3-turbo-544p`, `960x544`, 124 frames, 8 steps, `--quantize 8`, Apple M5 Max.

**Fox, seed 42** ([clip](assets/minimax-h3/fox_turbo544_q8_seed42.mp4),
[prompt](assets/minimax-h3/prompt_fox.txt)): the fox walks through the birch forest and turns
its head toward the camera in the second half as the prompt asks; the quiet track carries paw
crunches under a steady wind (about -41 dBFS RMS), stereo correlation 0.91.

![Fox seed 42](assets/minimax-h3/sheet_fox_turbo544_q8_seed42.jpg)

**Fox, seed 43** ([clip](assets/minimax-h3/fox_turbo544_q8_seed43.mp4)): a different fox and
framing from the same prompt, with a louder track (-21 dBFS RMS) and a burst near the end where the
prompt places the snow spray and crow caws.

![Fox seed 43](assets/minimax-h3/sheet_fox_turbo544_q8_seed43.jpg)

**Turbo versus base, seed 42** ([base clip](assets/minimax-h3/fox_base544_q8_seed42.mp4)): the same
prompt and seed through the 8-step 544p adapter (top, 12.5 min) and the 50-step base schedule at the
same canvas (bottom, 78 min). On this seed the adapter kept the medium shot, the birch forest and the
head turn toward the camera; the base schedule settled on a wider open-snowfield framing with a
smaller fox and a quieter track. The base entry remains the reference schedule for the 768p canvas it
was released with; for 544p iteration the adapter is both the faster and, here, the more faithful
choice.

![Turbo versus base](assets/minimax-h3/sheet_fox_turbo_vs_base_544.jpg)

**Ocean waves, seed 42** ([clip](assets/minimax-h3/ocean_turbo544_q8_seed42.mp4),
[prompt](assets/minimax-h3/prompt_ocean.txt)): waves break over basalt rocks with a large spray
mid-clip; the broadband ocean track builds into a crash at the same point.

![Ocean seed 42](assets/minimax-h3/sheet_ocean_turbo544_q8_seed42.jpg)

**Fox at 768p, seed 42** ([clip](assets/minimax-h3/fox_turbo768_q8_seed42.mp4)): the same prompt
through `minimax-h3-turbo` on its native `1344x768` canvas (8 steps, 36 min). Denser fur and snow
detail than the 544p clips, the same walk-then-look-at-camera staging, and a quiet stereo track with
the paw crunches tracking the motion.

![Fox 768p seed 42](assets/minimax-h3/sheet_fox_turbo768_q8_seed42.jpg)

**Wan2.2 TI2V-5B on the fox description** ([clip](assets/minimax-h3/fox_wan_ti2v5b_q8_seed42.mp4)):
the visual part of the same prompt through the silent Wan route (`832x480`, 121 frames, 50 steps,
23 min). Included as a reference point for pacing and style, not as a like-for-like quality
comparison: the two models have different canvases, schedules, and training data, and only MiniMax-H3
generates the soundtrack.

![Wan TI2V-5B fox](assets/minimax-h3/sheet_fox_wan_ti2v5b_q8_seed42.jpg)

**Street guitarist, seed 42** ([clip](assets/minimax-h3/guitar_turbo544_q8_seed42.mp4),
[prompt](assets/minimax-h3/prompt_guitar.txt)): a slow dolly-in on a fingerpicking musician; the
spectrogram shows the plucked-string harmonics and rhythm of the on-camera guitar (-14 dBFS RMS,
stereo correlation 0.98).

![Guitar seed 42](assets/minimax-h3/sheet_guitar_turbo544_q8_seed42.jpg)
