# MiniMax-H3: Video With A Generated Soundtrack

Open this file when the task needs a video clip with synchronized sound. MiniMax-H3 is the only
family here that generates audio: one prompt (optionally with a first-frame image) produces a
24 fps clip and a stereo soundtrack in a single MP4 with an AAC track, 5 to 15 seconds long. It
runs through `mlxgen generate` as `text-to-video`, or `image-to-video` when `--image-path` is given.

## Before you run it

- **License.** The weights and the packages derived from them are under the MiniMax H3 Community
  License. Its grant excludes the European Union, the United Kingdom, the Republic of Korea, and
  the United States of America, and its use restrictions follow the weights and the outputs.
  Confirm with the user that they may use it before downloading; the `LICENSE` file ships inside
  each package.
- **Access.** The package repositories are gated: the user signs in with `hf auth login` and accepts
  the license terms on the model page once before `mlxgen download`.
- **Cost.** This is the heaviest model in the package. Agree on the schedule, canvas, and frame
  count with the user before starting, and run nothing else heavy alongside it.

| Profile (124 frames) | Peak memory | Peak with `--low-ram` | Time per clip |
| --- | ---: | ---: | --- |
| 8-bit package, `minimax-h3-turbo-544p`, `960x544` | 77.0 GiB | 65.0 GiB | 11 to 17 min |
| 4-bit package, `minimax-h3-turbo-544p`, `960x544` | 57.6 GiB | 45.6 GiB | 11 to 17 min |
| 8-bit package, `minimax-h3-turbo`, `1344x768` | 82.0 GiB | not measured | about 34 min |
| 8-bit package, base schedule (50 steps), `960x544` | as the 8-bit `960x544` row | | about 62 min |

Peak memory is the whole-process footprint on an Apple M5 Max; times assume an otherwise idle
machine. Pick the package by machine memory: the 8-bit package from 96 GB, the 4-bit package with
`--low-ram` on 64 GB (close other memory-heavy applications first), neither below that. A longer
clip or larger canvas costs more, about 0.25 GiB per thousand packed rows (`960x544` at 124 frames
is 19,484 rows; `1344x768` at 124 frames and `960x544` at 243 frames are about 37,800).

## Packages and schedules

Two published packages hold the base weights already quantized. They need mlx-gen 0.39.0 or newer.

| Package | Contents | Disk |
| --- | --- | ---: |
| `AbstractFramework/minimax-h3-8bit` | 8-bit; the closest to the released weights | about 64 GiB |
| `AbstractFramework/minimax-h3-4bit` | Mixed: 4-bit conditioner, 6-bit transformer; for less memory | about 45 GiB |

Neither package contains a speed adapter. `--base-model` chooses the schedule the package runs as:

| `--base-model` | Steps | Default canvas | Use |
| --- | ---: | --- | --- |
| `minimax-h3-turbo-544p` | 8 | `960x544` | Recommended: the practical setting for iteration |
| `minimax-h3-turbo` | 8 | `1344x768` | Final renders at 768p; much slower per step |
| omitted | 50 | `1344x768` | Base schedule, no adapter |

Pass the same `--base-model` to `mlxgen download`, so the matching Turbo adapter is cached too, and
never pass `--quantize` with a package.

```sh
mlxgen capabilities --model "AbstractFramework/minimax-h3-8bit" --base-model minimax-h3-turbo-544p
mlxgen download --model "AbstractFramework/minimax-h3-8bit" --base-model minimax-h3-turbo-544p

mlxgen generate \
  --model "AbstractFramework/minimax-h3-8bit" \
  --base-model minimax-h3-turbo-544p \
  --prompt "<WHAT HAPPENS ON SCREEN, SHOT BY SHOT>" \
  --soundscape "<WHAT THE SCENE SOUNDS LIKE>" \
  --music "<THE SCORE, OR: No non-diegetic music; natural sound only.>" \
  --seed 42 --metadata \
  --output "<CLIP.mp4>"
```

Success: `<CLIP.mp4>` (124 frames at `960x544`, 24 fps, stereo AAC) and `<CLIP>.metadata.json`.
The `capabilities` call should print `"label": "MiniMax-H3 Turbo 544p"` with `default_steps` 8;
if it prints `"MiniMax-H3"` and 50 steps, `--base-model` is missing.

### Running from the source snapshot instead

The catalog entries `minimax-h3`, `minimax-h3-turbo`, and `minimax-h3-turbo-544p` address the
source repository `MiniMaxAI/MiniMax-H3` (about 140 GB on disk) with the same three schedules.
There `--quantize` accepts only `8` or `4` for this model (`4` is the same mixed policy as the
4-bit package); other values are refused. Omitting it loads the weights unquantized, which needs a
machine larger than their `unquantized_weights_bytes` (published by `capabilities`); a load that
cannot fit is refused with a message naming the remedy, instead of being killed mid-load.

```sh
mlxgen download --model minimax-h3-turbo-544p
mlxgen generate --model minimax-h3-turbo-544p --quantize 8 \
  --prompt "<PROMPT>" --soundscape "<SOUND>" --music "<MUSIC>" \
  --seed 42 --metadata --output "<CLIP.mp4>"
```

A package gives the same result as load-time quantization at the same width and skips the
quantization pass on every load, so prefer a package.

## Prompting

The model is trained on three labelled sections:

| Section | Option | Content |
| --- | --- | --- |
| `integrated_multimodal_description` | `--prompt` (or `--prompt-file`) | What happens on screen, with timing ("halfway through the clip") |
| `overall_soundscape` | `--soundscape` | Diegetic sound: room tone, weather, footsteps, impacts |
| `non_diegetic_music` | `--music` | The score, or "No non-diegetic music; natural sound only." |

- A prompt that already contains the section labels is passed through verbatim. Do not also pass
  the option for a section the prompt already carries: that is refused.
- Speech goes in the description with a speaker id and a dialogue tag:
  `The woman with a warm mid-pitched voice (S1) says: <d>[English] Good morning!</d>`. Keep one id
  per speaker and only the language tag and the exact words inside `<d>`. Dialogue is generated
  speech, not a supplied voice.
- There is no negative prompt and no guidance scale: the model is guidance-distilled.
- With a first-frame image, describe the motion that follows, as one continuous action with its
  end state, rather than re-describing the picture.

## Options

| Option | Behavior |
| --- | --- |
| `--image-path` | First-frame image (`image-to-video`). The canvas follows its aspect ratio at the schedule's short edge (544 or 768) unless `--width`/`--height` are given. |
| `--width`, `--height` | Multiples of 32, aspect ratio between 1:4 and 4:1. |
| `--frames` | Default `124`. Valid counts are `17n + 5` from `124` to `345`: 124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345 (5.17 to 14.375 s). Other values round up with a warning; above `345` is rejected. |
| `--steps` | Transformer evaluations: `8` on the Turbo schedules, `50` on the base schedule. |
| `--video-shift`, `--audio-shift` | Schedule shifts. Defaults `12` / `3` (base and 544p), `6` / `3` (768p Turbo). |
| `--no-audio` | Skip the audio decode and write a silent clip. |
| `--release-text-encoder` | Drop the text conditioner after encoding the prompt; lowers the peak during denoising, not during loading. |
| `--low-ram` | Tightens the MLX cache and releases the conditioner after encoding. |
| `--mlx-cache-limit-gb` | Caps the MLX free-buffer cache. Raising it raises the peak more than a larger canvas does. |
| `--lora-paths`, `--lora-scales` | Replace the automatic Turbo adapter with the list you pass (see below). |
| `--seed`, `--auto-seeds`, `--metadata`, `--output`, `--json-events`, `--no-replace` | As on other generate routes. |

Memory follows the packed sequence length, which grows with both canvas and frame count: a longer
clip costs more than the 124-frame figure. Before a long run the runtime estimates the peak,
refuses a request that cannot fit, and warns when other processes will decide the outcome. Read
`measured_runs`, `max_validated_frames`, `peak_bytes_fixed`, and `peak_bytes_per_packed_row` from
`mlxgen capabilities` for the published measurements; `max_frames` is only the decode limit.

## Output

- The sidecar records `steps`, `video_shift`, `audio_shift`, `frames` and `requested_frames`
  (they differ when `--frames` was rounded), `duration_seconds`, and the audio fields
  `audio_present`, `audio_source`, `audio_channels`, `audio_sample_rate`,
  `audio_duration_seconds`, `audio_muxed`, `audio_codec`, `audio_mux_mode`.
- If the soundtrack cannot be muxed (no `ffmpeg` on `PATH` and the fallback fails), the video is
  still written and the track is saved as `<output>.wav`; `audio_sidecar_path` names it. Check
  `audio_muxed` before reporting a clip with sound.

## Adapters (LoRA)

On a Turbo schedule, `--lora-paths` is the complete adapter list: a character or style adapter
passed alone drops the Turbo adapter and produces a noisy clip. Pass the Turbo file with it:

```sh
mlxgen generate --model "AbstractFramework/minimax-h3-8bit" --base-model minimax-h3-turbo-544p \
  --prompt "<PROMPT>" \
  --lora-paths "hf:lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors" \
               "<PATH/TO/STYLE_ADAPTER.safetensors>" \
  --lora-scales 1.0 1.0 --seed 42 --output "<CLIP.mp4>"
```

Adapters in the diffusers and the original-checkpoint key layouts load (PEFT or kohya naming).
"did not match any known adapter keys" means the file is not a MiniMax-H3 transformer adapter; a
DiffSynth-Studio export is refused and must be re-exported.

## Python

```python
from mflux.models.common.download_policy import DownloadRequiredError
from mlxgen import load_generation_model

MODEL, SCHEDULE = "AbstractFramework/minimax-h3-8bit", "minimax-h3-turbo-544p"
try:
    loaded = load_generation_model(model=MODEL, base_model=SCHEDULE)  # no `quantize` for a package
except DownloadRequiredError:
    raise SystemExit(f"files missing, run: mlxgen download --model {MODEL} --base-model {SCHEDULE}")

result = loaded.generate_output(
    seed=42,
    prompt="<WHAT HAPPENS ON SCREEN>",
    soundscape="<WHAT THE SCENE SOUNDS LIKE>",
    music="<THE SCORE>",
    output="<CLIP.mp4>",
    save_kwargs={"export_json_metadata": True},
)
print(result.saved_path, result.artifact.audio.sample_rate)
```

The generate keywords are `prompt`, `soundscape`, `music`, `image_path`, `width`, `height`,
`num_frames`, `num_inference_steps`, `video_shift`, `audio_shift`, `generate_audio`, and
`release_text_encoder`. The artifact carries the soundtrack as `artifact.audio` (`waveform`,
`sample_rate`).

## Limits and failures

- Reference-to-video and a closing keyframe are not available.
- Fast hand-object interaction is where the 8-step schedules are least reliable.
- "splits `<d>` ... into several tokens" when loading a package you prepared: its tokenizer lost
  the dialogue markers. Prepare the package again with the command in the message.
- Out of memory or disk: use the 4-bit package, the 544p schedule, 124 frames, close other
  GPU-heavy applications, and rerun an interrupted download (downloads resume).

Source docs: <https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/minimax-h3.md>.
