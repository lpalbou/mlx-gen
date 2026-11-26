# SeedVR2 audio copy-through command log

Default contract: preserve source audio when it is present. Use `--drop-audio` only when you
intentionally want a silent restored MP4.

## Real-source proof

```bash
ffmpeg -y -v error \
  -ss 25 -t 10 \
  -i "air_france_commercial_2011.mp4" \
  -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart \
  validation_outputs/seedvr2_video_2026_06_21_audio/air_france_25s_10s_source_excerpt.mp4
```

```bash
ffmpeg -y -v error \
  -ss 25 -t 10 \
  -i "air_france_commercial_2011.mp4" \
  -map 0:v:0 -an \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  validation_outputs/seedvr2_video_2026_06_21_audio/air_france_25s_10s_silent_video.mp4
```

```bash
uv run python - <<'PY'
from pathlib import Path
from mflux.utils.video_util import VideoUtil

outdir = Path("validation_outputs/seedvr2_video_2026_06_21_audio")
copied = outdir / "air_france_25s_10s_audio_copied.mp4"
(outdir / "air_france_25s_10s_silent_video.mp4").replace(copied)

print(
    VideoUtil.copy_source_audio_to_video(
        source_video_path="air_france_commercial_2011.mp4",
        restored_video_path=copied,
        clip_start_seconds=25.0,
        clip_duration_seconds=10.0,
    )
)
PY
```

## Fail-closed refusal proof

```bash
ffmpeg -y -v error \
  -ss 25 -t 9 \
  -i "air_france_commercial_2011.mp4" \
  -map 0:v:0 -an \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  validation_outputs/seedvr2_video_2026_06_21_audio/air_france_25s_9s_silent_video.mp4
```

```bash
uv run python - <<'PY'
from mflux.utils.video_util import VideoUtil

print(
    VideoUtil.copy_source_audio_to_video(
        source_video_path="air_france_commercial_2011.mp4",
        restored_video_path="validation_outputs/seedvr2_video_2026_06_21_audio/air_france_25s_9s_silent_video.mp4",
        clip_start_seconds=25.0,
        clip_duration_seconds=10.0,
    )
)
PY
```
