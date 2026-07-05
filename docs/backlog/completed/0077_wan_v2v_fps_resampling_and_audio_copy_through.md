# 0077 - Wan V2V fps Resampling And Audio Copy-Through

- Status: completed (2026-07-05)
- Scope: Wan `Wan2.2-T2V-A14B` video-to-video (plain and masked), shared video IO utilities
- Proof: published in-repo at `docs/assets/examples/conference-fps-audio/` (source clip, output
  MP4 with audio, sidecar, contact sheets; full command in `docs/wan-video.md`); working bundle
  preserved locally at `validation_outputs/fps_audio_proof_2026_07_05/`

## Problem

The truth-patch band (folded into 0.19.0) documented two V2V limitations honestly but left them
as user problems:

1. No temporal resampling: the first `--frames` source frames were consumed as-is and re-timed
   to `--fps`, so a 30 fps phone clip run at `--fps 16` played 1.88x slower than the source.
2. Output was video-only: any source audio track was dropped with a warning.

Both mechanisms already existed in `VideoUtil` for the SeedVR2 restore surface.

## What shipped

1. Decode-time fps resampling (default on, no new flag):
   - `VideoUtil.read_video_clip(target_fps=...)` samples frames on the output timeline via
     ffmpeg `-vf fps=` (nearest-PTS greedy fallback for PyAV), returning `sampled_fps`.
   - Frame-drift skip tolerance `|source/target - 1| < 1/(2*frames)` keeps matching-fps runs
     bit-identical with prior releases (all masked-V2V frame-exact workflows are unaffected).
   - Downsampling prints an informational note; upsampling (frame duplication) warns.
   - The V2V latent cache key includes fps; the CLI pre-load probe is duration-based
     (`frames/fps` seconds needed, one frame period slack).
   - Metadata gains `source_video_resampled`.
2. Audio copy-through (default on for V2V, best-effort):
   - `SourceAudioCopySpec` threaded through `save_video` and `save_video_batches` (both the
     regular and `--low-ram` streamed paths), executed between file health validation and the
     single sidecar write, so the sidecar records `audio_present` / `audio_copied` /
     `audio_copy_mode` / `audio_copy_reason` alongside `video_health`.
   - Best-effort by design (unlike the strict SeedVR2 restore contract, documented in README):
     a failed mux saves the silent video, prints the reason plus a manual ffmpeg remux
     command, and never discards a finished generation.
   - `GeneratedVideo.save` requests the copy for `task == "video-to-video"` with a source
     path; T2V/I2V are untouched.
3. Bug found by the proof's first run and fixed: `copy_source_audio_to_video` used
   `-shortest`, which truncated stream-copied video packets at AAC audio EOF and dropped the
   last 2 of 17 frames (post-mux validation rejected the mux; the run degraded exactly as
   designed). Removed `-shortest` (the audio input is already bounded by `-ss`/`-t`);
   regression test `test_copy_source_audio_keeps_all_video_frames_on_short_clips` pins it.
   This also fixes short-clip audio copy for the SeedVR2 path, which shares the command.

## Validation

- Two adversarial subagents: a design review (approved with corrections that were all
  adopted: `sampled_fps` default, drift-based skip tolerance, upsample warning, mux placement
  inside the save functions to cover `--low-ram`, exception-catching best-effort wrapper,
  duration-based probe slack) and a code verification (SHIP; all attack vectors passed, two
  cosmetic findings fixed: `.6g` fps formatting, quoted/corrected manual remux hint).
- Full no-weights band green: 1254 passed, lint and format clean.
- Model-backed proof: 30 fps + audio source through the documented Lightning V2V recipe;
  output verified by ffprobe (17 frames, 16 fps, 1.062 s, AAC audio 1.062 s) and sidecar
  (`source_video_resampled: true`, `audio_copied: true`). Red-necktie edit applied with
  scene, motion, and real-time speed preserved.

## Docs

`docs/wan-video.md` temporal/audio contract rewritten; `docs/faq.md` V2V limits updated;
`docs/api.md` V2V metadata paragraph updated; `README.md` feature bullet + SeedVR2 strictness
contrast; `llms-full.txt` V2V paragraph; changelog Unreleased entry (Changed + Added).
