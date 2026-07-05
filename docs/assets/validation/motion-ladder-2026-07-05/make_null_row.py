"""Null row for the motion-fidelity ladder: source re-encoded through the exact writer
(VideoUtil.save_video -> VideoStreamWriter: libx264, crf 18, preset medium, yuv420p,
+faststart) that every generation run uses. Measures the codec/measurement floor with
zero GPU work.

Frames are obtained via the same VideoUtil.read_video_clip call the V2V runtime uses
(max_frames=25, target_fps=16.0); with a 16 fps source the resample is a no-op, so the
writer sees exactly the frames the model would consume.
"""

from pathlib import Path

from mflux.utils.video_util import VideoUtil

SOURCE = Path("validation_outputs/v2v_conference_480p_2026_07_04/man_conference_480p_lightning_source.mp4")
OUTPUT = Path("validation_outputs/motion_fidelity_ladder_2026_07_05/null_row_source_reencode.mp4")

clip = VideoUtil.read_video_clip(SOURCE, max_frames=25, target_fps=16.0)
assert clip.clip_frame_count == 25, clip.clip_frame_count
assert clip.sampled_fps is None, "16 fps source must not be resampled (matches runtime no-op path)"
saved = VideoUtil.save_video(frames=clip.frames, path=OUTPUT, fps=16)
print(f"null row written: {saved} ({len(clip.frames)} frames)")
