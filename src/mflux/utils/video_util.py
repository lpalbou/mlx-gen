import json
import logging
import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Iterator

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.image_util import ImageUtil
from mflux.utils.tensor_health import TensorHealth
from mflux.utils.video_health import VideoHealth

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodedVideoClip:
    frames: list[PIL.Image.Image]
    fps: float
    source_width: int
    source_height: int
    source_frame_count: int | None
    source_duration_seconds: float | None
    audio_present: bool
    clip_start_frame: int
    clip_frame_count: int
    # Set only when frames were resampled onto a different output timeline; None means the
    # frames are raw source frames at `fps`.
    sampled_fps: float | None = None


@dataclass(frozen=True)
class SourceVideoInfo:
    fps: float
    source_width: int
    source_height: int
    source_frame_count: int | None
    source_duration_seconds: float | None
    audio_present: bool


@dataclass(frozen=True)
class AudioCopyResult:
    audio_present: bool
    audio_copied: bool
    copy_mode: str | None
    reason: str | None


@dataclass(frozen=True)
class SourceAudioCopySpec:
    source_video_path: str | Path
    clip_start_seconds: float
    clip_duration_seconds: float


class VideoStreamWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        fps: int | float,
        width: int,
        height: int,
        overwrite: bool = True,
    ):
        self.file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self._should_replace = False
        self._process = None
        self.container = None
        self.stream = None

        with NamedTemporaryFile(
            suffix=self.file_path.suffix or ".mp4",
            prefix=f".{self.file_path.stem}-",
            dir=self.file_path.parent,
            delete=False,
        ) as temp_file:
            self.temp_path = Path(temp_file.name)

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None:
            # Color pipeline (measured 2026-07-22, ffmpeg 8.1): we pipe sRGB-family
            # RGB24 and ffmpeg's auto-inserted swscale converts to yuv420p with the
            # BT.601 (SMPTE 170M) matrix at EVERY resolution - but used to leave the
            # stream untagged, so players (AVFoundation) assumed BT.709 for >=720p
            # and shifted colors on HD clips. Fix is metadata-only and seed-stable:
            # pin the conversion matrix to what ffmpeg already applies
            # (scale=out_color_matrix=bt601 - YUV bytes verified bitwise identical)
            # and tag the truth: matrix=smpte170m for the coding matrix, with
            # bt709 primaries/transfer since sRGB-origin RGB shares Rec.709
            # primaries/white point (tagging smpte170m primaries would falsely
            # claim SMPTE-C phosphors). setparams is used because the plain
            # -color_primaries/-color_trc output options do not survive into the
            # x264 VUI in this pipeline (verified with ffprobe).
            # Switching HD output to a real BT.709 encode would change pixel data
            # (non-bitwise) and needs a visual A/B first - tracked next to 0089.
            command = [
                ffmpeg_path,
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(float(fps)),
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                "scale=out_color_matrix=bt601,setparams=color_primaries=bt709:color_trc=bt709:colorspace=smpte170m",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-movflags",
                "+faststart",
                str(self.temp_path),
            ]
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            import av

            # The pyav fallback also converts RGB->YUV with the 601 default but
            # cannot reliably set VUI color tags across pyav versions; tagging
            # that path is a 0089-adjacent follow-up.
            self.container = av.open(str(self.temp_path), mode="w", options={"movflags": "+faststart"})
            self.stream = self.container.add_stream("libx264", rate=VideoUtil._fps_to_rate(fps))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "18", "preset": "medium"}
        self._closed = False

    def write_frames(self, frames: list[PIL.Image.Image]) -> None:
        for frame in frames:
            rgb = frame.convert("RGB")
            if rgb.size != (self.width, self.height):
                rgb = rgb.resize((self.width, self.height), PIL.Image.Resampling.LANCZOS)
            self._write_rgb_array(VideoUtil._pil_rgb_to_array(rgb))

    def write_frame_arrays(self, frames: np.ndarray) -> None:
        for frame in frames:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                rgb = PIL.Image.fromarray(frame, mode="RGB").resize(
                    (self.width, self.height), PIL.Image.Resampling.LANCZOS
                )
                frame = np.array(rgb, dtype=np.uint8)
            self._write_rgb_array(frame)

    def close(self) -> Path:
        if self._closed:
            return self.file_path
        try:
            if self._process is not None:
                process = self._process
                self._process = None
                assert process.stdin is not None
                process.stdin.close()
                stderr = process.stderr.read() if process.stderr is not None else b""
                if process.stderr is not None:
                    process.stderr.close()
                return_code = process.wait()
                if return_code != 0:
                    message = stderr.decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"ffmpeg video encode failed with code {return_code}: {message}")
            else:
                assert self.stream is not None and self.container is not None
                for packet in self.stream.encode():
                    self.container.mux(packet)
                self.container.close()
            self._should_replace = True
            self._closed = True
        finally:
            if self._should_replace:
                self.temp_path.replace(self.file_path)
            elif self.temp_path.exists():
                self.temp_path.unlink()
        return self.file_path

    def abort(self) -> None:
        if self._closed:
            return
        if self._process is not None:
            process = self._process
            self._process = None
            with suppress(OSError, RuntimeError, ValueError):
                if process.stdin is not None:
                    process.stdin.close()
                if process.stderr is not None:
                    process.stderr.close()
                process.kill()
                process.wait()
        elif self.container is not None:
            with suppress(OSError, RuntimeError, ValueError):
                self.container.close()
        self._closed = True
        if self.temp_path.exists():
            self.temp_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def _write_rgb_array(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.write(frame.tobytes())
            return

        import av

        assert self.stream is not None and self.container is not None
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)


class VideoUtil:
    @staticmethod
    def copy_source_audio_to_video(
        *,
        source_video_path: str | Path,
        restored_video_path: str | Path,
        clip_start_seconds: float,
        clip_duration_seconds: float,
    ) -> AudioCopyResult:
        if clip_start_seconds < 0:
            raise ValueError("clip_start_seconds must be greater than or equal to zero.")
        if clip_duration_seconds <= 0:
            raise ValueError("clip_duration_seconds must be greater than zero.")

        source_info = VideoUtil.inspect_video(source_video_path)
        if not source_info.audio_present:
            return AudioCopyResult(
                audio_present=False,
                audio_copied=False,
                copy_mode=None,
                reason="no_source_audio",
            )

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="ffmpeg_not_found",
            )

        restored_path = Path(restored_video_path)
        restored_info = VideoUtil.inspect_video(restored_path)
        alignment_tolerance = VideoUtil._media_alignment_tolerance_seconds(restored_info.fps)
        if restored_info.source_duration_seconds is None:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="restored_duration_unknown",
            )
        if abs(restored_info.source_duration_seconds - clip_duration_seconds) > alignment_tolerance:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="restored_duration_mismatch",
            )
        if (
            source_info.source_duration_seconds is not None
            and clip_start_seconds + clip_duration_seconds > source_info.source_duration_seconds + alignment_tolerance
        ):
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="source_clip_out_of_range",
            )

        with NamedTemporaryFile(
            suffix=restored_path.suffix or ".mp4",
            prefix=f".{restored_path.stem}-audio-",
            dir=restored_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        command = VideoUtil._build_audio_copy_command(
            ffmpeg_path=ffmpeg_path,
            restored_video_path=restored_path,
            source_video_path=Path(source_video_path),
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            output_path=temp_path,
        )

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            with suppress(FileNotFoundError):
                temp_path.unlink()
            stderr = (error.stderr or "").strip()
            if stderr:
                log.warning("Audio copy-through failed for %s: %s", restored_path, stderr.splitlines()[-1])
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="ffmpeg_mux_failed",
            )

        validation_error = VideoUtil._validate_copied_audio_output(
            output_path=temp_path,
            expected_video=restored_info,
            expected_audio_duration_seconds=clip_duration_seconds,
        )
        if validation_error is not None:
            with suppress(FileNotFoundError):
                temp_path.unlink()
            log.warning("Audio copy-through rejected for %s: %s", restored_path, validation_error)
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason=validation_error,
            )

        temp_path.replace(restored_path)
        return AudioCopyResult(
            audio_present=True,
            audio_copied=True,
            copy_mode="ffmpeg_copy_video_aac_audio",
            reason=None,
        )

    @staticmethod
    def to_video(
        decoded_latents: mx.array,
        fps: int | float,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        quantization: int,
        generation_time: float,
        flow_shift: float | None = None,
        solver: str | None = None,
        guidance_2: float | None = None,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        negative_prompt: str | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        requested_width: int | None = None,
        requested_height: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
        materialize_frames: bool = True,
    ):
        from mflux.utils.generated_video import GeneratedVideo

        if materialize_frames:
            frames = VideoUtil._latents_to_frames(decoded_latents)
            height = frames[0].height
            width = frames[0].width
            frame_batches_factory = None
            frame_count = len(frames)
        else:
            frames = None

            def frame_batches_factory() -> Iterable[list[PIL.Image.Image]]:
                return VideoUtil._latents_to_frame_batches(decoded_latents)

            frame_count = int(decoded_latents.shape[2])
            height = int(decoded_latents.shape[3])
            width = int(decoded_latents.shape[4])
        return GeneratedVideo(
            frames=frames,
            fps=fps,
            model_config=model_config,
            seed=seed,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            flow_shift=flow_shift,
            solver=solver,
            guidance_2=guidance_2,
            precision=ModelConfig.precision,
            quantization=quantization,
            generation_time=generation_time,
            height=height,
            width=width,
            task=task,
            image_path=image_path,
            video_path=video_path,
            negative_prompt=negative_prompt,
            source_width=source_width,
            source_height=source_height,
            requested_width=requested_width,
            requested_height=requested_height,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            extra_metadata=extra_metadata,
            frame_batches_factory=frame_batches_factory,
            frame_count=frame_count,
        )

    @staticmethod
    def to_video_from_frame_batches(
        frame_batches_factory: Callable[[], Iterable[list[PIL.Image.Image]]],
        fps: int | float,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        quantization: int,
        generation_time: float,
        height: int,
        width: int,
        frame_count: int,
        flow_shift: float | None = None,
        solver: str | None = None,
        guidance_2: float | None = None,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        negative_prompt: str | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        requested_width: int | None = None,
        requested_height: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
    ):
        from mflux.utils.generated_video import GeneratedVideo

        return GeneratedVideo(
            frames=None,
            fps=fps,
            model_config=model_config,
            seed=seed,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            flow_shift=flow_shift,
            solver=solver,
            guidance_2=guidance_2,
            precision=ModelConfig.precision,
            quantization=quantization,
            generation_time=generation_time,
            height=height,
            width=width,
            task=task,
            image_path=image_path,
            video_path=video_path,
            negative_prompt=negative_prompt,
            source_width=source_width,
            source_height=source_height,
            requested_width=requested_width,
            requested_height=requested_height,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            extra_metadata=extra_metadata,
            frame_batches_factory=frame_batches_factory,
            frame_count=frame_count,
        )

    @staticmethod
    def save_video(
        frames: list[PIL.Image.Image],
        path: str | Path,
        fps: int | float,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
        source_audio_copy: "SourceAudioCopySpec | None" = None,
    ) -> Path:
        if not frames:
            raise ValueError("Cannot save a video without frames.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frames[0].size
        frame_health = None
        file_health = None
        if validate_health:
            frame_health = VideoHealth.validate_frames(
                frames,
                fps=fps,
                expected_width=width,
                expected_height=height,
                strict_visual=True,
            )
        VideoUtil._save_video_with_pyav(frames=frames, file_path=file_path, fps=fps, width=width, height=height)
        if validate_health:
            file_health = VideoHealth.validate_file(
                file_path,
                expected_width=width,
                expected_height=height,
                expected_frames=len(frames),
                expected_fps=fps,
                strict_visual=True,
            )

        if metadata is not None:
            metadata = dict(metadata)
            if validate_health:
                metadata["video_health"] = {
                    "frames": frame_health.to_metadata(),
                    "file": file_health.to_metadata(),
                }
        audio_fields = VideoUtil._apply_source_audio_copy(spec=source_audio_copy, file_path=file_path)
        if metadata is not None and audio_fields:
            metadata.update(audio_fields)

        VideoUtil._save_metadata(
            file_path=file_path,
            metadata=metadata,
            export_json_metadata=export_json_metadata,
        )

        log.info(f"Video saved successfully at: {file_path}")
        return file_path

    @staticmethod
    def save_video_batches(
        frame_batches: Iterable[list[PIL.Image.Image]],
        path: str | Path,
        fps: int | float,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
        source_audio_copy: "SourceAudioCopySpec | None" = None,
    ) -> Path:
        batch_iterator = iter(frame_batches)
        first_batch = next(batch_iterator, None)
        if first_batch is None or not first_batch:
            raise ValueError("Cannot save a video without frames.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        first_frame = first_batch[0]
        width, height = first_frame.size
        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        VideoUtil._save_video_batches_with_pyav(
            frame_batches=chain([first_batch], batch_iterator),
            file_path=file_path,
            fps=fps,
            width=width,
            height=height,
        )

        if metadata is not None:
            metadata = dict(metadata)
        if validate_health:
            file_health = VideoHealth.validate_file(
                file_path,
                expected_width=width,
                expected_height=height,
                expected_frames=VideoUtil._metadata_frame_count(metadata),
                expected_fps=fps,
                strict_visual=True,
            )
            if metadata is not None:
                metadata["video_health"] = {
                    "file": file_health.to_metadata(),
                }
        audio_fields = VideoUtil._apply_source_audio_copy(spec=source_audio_copy, file_path=file_path)
        if metadata is not None and audio_fields:
            metadata.update(audio_fields)

        VideoUtil._save_metadata(
            file_path=file_path,
            metadata=metadata,
            export_json_metadata=export_json_metadata,
        )

        log.info(f"Video saved successfully at: {file_path}")
        return file_path

    @staticmethod
    def _apply_source_audio_copy(
        *,
        spec: "SourceAudioCopySpec | None",
        file_path: Path,
    ) -> dict:
        if spec is None:
            return {}
        # Best-effort by design: a failed audio mux degrades to the previous video-only contract
        # with a printed warning and a recorded reason; it must never fail a finished generation.
        try:
            result = VideoUtil.copy_source_audio_to_video(
                source_video_path=spec.source_video_path,
                restored_video_path=file_path,
                clip_start_seconds=spec.clip_start_seconds,
                clip_duration_seconds=spec.clip_duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            result = AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason=f"{exc.__class__.__name__}: {exc}",
            )
        if result.audio_present and not result.audio_copied:
            remux_target = file_path.with_name(f"{file_path.stem}_with_audio.mp4")
            print(
                f"⚠️  Source audio could not be preserved ({result.reason}); the saved video is silent. "
                f'To remux manually: ffmpeg -i "{file_path}" -ss {spec.clip_start_seconds:.3f} '
                f'-t {spec.clip_duration_seconds:.3f} -i "{spec.source_video_path}" '
                f'-map 0:v -map 1:a -c:v copy -c:a aac "{remux_target}"'
            )
        return {
            "audio_present": result.audio_present,
            "audio_copied": result.audio_copied,
            "audio_copy_mode": result.copy_mode,
            "audio_copy_reason": result.reason,
        }

    @staticmethod
    def extract_frame(path: str | Path, index: int = 0) -> PIL.Image.Image:
        if index < 0:
            raise ValueError("Frame index must be greater than or equal to zero.")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None:
            source_info = VideoUtil.inspect_video(path)
            command = [
                ffmpeg_path,
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"select=eq(n\\,{index})",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Could not read frame {index} from {path}: {stderr or result.returncode}")
            frame_size = source_info.source_width * source_info.source_height * 3
            if len(result.stdout) != frame_size:
                raise RuntimeError(f"Could not read frame {index} from {path}")
            frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
                source_info.source_height,
                source_info.source_width,
                3,
            )
            return PIL.Image.fromarray(frame.copy())

        import av

        with av.open(str(path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")
            video_stream = container.streams.video[0]
            for frame_number, frame in enumerate(container.decode(video_stream)):
                if frame_number == index:
                    return PIL.Image.fromarray(frame.to_ndarray(format="rgb24"))
        raise RuntimeError(f"Could not read frame {index} from {path}")

    @staticmethod
    def read_video_clip(
        path: str | Path,
        *,
        start_seconds: float = 0.0,
        max_frames: int | None = None,
        target_fps: float | None = None,
    ) -> DecodedVideoClip:
        if start_seconds < 0:
            raise ValueError("start_seconds must be greater than or equal to zero.")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be greater than zero when provided.")
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be greater than zero when provided.")
        if target_fps is not None and start_seconds > 0:
            raise ValueError("target_fps resampling currently requires start_seconds == 0.")

        file_path = Path(path)
        source_info = VideoUtil.inspect_video(file_path)
        fps = source_info.fps
        source_width = source_info.source_width
        source_height = source_info.source_height
        source_frame_count = source_info.source_frame_count
        source_duration_seconds = source_info.source_duration_seconds
        audio_present = source_info.audio_present

        resample_fps = (
            target_fps
            if VideoUtil._should_resample_fps(
                source_fps=fps,
                target_fps=target_fps,
                max_frames=max_frames,
            )
            else None
        )
        frame_arrays, clip_start_frame = VideoUtil._read_video_clip_frame_arrays(
            file_path=file_path,
            start_seconds=start_seconds,
            max_frames=max_frames,
            fps=fps,
            width=source_width,
            height=source_height,
            target_fps=resample_fps,
        )
        frames = [PIL.Image.fromarray(frame) for frame in frame_arrays]

        if not frames:
            raise RuntimeError(f"Could not decode any video frames from {path}")

        return DecodedVideoClip(
            frames=frames,
            fps=fps,
            source_width=source_width,
            source_height=source_height,
            source_frame_count=source_frame_count,
            source_duration_seconds=source_duration_seconds,
            audio_present=audio_present,
            clip_start_frame=clip_start_frame,
            clip_frame_count=len(frames),
            sampled_fps=resample_fps,
        )

    @staticmethod
    def _should_resample_fps(
        *,
        source_fps: float,
        target_fps: float | None,
        max_frames: int | None,
    ) -> bool:
        if target_fps is None or source_fps <= 0:
            return False
        # Skip only when re-timing cannot move any sampled frame over the whole clip; this keeps
        # 29.97-vs-30 style differences bit-identical while catching every real speed change.
        drift = abs(source_fps / target_fps - 1.0)
        frame_budget = max_frames if max_frames is not None else 1000
        return drift >= 1.0 / (2.0 * frame_budget)

    @staticmethod
    def _read_video_clip_frame_arrays(
        *,
        file_path: Path,
        start_seconds: float,
        max_frames: int | None,
        fps: float,
        width: int,
        height: int,
        target_fps: float | None = None,
    ) -> tuple[list[np.ndarray], int]:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return VideoUtil._read_video_clip_frame_arrays_pyav(
                file_path=file_path,
                start_seconds=start_seconds,
                max_frames=max_frames,
                fps=fps,
                target_fps=target_fps,
            )

        command = [ffmpeg_path, "-v", "error"]
        if start_seconds > 0:
            command.extend(["-ss", f"{start_seconds:.9f}"])
        command.extend(["-i", str(file_path)])
        if target_fps is not None:
            # Sample frames on the output timeline so playback keeps real-time speed.
            command.extend(["-vf", f"fps={target_fps:.6g}"])
        if max_frames is not None:
            command.extend(["-frames:v", str(max_frames)])
        command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
        result = subprocess.run(command, check=False, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not decode video frames from {file_path}: {stderr or result.returncode}")
        frame_size = width * height * 3
        if frame_size <= 0 or len(result.stdout) % frame_size != 0:
            raise RuntimeError(f"Decoded video frame payload from {file_path} has an unexpected size.")
        video_np = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
        clip_start_frame = int(round(start_seconds * fps))
        return [frame.copy() for frame in video_np], clip_start_frame

    @staticmethod
    def _read_video_clip_frame_arrays_pyav(
        *,
        file_path: Path,
        start_seconds: float,
        max_frames: int | None,
        fps: float,
        target_fps: float | None = None,
    ) -> tuple[list[np.ndarray], int]:
        import av

        frames: list[np.ndarray] = []
        clip_start_frame: int | None = None
        # Nearest-PTS greedy resampling; matches ffmpeg's fps filter on CFR sources and is an
        # approximation on VFR sources (acceptable for the no-ffmpeg fallback path).
        source_period = 1.0 / fps if fps > 0 else 0.0
        next_target_time = 0.0
        with av.open(str(file_path)) as container:
            video_stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(video_stream)):
                frame_time = frame.time
                if frame_time is None:
                    frame_time = frame_index / fps
                if frame_time + 1e-9 < start_seconds:
                    continue
                if clip_start_frame is None:
                    clip_start_frame = frame_index
                if target_fps is None:
                    frames.append(frame.to_ndarray(format="rgb24").copy())
                else:
                    frame_array = None
                    while next_target_time < frame_time + source_period / 2.0 and (
                        max_frames is None or len(frames) < max_frames
                    ):
                        if frame_array is None:
                            frame_array = frame.to_ndarray(format="rgb24").copy()
                        frames.append(frame_array)
                        next_target_time += 1.0 / target_fps
                if max_frames is not None and len(frames) >= max_frames:
                    break
        return frames, clip_start_frame or 0

    @staticmethod
    def inspect_video(path: str | Path) -> SourceVideoInfo:
        ffprobe_info = VideoUtil._inspect_video_ffprobe(Path(path))
        if ffprobe_info is not None:
            return ffprobe_info

        import av

        file_path = Path(path)
        with av.open(str(file_path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")

            video_stream = container.streams.video[0]
            fps = VideoUtil._rate_to_float(video_stream.average_rate or video_stream.base_rate)
            if fps is None or fps <= 0:
                raise RuntimeError(f"Could not determine a valid video fps for {path}")

            source_width = int(video_stream.width)
            source_height = int(video_stream.height)
            source_frame_count = int(video_stream.frames) if video_stream.frames else None
            source_duration_seconds = VideoUtil._duration_seconds(
                container=container,
                video_stream=video_stream,
                fps=fps,
                source_frame_count=source_frame_count,
            )
            audio_present = len(container.streams.audio) > 0

        return SourceVideoInfo(
            fps=fps,
            source_width=source_width,
            source_height=source_height,
            source_frame_count=source_frame_count,
            source_duration_seconds=source_duration_seconds,
            audio_present=audio_present,
        )

    @staticmethod
    def _inspect_video_ffprobe(path: Path) -> SourceVideoInfo | None:
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            return None
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
            streams = payload.get("streams", [])
            video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
            audio_present = any(stream.get("codec_type") == "audio" for stream in streams)
            fps = VideoUtil._rate_to_float(VideoUtil._fraction_from_string(video_stream.get("avg_frame_rate")))
            if fps is None or fps <= 0:
                fps = VideoUtil._rate_to_float(VideoUtil._fraction_from_string(video_stream.get("r_frame_rate")))
            if fps is None or fps <= 0:
                return None
            source_frame_count = VideoUtil._optional_int(video_stream.get("nb_frames"))
            duration = VideoUtil._optional_float(video_stream.get("duration"))
            if duration is None:
                duration = VideoUtil._optional_float((payload.get("format") or {}).get("duration"))
            if source_frame_count is None and duration is not None:
                source_frame_count = max(1, int(round(duration * fps)))
            return SourceVideoInfo(
                fps=fps,
                source_width=int(video_stream["width"]),
                source_height=int(video_stream["height"]),
                source_frame_count=source_frame_count,
                source_duration_seconds=duration,
                audio_present=audio_present,
            )
        except (KeyError, StopIteration, TypeError, ValueError):
            return None

    @staticmethod
    def _fraction_from_string(value: str | None) -> Fraction | None:
        if not value or value == "0/0":
            return None
        with suppress(ZeroDivisionError, ValueError):
            return Fraction(value)
        return None

    @staticmethod
    def _optional_int(value) -> int | None:
        if value in (None, "N/A"):
            return None
        with suppress(TypeError, ValueError):
            return int(value)
        return None

    @staticmethod
    def _optional_float(value) -> float | None:
        if value in (None, "N/A"):
            return None
        with suppress(TypeError, ValueError):
            return float(value)
        return None

    @staticmethod
    def read_video_frame_window(
        path: str | Path,
        *,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> DecodedVideoClip:
        if start_frame < 0:
            raise ValueError("start_frame must be greater than or equal to zero.")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be greater than zero when provided.")

        import av

        source_info = VideoUtil.inspect_video(path)
        file_path = Path(path)
        with av.open(str(file_path)) as container:
            video_stream = container.streams.video[0]
            frames: list[PIL.Image.Image] = []
            for frame_index, frame in enumerate(container.decode(video_stream)):
                if frame_index < start_frame:
                    continue
                frames.append(PIL.Image.fromarray(frame.to_ndarray(format="rgb24")))
                if max_frames is not None and len(frames) >= max_frames:
                    break

        if not frames:
            raise RuntimeError(f"Could not decode any video frames from {path}")

        return DecodedVideoClip(
            frames=frames,
            fps=source_info.fps,
            source_width=source_info.source_width,
            source_height=source_info.source_height,
            source_frame_count=source_info.source_frame_count,
            source_duration_seconds=source_info.source_duration_seconds,
            audio_present=source_info.audio_present,
            clip_start_frame=start_frame,
            clip_frame_count=len(frames),
        )

    @staticmethod
    def iter_video_frame_windows(
        path: str | Path,
        *,
        start_frame: int = 0,
        windows: list[tuple[int, int]],
    ) -> Iterator[DecodedVideoClip]:
        if start_frame < 0:
            raise ValueError("start_frame must be greater than or equal to zero.")
        if not windows:
            raise ValueError("windows must contain at least one frame range.")

        normalized_windows: list[tuple[int, int]] = []
        previous_start = -1
        for window_start, window_end in windows:
            if window_start < 0:
                raise ValueError("window start_frame must be greater than or equal to zero.")
            if window_end <= window_start:
                raise ValueError("window end_frame must be greater than start_frame.")
            if window_start < previous_start:
                raise ValueError("windows must be sorted by start_frame.")
            normalized_windows.append((window_start, window_end))
            previous_start = window_start

        ffmpeg_path = shutil.which("ffmpeg")
        source_info = VideoUtil.inspect_video(path)
        if ffmpeg_path is not None:
            yield from VideoUtil._iter_video_frame_windows_ffmpeg(
                path=Path(path),
                ffmpeg_path=ffmpeg_path,
                source_info=source_info,
                start_frame=start_frame,
                normalized_windows=normalized_windows,
            )
            return

        yield from VideoUtil._iter_video_frame_windows_pyav(
            path=Path(path),
            source_info=source_info,
            start_frame=start_frame,
            normalized_windows=normalized_windows,
        )

    @staticmethod
    def _iter_video_frame_windows_pyav(
        *,
        path: Path,
        source_info: SourceVideoInfo,
        start_frame: int,
        normalized_windows: list[tuple[int, int]],
    ) -> Iterator[DecodedVideoClip]:
        import av

        absolute_windows = [
            (start_frame + window_start, start_frame + window_end) for window_start, window_end in normalized_windows
        ]
        with av.open(str(path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")

            video_stream = container.streams.video[0]
            active_windows: list[dict] = []
            next_window_index = 0
            for frame_index, frame in enumerate(container.decode(video_stream)):
                while (
                    next_window_index < len(absolute_windows) and absolute_windows[next_window_index][0] == frame_index
                ):
                    absolute_start, absolute_end = absolute_windows[next_window_index]
                    relative_start, _ = normalized_windows[next_window_index]
                    active_windows.append(
                        {
                            "absolute_end": absolute_end,
                            "relative_start": relative_start,
                            "frames": [],
                        }
                    )
                    next_window_index += 1

                if not active_windows:
                    if next_window_index >= len(absolute_windows):
                        break
                    if frame_index < absolute_windows[next_window_index][0]:
                        continue

                pil_frame = PIL.Image.fromarray(frame.to_ndarray(format="rgb24"))
                for active_window in active_windows:
                    if frame_index < active_window["absolute_end"]:
                        active_window["frames"].append(pil_frame)

                completed_windows: list[dict] = []
                while active_windows and frame_index + 1 >= active_windows[0]["absolute_end"]:
                    completed_windows.append(active_windows.pop(0))

                for completed_window in completed_windows:
                    yield DecodedVideoClip(
                        frames=completed_window["frames"],
                        fps=source_info.fps,
                        source_width=source_info.source_width,
                        source_height=source_info.source_height,
                        source_frame_count=source_info.source_frame_count,
                        source_duration_seconds=source_info.source_duration_seconds,
                        audio_present=source_info.audio_present,
                        clip_start_frame=completed_window["relative_start"],
                        clip_frame_count=len(completed_window["frames"]),
                    )

        if active_windows or next_window_index < len(absolute_windows):
            raise RuntimeError(f"Could not decode all requested video windows from {path}")

    @staticmethod
    def _iter_video_frame_windows_ffmpeg(
        *,
        path: Path,
        ffmpeg_path: str,
        source_info: SourceVideoInfo,
        start_frame: int,
        normalized_windows: list[tuple[int, int]],
    ) -> Iterator[DecodedVideoClip]:
        absolute_windows = [
            (start_frame + window_start, start_frame + window_end) for window_start, window_end in normalized_windows
        ]
        first_frame = absolute_windows[0][0]
        final_frame_exclusive = max(window_end for _, window_end in absolute_windows)
        if final_frame_exclusive <= first_frame:
            return

        frame_size = source_info.source_width * source_info.source_height * 3
        filter_expression = f"select=between(n\\,{first_frame}\\,{final_frame_exclusive - 1})"
        command = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            filter_expression,
            "-vsync",
            "0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        active_windows: list[dict] = []
        next_window_index = 0
        frame_index = first_frame
        try:
            while True:
                payload = process.stdout.read(frame_size)
                if not payload:
                    break
                if len(payload) != frame_size:
                    raise RuntimeError(f"Decoded video frame payload from {path} has an unexpected size.")

                while (
                    next_window_index < len(absolute_windows) and absolute_windows[next_window_index][0] == frame_index
                ):
                    absolute_start, absolute_end = absolute_windows[next_window_index]
                    relative_start, _ = normalized_windows[next_window_index]
                    active_windows.append(
                        {
                            "absolute_end": absolute_end,
                            "relative_start": relative_start,
                            "frames": [],
                        }
                    )
                    next_window_index += 1

                if active_windows:
                    frame_array = np.frombuffer(payload, dtype=np.uint8).reshape(
                        source_info.source_height,
                        source_info.source_width,
                        3,
                    )
                    pil_frame = PIL.Image.fromarray(frame_array.copy())
                    for active_window in active_windows:
                        if frame_index < active_window["absolute_end"]:
                            active_window["frames"].append(pil_frame)

                completed_windows: list[dict] = []
                while active_windows and frame_index + 1 >= active_windows[0]["absolute_end"]:
                    completed_windows.append(active_windows.pop(0))

                for completed_window in completed_windows:
                    yield DecodedVideoClip(
                        frames=completed_window["frames"],
                        fps=source_info.fps,
                        source_width=source_info.source_width,
                        source_height=source_info.source_height,
                        source_frame_count=source_info.source_frame_count,
                        source_duration_seconds=source_info.source_duration_seconds,
                        audio_present=source_info.audio_present,
                        clip_start_frame=completed_window["relative_start"],
                        clip_frame_count=len(completed_window["frames"]),
                    )
                frame_index += 1
        finally:
            if process.stdout is not None:
                process.stdout.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Could not decode video frame windows from {path}: {message or return_code}")
        if active_windows or next_window_index < len(absolute_windows):
            raise RuntimeError(f"Could not decode all requested video windows from {path}")

    @staticmethod
    def _latents_to_frames(decoded_latents: mx.array) -> list[PIL.Image.Image]:
        video_np = VideoUtil._latents_to_frame_arrays(decoded_latents)
        return [PIL.Image.fromarray(frame) for frame in video_np]

    @staticmethod
    def _pil_rgb_to_array(image: PIL.Image.Image) -> np.ndarray:
        # Buffer-protocol conversion; getdata() would build ~1M Python tuples per frame
        # (measured ~350x slower) and is removed in Pillow 14.
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.uint8)

    @staticmethod
    def _latents_to_frame_batches(decoded_latents: mx.array, batch_size: int = 8) -> Iterable[list[PIL.Image.Image]]:
        total_frames = int(decoded_latents.shape[2])
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            video_np = VideoUtil._latents_to_frame_arrays(
                decoded_latents[:, :, start:end, :, :],
                frame_offset=start,
                total_frames=total_frames,
            )
            yield [PIL.Image.fromarray(frame) for frame in video_np]

    @staticmethod
    def decoded_latent_slices_to_frame_batches(
        decoded_latent_slices: Iterable[mx.array],
        batch_size: int = 8,
        total_frames: int | None = None,
    ) -> Iterable[list[PIL.Image.Image]]:
        pending: list[PIL.Image.Image] = []
        frame_offset = 0
        for decoded_slice in decoded_latent_slices:
            remaining_frames = None if total_frames is None else total_frames - frame_offset
            if remaining_frames is not None and remaining_frames <= 0:
                break
            if remaining_frames is not None and int(decoded_slice.shape[2]) > remaining_frames:
                decoded_slice = decoded_slice[:, :, :remaining_frames]
            video_np = VideoUtil._latents_to_frame_arrays(
                decoded_slice,
                frame_offset=frame_offset,
                total_frames=total_frames,
            )
            if remaining_frames is not None:
                video_np = video_np[:remaining_frames]
            for frame in video_np:
                pending.append(PIL.Image.fromarray(frame))
                frame_offset += 1
                if len(pending) >= batch_size:
                    yield pending
                    pending = []
        if pending:
            yield pending
        if total_frames is not None and frame_offset != total_frames:
            raise ValueError(f"Expected {total_frames} decoded frames, got {frame_offset}.")

    @staticmethod
    def _latents_to_frame_arrays(
        decoded_latents: mx.array,
        frame_offset: int = 0,
        total_frames: int | None = None,
    ) -> np.ndarray:
        if decoded_latents.ndim != 5:
            raise ValueError(f"Expected decoded video latents with shape [B, C, F, H, W], got {decoded_latents.shape}")
        video = decoded_latents[0]
        video = mx.transpose(video, (1, 2, 3, 0)).astype(mx.float32)
        video_np = np.array(video, dtype=np.float32)
        total_frames = total_frames or video_np.shape[0]
        for frame_index in range(video_np.shape[0]):
            frame_np = video_np[frame_index]
            TensorHealth.ensure_finite(
                frame_np,
                name="decoded_video_frame",
                phase="video-frame-conversion",
                frame=frame_offset + frame_index + 1,
                total_frames=total_frames,
            )
        return (np.clip(video_np / 2 + 0.5, 0, 1) * 255).round().astype("uint8")

    @staticmethod
    def _save_video_with_pyav(
        frames: list[PIL.Image.Image],
        file_path: Path,
        fps: int | float,
        width: int,
        height: int,
    ) -> None:
        with VideoStreamWriter(path=file_path, fps=fps, width=width, height=height, overwrite=True) as writer:
            writer.write_frames(frames)

    @staticmethod
    def _save_video_batches_with_pyav(
        frame_batches: Iterable[list[PIL.Image.Image]],
        file_path: Path,
        fps: int | float,
        width: int,
        height: int,
    ) -> None:
        with VideoStreamWriter(path=file_path, fps=fps, width=width, height=height, overwrite=True) as writer:
            for batch in frame_batches:
                writer.write_frames(batch)

    @staticmethod
    def _save_metadata(file_path: Path, metadata: dict | None, export_json_metadata: bool) -> None:
        if export_json_metadata and metadata is not None:
            from mflux.utils.generated_video import GeneratedVideo

            GeneratedVideo.save_metadata(file_path, metadata)

    @staticmethod
    def _metadata_frame_count(metadata: dict | None) -> int | None:
        if metadata is None or metadata.get("frames") is None:
            return None
        try:
            return int(metadata["frames"])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_audio_copy_command(
        *,
        ffmpeg_path: str,
        restored_video_path: Path,
        source_video_path: Path,
        clip_start_seconds: float,
        clip_duration_seconds: float,
        output_path: Path,
    ) -> list[str]:
        return [
            ffmpeg_path,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(restored_video_path),
            "-ss",
            f"{clip_start_seconds:.6f}",
            "-t",
            f"{clip_duration_seconds:.6f}",
            "-i",
            str(source_video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1:first_pts=0",
            "-movflags",
            "+faststart",
            # No -shortest: the audio input is already bounded by -ss/-t, and -shortest can
            # truncate stream-copied video packets at audio EOF (drops trailing frames on
            # short clips). Post-mux validation still enforces exact frame count and durations.
            str(output_path),
        ]

    @staticmethod
    def _validate_copied_audio_output(
        *,
        output_path: Path,
        expected_video: SourceVideoInfo,
        expected_audio_duration_seconds: float,
    ) -> str | None:
        output_info = VideoUtil.inspect_video(output_path)
        if not output_info.audio_present:
            return "output_missing_audio"
        if (
            output_info.source_width != expected_video.source_width
            or output_info.source_height != expected_video.source_height
        ):
            return "output_video_dimensions_mismatch"
        if abs(output_info.fps - expected_video.fps) > 1e-6:
            return "output_video_fps_mismatch"
        if (
            expected_video.source_frame_count is not None
            and output_info.source_frame_count is not None
            and output_info.source_frame_count != expected_video.source_frame_count
        ):
            return "output_video_frame_count_mismatch"
        if (
            expected_video.source_duration_seconds is not None
            and output_info.source_duration_seconds is not None
            and abs(output_info.source_duration_seconds - expected_video.source_duration_seconds)
            > VideoUtil._media_alignment_tolerance_seconds(expected_video.fps)
        ):
            return "output_video_duration_mismatch"

        audio_duration_seconds = VideoUtil._audio_duration_seconds(output_path)
        if audio_duration_seconds is None:
            return "output_audio_duration_unknown"
        if abs(audio_duration_seconds - expected_audio_duration_seconds) > VideoUtil._media_alignment_tolerance_seconds(
            expected_video.fps
        ):
            return "output_audio_duration_mismatch"
        return None

    @staticmethod
    def _duration_seconds(
        *,
        container,
        video_stream,
        fps: float,
        source_frame_count: int | None,
    ) -> float | None:
        if video_stream.duration is not None and video_stream.time_base is not None:
            return float(video_stream.duration * video_stream.time_base)
        if container.duration is not None:
            return float(container.duration) / 1_000_000.0
        if source_frame_count is not None and fps > 0:
            return source_frame_count / fps
        return None

    @staticmethod
    def _fps_to_rate(fps: int | float) -> Fraction:
        if float(fps).is_integer():
            return Fraction(int(fps), 1)
        return Fraction(str(float(fps))).limit_denominator(1001)

    @staticmethod
    def _rate_to_float(rate: Fraction | int | float | None) -> float | None:
        if rate is None:
            return None
        return float(rate)

    @staticmethod
    def _audio_duration_seconds(path: str | Path) -> float | None:
        import av

        with av.open(str(path)) as container:
            if len(container.streams.audio) == 0:
                return None
            audio_stream = container.streams.audio[0]
            if audio_stream.duration is not None and audio_stream.time_base is not None:
                return float(audio_stream.duration * audio_stream.time_base)
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
        return None

    @staticmethod
    def _media_alignment_tolerance_seconds(fps: float) -> float:
        return max(1.0 / max(float(fps), 1.0), 0.05)
