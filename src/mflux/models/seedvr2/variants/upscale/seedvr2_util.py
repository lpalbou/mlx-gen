import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.utils.scale_factor import ScaleFactor


@dataclass(frozen=True)
class StreamedVideoChunk:
    input_start_frame: int
    input_end_frame: int
    trim_leading_context_frames: int
    output_frame_count: int
    target_input_frame_count: int


class SeedVR2Util:
    PATCH_AREA = 4
    LATENT_SPATIAL_SCALE = 8
    VIDEO_SAFE_MEMORY_FRACTION = 0.5
    VIDEO_SAFE_MEMORY_CEILING_BYTES = 48 * (1000**3)
    VIDEO_RUNTIME_SLACK_BYTES = 6 * (1000**3)
    VIDEO_MIN_PRODUCTION_STREAMING_CHUNK_FRAMES = 29
    VIDEO_MIN_PRODUCTION_STREAMING_OVERLAP_FRAMES = 8
    # The video VAE (encode and decode in causal slices of whole frames) is most of a chunk's working memory and
    # grows with the output frame area, not with the chunk length. Measured through the full restoration
    # (`restore_video_to_path` with the plan's values) on MLX 0.32.5 with the per-frame convolution taps of
    # CausalConv3d, on SeedVR2 3B (the 7B variants use the same VAE; their working set was not measured), table and
    # fit in tests/seedvr2/test_seedvr2_video_vae_memory.py. Rows: (largest padded chunk frames, bytes per output
    # pixel); above 49 frames the transformer terms of the estimate over-count, so the last row is lower. Where the
    # per-pixel term is smaller, the floor applies instead (up to about 44,000 output pixels for 9 to 29 frames):
    # under 16,384 pixels the decoder's half-resolution layers have frames under 4096 pixels and keep MLX's single
    # call, which needs up to 1.32 GB (144x112). Re-measure before widening the MLX range.
    VIDEO_VAE_MEMORY_MLX_RANGE = ((0, 32, 4), (0, 33, 0))
    VIDEO_VAE_MEMORY_FLOOR_BYTES = 1_400_000_000
    VIDEO_VAE_MEMORY_BUCKETS = (
        (1, 12_000),
        (5, 25_000),
        (29, 32_000),
        (49, 30_000),
        (None, 28_000),
    )

    @staticmethod
    def preprocess_image(
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> tuple[mx.array, int, int]:
        image = Image.open(image_path).convert("RGB")
        resized, true_h, true_w = SeedVR2Util._resize_and_soften(image=image, resolution=resolution, softness=softness)
        resized = SeedVR2Util._pad_to_multiple(resized, factor=16)
        img_mx = SeedVR2Util._pil_to_mx_image(resized)
        return img_mx, true_h, true_w

    @staticmethod
    def resolved_video_frame_size(
        *,
        source_width: int,
        source_height: int,
        resolution: int | ScaleFactor,
    ) -> tuple[int, int]:
        """Exact ``(height, width)`` that :meth:`preprocess_video_frames` will produce.

        Derived by running the real resize-and-crop path on a placeholder frame instead of
        reimplementing the arithmetic. Callers that must agree with the frames the VAE
        actually encodes - the streamed noise provider, the memory budget, progress
        reporting - cannot then drift from it. A parallel reimplementation is what produced
        a streamed latent-width mismatch on sources whose scaled size is not already a
        multiple of 16.
        """
        probe = Image.new("RGB", (int(source_width), int(source_height)))
        resized, _, _ = SeedVR2Util._resize_and_soften(image=probe, resolution=resolution, softness=0.0)
        cropped = SeedVR2Util._center_crop_to_multiple(resized, factor=16)
        width, height = cropped.size
        return height, width

    @staticmethod
    def preprocess_video_frames(
        frames: list[Image.Image],
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> tuple[mx.array, int, int]:
        if not frames:
            raise ValueError("preprocess_video_frames requires at least one frame.")

        first = frames[0].convert("RGB")
        resized_first, _, _ = SeedVR2Util._resize_and_soften(
            image=first,
            resolution=resolution,
            softness=softness,
        )
        cropped_first = SeedVR2Util._center_crop_to_multiple(resized_first, factor=16)
        true_w, true_h = cropped_first.size

        frame_count = len(frames)
        video_np = np.empty((frame_count, true_h, true_w, 3), dtype=np.float32)
        video_np[0] = SeedVR2Util._pil_to_numpy_video_frame(cropped_first)
        for index, frame in enumerate(frames[1:], start=1):
            rgb_frame = frame.convert("RGB")
            resized, _, _ = SeedVR2Util._resize_and_soften(
                image=rgb_frame,
                resolution=resolution,
                softness=softness,
            )
            cropped = SeedVR2Util._center_crop_to_multiple(resized, factor=16)
            if cropped.size != (true_w, true_h):
                cropped = cropped.resize((true_w, true_h), Image.Resampling.BICUBIC)
            video_np[index] = SeedVR2Util._pil_to_numpy_video_frame(cropped)

        video_mx = mx.array(video_np, dtype=mx.float32)
        video_mx = mx.transpose(video_mx, (3, 0, 1, 2))
        video_mx = video_mx[None, ...]
        return video_mx, true_h, true_w

    @staticmethod
    def apply_color_correction(
        content: mx.array,
        style: mx.array,
        mode: str = "lab",
        luminance_weight: float = 0.8,
    ) -> mx.array:
        if mode == "off":
            return content
        if mode == "wavelet":
            return SeedVR2Util._apply_wavelet_color_reconstruction(content=content, style=style)
        if mode != "lab":
            raise ValueError(f"Unsupported SeedVR2 color correction mode: {mode}")
        if content.ndim == 5 and style.ndim == 5:
            return SeedVR2Util._apply_video_color_correction_framewise(
                content=content,
                style=style,
                luminance_weight=luminance_weight,
            )
        return SeedVR2Util._lab_color_transfer_exact(content, style, luminance_weight=luminance_weight)

    @staticmethod
    def _apply_video_color_correction_framewise(
        content: mx.array,
        style: mx.array,
        luminance_weight: float = 0.8,
    ) -> mx.array:
        if content.shape != style.shape:
            raise ValueError(f"Video color correction requires same shapes, got {content.shape} vs {style.shape}")

        frame_outputs: list[mx.array] = []
        for frame_index in range(content.shape[2]):
            corrected_frame = SeedVR2Util._lab_color_transfer_exact(
                content[:, :, frame_index, :, :],
                style[:, :, frame_index, :, :],
                luminance_weight=luminance_weight,
            )
            frame_outputs.append(corrected_frame[:, :, None, :, :])
        return mx.concatenate(frame_outputs, axis=2)

    @staticmethod
    def _apply_wavelet_color_reconstruction(content: mx.array, style: mx.array) -> mx.array:
        if content.shape != style.shape:
            raise ValueError(f"Wavelet reconstruction requires same shapes, got {content.shape} vs {style.shape}")

        if content.ndim == 5:
            frame_outputs: list[mx.array] = []
            for frame_index in range(content.shape[2]):
                reconstructed_frame = SeedVR2Util._apply_wavelet_color_reconstruction(
                    content[:, :, frame_index, :, :],
                    style[:, :, frame_index, :, :],
                )
                frame_outputs.append(reconstructed_frame[:, :, None, :, :])
            return mx.concatenate(frame_outputs, axis=2)

        content_np = np.array(content.astype(mx.float32), dtype=np.float32)
        style_np = np.array(style.astype(mx.float32), dtype=np.float32)
        reconstructed = SeedVR2Util._wavelet_reconstruction(content_np, style_np)
        return mx.array(reconstructed, dtype=content.dtype)

    @staticmethod
    def pad_video_frames(video: mx.array) -> tuple[mx.array, int]:
        if video.ndim != 5:
            raise ValueError(f"Expected video tensor [B, C, T, H, W], got {video.shape}")

        frame_count = int(video.shape[2])
        if frame_count == 1:
            return video, frame_count
        if (frame_count - 1) % 4 == 0:
            return video, frame_count

        pad_frames = 4 - ((frame_count - 1) % 4)
        last_frame = video[:, :, -1:, :, :]
        padding = mx.repeat(last_frame, pad_frames, axis=2)
        return mx.concatenate([video, padding], axis=2), frame_count

    @staticmethod
    def padded_video_frame_count(frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero.")
        if frame_count == 1 or (frame_count - 1) % 4 == 0:
            return frame_count
        return frame_count + (4 - ((frame_count - 1) % 4))

    @staticmethod
    def latent_video_frame_count(frame_count: int) -> int:
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero.")
        if frame_count == 1:
            return 1
        return ((frame_count - 1) // 4) + 1

    @staticmethod
    def streamed_video_temporal_quality_error(
        *,
        frame_count: int,
        chunk_size: int,
        overlap: int,
    ) -> str | None:
        if frame_count <= 0 or chunk_size <= 0:
            raise ValueError("frame_count and chunk_size must be greater than zero.")
        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to zero.")
        if chunk_size >= frame_count:
            return None
        if chunk_size < SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_CHUNK_FRAMES:
            return (
                "SeedVR2 video restore refuses temporal chunks smaller than "
                f"{SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_CHUNK_FRAMES} frames because short "
                "independent chunks can distort object continuity. Use whole-shot restore, reduce "
                "the clip/resolution, or use a larger --temporal-chunk-size."
            )
        if overlap < SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_OVERLAP_FRAMES:
            return (
                "SeedVR2 video restore refuses chunked overlap smaller than "
                f"{SeedVR2Util.VIDEO_MIN_PRODUCTION_STREAMING_OVERLAP_FRAMES} frames because "
                "insufficient context can distort object continuity. Use whole-shot restore, "
                "reduce the clip/resolution, or increase --temporal-chunk-overlap."
            )
        return None

    @staticmethod
    def plan_streamed_video_chunks(
        frame_count: int,
        chunk_size: int,
        overlap: int,
    ) -> list[StreamedVideoChunk]:
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to zero.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        if chunk_size > 1 and (chunk_size - 1) % 4 != 0:
            raise ValueError("chunk_size must satisfy 4n+1 for streamed SeedVR2 video restore.")
        if overlap % 4 != 0:
            raise ValueError("overlap must be a multiple of 4 for streamed SeedVR2 video restore.")

        chunks: list[StreamedVideoChunk] = []
        input_frame_count = min(chunk_size, frame_count)
        output_stride = input_frame_count if input_frame_count >= frame_count else input_frame_count - overlap - 1
        if output_stride <= 0:
            raise ValueError("overlap must leave at least one output frame per streamed SeedVR2 video chunk.")
        output_start = 0
        while output_start < frame_count:
            remaining_output_frames = frame_count - output_start
            input_start_frame = max(0, output_start - overlap)
            input_end_frame = min(frame_count, input_start_frame + input_frame_count)
            if input_end_frame - input_start_frame < input_frame_count:
                input_start_frame = max(0, input_end_frame - input_frame_count)
            output_count = remaining_output_frames if input_end_frame >= frame_count else min(output_stride, remaining_output_frames)
            trim_leading_context_frames = output_start - input_start_frame
            target_input_frame_count = SeedVR2Util.padded_video_frame_count(input_end_frame - input_start_frame)
            chunks.append(
                StreamedVideoChunk(
                    input_start_frame=input_start_frame,
                    input_end_frame=input_end_frame,
                    trim_leading_context_frames=trim_leading_context_frames,
                    output_frame_count=output_count,
                    target_input_frame_count=target_input_frame_count,
                )
            )
            output_start += output_count
        return chunks

    @staticmethod
    def estimate_video_restore_working_set_bytes(
        *,
        frame_count: int,
        height: int,
        width: int,
        inner_dim: int,
        text_attention_mode: str,
    ) -> int:
        if frame_count <= 0 or height <= 0 or width <= 0 or inner_dim <= 0:
            raise ValueError("frame_count, height, width, and inner_dim must be greater than zero.")

        latent_height = max(1, height // SeedVR2Util.LATENT_SPATIAL_SCALE)
        latent_width = max(1, width // SeedVR2Util.LATENT_SPATIAL_SCALE)
        patch_tokens_per_frame = max(1, (latent_height // 2) * (latent_width // 2))
        video_tokens = frame_count * patch_tokens_per_frame
        qkv_bytes = video_tokens * 3 * inner_dim * 2
        attention_multiplier = 2.75 if text_attention_mode == "global_text" else 2.0

        processed_video_bytes = frame_count * height * width * 3 * 4
        decoded_video_bytes = frame_count * height * width * 3 * 2
        latent_bytes = frame_count * latent_height * latent_width * 16 * 2

        return int(
            (qkv_bytes * attention_multiplier)
            + processed_video_bytes
            + decoded_video_bytes
            + (latent_bytes * 4)
            + SeedVR2Util.estimate_video_vae_working_set_bytes(frame_count=frame_count, height=height, width=width)
        )

    @staticmethod
    def estimate_video_vae_working_set_bytes(*, frame_count: int, height: int, width: int) -> int:
        for max_frames, per_pixel in SeedVR2Util.VIDEO_VAE_MEMORY_BUCKETS:
            if max_frames is None or frame_count <= max_frames:
                return max(SeedVR2Util.VIDEO_VAE_MEMORY_FLOOR_BYTES, int(height * width * per_pixel))
        raise AssertionError("the last VAE memory bucket has no frame limit")

    @staticmethod
    def estimate_video_restore_total_bytes(
        *,
        frame_count: int,
        height: int,
        width: int,
        inner_dim: int,
        text_attention_mode: str,
        resident_weight_bytes: int,
    ) -> int:
        return int(
            resident_weight_bytes
            + SeedVR2Util.estimate_video_restore_working_set_bytes(
                frame_count=frame_count,
                height=height,
                width=width,
                inner_dim=inner_dim,
                text_attention_mode=text_attention_mode,
            )
        )

    @staticmethod
    def video_restore_budget_bytes(*, weights_resident: bool, weight_bytes: int = 0) -> int:
        # The one working-memory budget for video restore: the plan and every run-time check
        # use it. Available memory is read now, and weights already read in this process are
        # missing from it, so they are reserved only while they are still to be read: a plan
        # made before the load, and a fresh model's first chunk (its weights are bound lazily).
        # Once read (weights_resident=True), reserving them again would refuse chunks the plan
        # admitted.
        return SeedVR2Util.host_safe_video_memory_budget_bytes(reserve_bytes=0 if weights_resident else weight_bytes)

    @staticmethod
    def host_safe_video_memory_budget_bytes(*, reserve_bytes: int = 0) -> int:
        total_bytes = SeedVR2Util._host_total_memory_bytes()
        available_bytes = SeedVR2Util._host_available_memory_bytes()
        basis_bytes = available_bytes if available_bytes is not None and available_bytes > 0 else total_bytes
        if reserve_bytes > 0:
            basis_bytes = max(1 * (1000**3), basis_bytes - int(reserve_bytes))
        target = int(basis_bytes * SeedVR2Util.VIDEO_SAFE_MEMORY_FRACTION)
        return max(1 * (1000**3), min(target, SeedVR2Util.VIDEO_SAFE_MEMORY_CEILING_BYTES))

    @staticmethod
    def _host_total_memory_bytes() -> int:
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                value = int(result.stdout.strip())
                if value > 0:
                    return value
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass

        page_size_name = "SC_PAGE_SIZE"
        page_count_name = "SC_PHYS_PAGES"
        if hasattr(os, "sysconf") and page_size_name in os.sysconf_names and page_count_name in os.sysconf_names:
            page_size = int(os.sysconf(page_size_name))
            page_count = int(os.sysconf(page_count_name))
            if page_size > 0 and page_count > 0:
                return page_size * page_count

        return 32 * (1000**3)

    @staticmethod
    def _host_available_memory_bytes() -> int | None:
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                lines = result.stdout.splitlines()
                if not lines:
                    return None
                page_size_line = lines[0]
                page_size = int(page_size_line.split("page size of", 1)[1].split("bytes", 1)[0].strip(" )"))
                page_counts: dict[str, int] = {}
                for line in lines[1:]:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    digits = value.strip().rstrip(".")
                    if digits.isdigit():
                        page_counts[key.strip()] = int(digits)
                available_pages = (
                    page_counts.get("Pages free", 0)
                    + page_counts.get("Pages inactive", 0)
                    + page_counts.get("Pages speculative", 0)
                )
                if page_size > 0 and available_pages > 0:
                    return page_size * available_pages
            except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
                return None
        return None

    @staticmethod
    def mlx_active_memory_bytes() -> int | None:
        try:
            return int(mx.get_active_memory())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def mlx_peak_memory_bytes() -> int | None:
        try:
            return int(mx.get_peak_memory())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def mlx_cache_memory_bytes() -> int | None:
        try:
            return int(mx.get_cache_memory())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _lab_color_transfer_exact(content: mx.array, style: mx.array, luminance_weight: float = 0.8) -> mx.array:
        content_f = content.astype(mx.float32)
        style_f = style.astype(mx.float32)

        content_np = np.array(content_f, dtype=np.float32)
        style_np = np.array(style_f, dtype=np.float32)

        content_np = SeedVR2Util._wavelet_reconstruction(content_np, style_np)

        c = np.transpose(content_np, (0, 2, 3, 1))
        s = np.transpose(style_np, (0, 2, 3, 1))
        c = np.clip((c + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        s = np.clip((s + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

        c_lab = SeedVR2Util._rgb_to_lab(c)
        s_lab = SeedVR2Util._rgb_to_lab(s)

        matched_a = SeedVR2Util._hist_match(c_lab[..., 1], s_lab[..., 1])
        matched_b = SeedVR2Util._hist_match(c_lab[..., 2], s_lab[..., 2])

        if luminance_weight < 1.0:
            matched_L = SeedVR2Util._hist_match(c_lab[..., 0], s_lab[..., 0])
            L = luminance_weight * c_lab[..., 0] + (1.0 - luminance_weight) * matched_L
        else:
            L = c_lab[..., 0]

        out_lab = np.stack([L, matched_a, matched_b], axis=-1)
        out_rgb = SeedVR2Util._lab_to_rgb(out_lab)
        out_rgb = np.clip(out_rgb, 0.0, 1.0)

        out = out_rgb * 2.0 - 1.0
        out = mx.array(out, dtype=mx.float32)
        out = mx.transpose(out, (0, 3, 1, 2))
        return out.astype(content.dtype)

    @staticmethod
    def _wavelet_blur(image: np.ndarray, radius: int) -> np.ndarray:
        if radius < 1:
            radius = 1

        h, w = int(image.shape[-2]), int(image.shape[-1])
        max_safe_radius = max(1, min(h, w) // 8)
        if radius > max_safe_radius:
            radius = max_safe_radius

        kernel = np.array(
            [
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625],
            ],
            dtype=np.float32,
        )

        p = radius
        padded = np.pad(image, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge")

        out = np.zeros_like(image, dtype=np.float32)
        H, W = image.shape[-2], image.shape[-1]

        for ky, dy in enumerate((-1, 0, 1)):
            ys = p + dy * radius
            ye = ys + H
            for kx, dx in enumerate((-1, 0, 1)):
                xs = p + dx * radius
                xe = xs + W
                out += kernel[ky, kx] * padded[:, :, ys:ye, xs:xe]

        return out

    @staticmethod
    def _wavelet_decomposition(image: np.ndarray, levels: int = 5) -> tuple[np.ndarray, np.ndarray]:
        high_freq = np.zeros_like(image, dtype=np.float32)
        cur = image.astype(np.float32)

        for i in range(levels):
            radius = 2**i
            low_freq = SeedVR2Util._wavelet_blur(cur, radius)
            high_freq += cur - low_freq
            cur = low_freq

        return high_freq, cur

    @staticmethod
    def _wavelet_reconstruction(content: np.ndarray, style: np.ndarray) -> np.ndarray:
        if content.shape != style.shape:
            raise ValueError(f"Wavelet reconstruction requires same shapes, got {content.shape} vs {style.shape}")

        content_high, _ = SeedVR2Util._wavelet_decomposition(content, levels=5)
        _, style_low = SeedVR2Util._wavelet_decomposition(style, levels=5)
        return np.clip(content_high + style_low, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    @staticmethod
    def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0031308, 1.055 * np.maximum(x, 0.0) ** (1.0 / 2.4) - 0.055, 12.92 * x)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        rgb_lin = SeedVR2Util._srgb_to_linear(rgb.astype(np.float32))
        M = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=np.float32,
        )
        xyz = np.tensordot(rgb_lin, M.T, axes=([3], [0]))

        xyz[..., 0] /= 0.95047
        xyz[..., 2] /= 1.08883

        eps = 6.0 / 29.0
        eps3 = eps**3
        kappa = (29.0 / 3.0) ** 3

        f = np.where(xyz > eps3, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    @staticmethod
    def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        eps = 6.0 / 29.0
        kappa = (29.0 / 3.0) ** 3

        x = np.where(fx > eps, fx**3, (116.0 * fx - 16.0) / kappa)
        y = np.where(fy > eps, fy**3, (116.0 * fy - 16.0) / kappa)
        z = np.where(fz > eps, fz**3, (116.0 * fz - 16.0) / kappa)

        x *= 0.95047
        z *= 1.08883

        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
        M_inv = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            dtype=np.float32,
        )
        rgb_lin = np.tensordot(xyz, M_inv.T, axes=([3], [0]))
        rgb = SeedVR2Util._linear_to_srgb(rgb_lin)
        return rgb.astype(np.float32)

    @staticmethod
    def _hist_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        out = np.empty_like(source, dtype=np.float32)
        B = source.shape[0]
        for i in range(B):
            src = source[i].reshape(-1).astype(np.float32)
            ref = reference[i].reshape(-1).astype(np.float32)
            src_idx = np.argsort(src, kind="stable")
            ref_sorted = np.sort(ref, kind="stable")
            inv = np.argsort(src_idx, kind="stable")
            out[i] = ref_sorted[inv].reshape(source.shape[1:]).astype(np.float32)
        return out

    # One-step sampling leaves a measurable residue of the seed noise in the x0 estimate on
    # flat/dark content, which the VAE decodes as a regular texture on its 8px stride grid.
    # Detection thresholds: the artifact tile must clearly exceed both an absolute lattice share
    # and the source's own lattice share (measured: defective outputs reach 5%+ while natural
    # content stays at or below its source level, ~0.1-2.3%).
    ONE_STEP_RESIDUE_MIN_LATTICE_PCT = 3.0
    ONE_STEP_RESIDUE_MIN_SOURCE_RATIO = 2.0
    _RESIDUE_TILE = 256

    @staticmethod
    def measure_latent_grid_residue(decoded: mx.array, reference: mx.array) -> tuple[float, float]:
        """Max share (%) of high-pass tile energy on the VAE's 8px lattice, for output and reference."""
        return (
            SeedVR2Util._max_tile_lattice_pct(decoded),
            SeedVR2Util._max_tile_lattice_pct(reference),
        )

    @staticmethod
    def one_step_residue_detected(decoded_pct: float, reference_pct: float) -> bool:
        if not np.isfinite(decoded_pct):
            return False
        threshold = max(
            SeedVR2Util.ONE_STEP_RESIDUE_MIN_LATTICE_PCT,
            SeedVR2Util.ONE_STEP_RESIDUE_MIN_SOURCE_RATIO * max(reference_pct, 0.5),
        )
        return decoded_pct > threshold

    @staticmethod
    def _max_tile_lattice_pct(image: mx.array) -> float:
        size = SeedVR2Util._RESIDUE_TILE
        array = np.asarray(image.astype(mx.float32))
        if array.ndim == 4:
            array = array[0]
        luminance = array.mean(axis=0)
        height, width = luminance.shape
        if height < size or width < size:
            return float("nan")
        window = np.hanning(size)
        fy, fx = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
        high_pass = fy * fy + fx * fx >= size
        on_lattice = ((np.abs(fy) % 32 <= 1) | (np.abs(fy) % 32 >= 31)) & (
            (np.abs(fx) % 32 <= 1) | (np.abs(fx) % 32 >= 31)
        )
        best = float("nan")
        for y0 in range(0, height - size + 1, size):
            for x0 in range(0, width - size + 1, size):
                patch = luminance[y0 : y0 + size, x0 : x0 + size]
                patch = (patch - patch.mean()) * window[:, None] * window[None, :]
                spectrum = np.abs(np.fft.fftshift(np.fft.fft2(patch))) ** 2
                total = spectrum[high_pass].sum()
                if total <= 0:
                    continue
                pct = 100.0 * spectrum[high_pass & on_lattice].sum() / total
                if not np.isfinite(best) or pct > best:
                    best = pct
        return best

    @staticmethod
    def _resize_and_soften(
        *,
        image: Image.Image,
        resolution: int | ScaleFactor,
        softness: float,
    ) -> tuple[Image.Image, int, int]:
        w, h = image.size
        if isinstance(resolution, ScaleFactor):
            scale = float(resolution.value)
        else:
            scale = resolution / min(w, h)

        # The exact requested geometry is kept: network divisibility is handled by padding
        # (images) or center-cropping (video), never by resizing to a snapped size.
        true_w = max(2, round(w * scale))
        true_h = max(2, round(h * scale))
        factor = 1.0 + (max(0.0, min(1.0, softness)) * 7.0)

        if factor <= 1.0 and true_w == w and true_h == h:
            return image.copy(), true_h, true_w

        if factor > 1.0:
            down_w = max(2, int(true_w / factor))
            down_h = max(2, int(true_h / factor))
            down = image.resize((down_w, down_h), Image.Resampling.BICUBIC)
            resized = down.resize((true_w, true_h), Image.Resampling.BICUBIC)
        else:
            resized = image.resize((true_w, true_h), Image.Resampling.BICUBIC)

        return resized, true_h, true_w

    @staticmethod
    def _pad_to_multiple(image: Image.Image, *, factor: int) -> Image.Image:
        width, height = image.size
        pad_w = (factor - (width % factor)) % factor
        pad_h = (factor - (height % factor)) % factor
        if pad_w == 0 and pad_h == 0:
            return image

        # Reflect-pad instead of black-pad: a hard synthetic edge in the conditioning image
        # bleeds into the restored content near the crop boundary.
        arr = np.asarray(image)
        mode = "reflect" if pad_h < arr.shape[0] and pad_w < arr.shape[1] else "edge"
        padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        return Image.fromarray(padded)

    @staticmethod
    def _center_crop_to_multiple(image: Image.Image, *, factor: int) -> Image.Image:
        width, height = image.size
        cropped_width = width - (width % factor)
        cropped_height = height - (height % factor)
        left = max((width - cropped_width) // 2, 0)
        top = max((height - cropped_height) // 2, 0)
        return image.crop((left, top, left + cropped_width, top + cropped_height))

    @staticmethod
    def _pil_to_mx_image(image: Image.Image) -> mx.array:
        img_mx = mx.array(np.array(image)).astype(mx.float32) / 255.0
        img_mx = mx.clip(img_mx, 0.0, 1.0)
        img_mx = img_mx * 2.0 - 1.0
        img_mx = mx.transpose(img_mx, (2, 0, 1))
        return img_mx[None, ...]

    @staticmethod
    def _pil_to_numpy_video_frame(image: Image.Image) -> np.ndarray:
        frame_np = np.asarray(image, dtype=np.float32) / 255.0
        frame_np = np.clip(frame_np, 0.0, 1.0)
        return frame_np * 2.0 - 1.0
