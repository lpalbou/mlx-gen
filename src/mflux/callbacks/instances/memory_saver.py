import mlx.core as mx
import PIL.Image
from tqdm import tqdm

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback
from mflux.models.common.config.config import Config
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.utils.runtime_memory import RuntimeMemory


class MemorySaver(BeforeLoopCallback, InLoopCallback, AfterLoopCallback):
    def __init__(self, model, keep_transformer: bool = True, cache_limit_bytes: int = 1000**3, args=None):
        self.model = model
        self.keep_transformer = keep_transformer
        self.peak_memory: int = 0
        self.latest_snapshot = None
        self.model.tiling_config = TilingConfig()
        mx.set_cache_limit(cache_limit_bytes)
        mx.clear_cache()
        mx.reset_peak_memory()

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        self.latest_snapshot = RuntimeMemory.snapshot("low-ram-before-loop", tensors=(latents,), synchronize=True)
        self._delete_text_encoders()

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        if not self.keep_transformer:
            self._delete_transformer()
        self.latest_snapshot = RuntimeMemory.snapshot("low-ram-after-loop", tensors=(latents,), synchronize=True)

    def _delete_text_encoders(self) -> None:
        # repeated image generation only works with the same prompt (cache)
        if hasattr(self.model, "clip_text_encoder"):
            self.model.clip_text_encoder = None
        if hasattr(self.model, "t5_text_encoder"):
            self.model.t5_text_encoder = None
        if hasattr(self.model, "text_encoder") and self.model.text_encoder is not None:
            self.model.text_encoder = None
        if hasattr(self.model, "image_encoder") and self.model.image_encoder is not None:
            self.model.image_encoder = None
        if hasattr(self.model, "image_embedder") and self.model.image_embedder is not None:
            self.model.image_embedder = None
        if hasattr(self.model, "depth_pro") and self.model.depth_pro is not None:
            self.model.depth_pro = None
        if hasattr(self.model, "qwen_vl_encoder") and self.model.qwen_vl_encoder is not None:
            self.model.qwen_vl_encoder = None
        # Clear VLM tokenizers from the tokenizers dict if present
        if hasattr(self.model, "tokenizers") and "qwen_vl" in self.model.tokenizers:
            self.model.tokenizers["qwen_vl"] = None
        mx.clear_cache()

    def _delete_transformer(self) -> None:
        # Compiled predicts close over the transformer; drop them first so the
        # release below actually frees the weights (wan 0090 d12 discipline).
        compiled_cache = getattr(self.model, "compiled_predict_cache", None)
        if compiled_cache is not None:
            compiled_cache.clear()
        self.model.transformer = None
        if hasattr(self.model, "transformer_controlnet"):
            self.model.transformer_controlnet = None
        mx.clear_cache()

    def memory_stats(self) -> str:
        snapshot = RuntimeMemory.snapshot("low-ram-summary", synchronize=True)
        self.latest_snapshot = snapshot
        self.peak_memory = snapshot.mlx_peak_memory_bytes or 0
        parts = [f"Peak MLX memory: {self.peak_memory / 10**9:.2f} GB"]
        if snapshot.process_peak_rss_bytes is not None:
            parts.append(f"Peak RSS: {snapshot.process_peak_rss_bytes / 10**9:.2f} GB")
        if snapshot.darwin_physical_footprint_bytes is not None:
            parts.append(f"Physical footprint: {snapshot.darwin_physical_footprint_bytes / 10**9:.2f} GB")
        return "; ".join(parts)
