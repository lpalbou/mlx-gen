import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.utils.runtime_memory import RuntimeMemory


class Flux2PromptEncoder:
    # Bounded LRU on the model's prompt_cache dict: ~12-13 MB of embeds per Klein 9B
    # entry, so 8 entries stay under ~100 MB while covering the multi-seed and
    # edit-host repeat patterns (0095). Embedding hosts stream many prompts through
    # one resident worker, hence the bound (flux/qwen caches are unbounded dicts).
    PROMPT_CACHE_MAX_ENTRIES = 8

    @staticmethod
    def encode_prompt(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
        prompt_cache: dict[str, tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        # Key on everything that changes the embeds, not just the prompt string,
        # so one cache dict safely serves any call pattern.
        cache_key = repr((prompt, num_images_per_prompt, max_sequence_length, text_encoder_out_layers))
        if prompt_cache is not None and cache_key in prompt_cache:
            # LRU touch: re-insert so streaming hosts evict the oldest prompt first.
            cached = prompt_cache.pop(cache_key)
            prompt_cache[cache_key] = cached
            return cached

        prompt_embeds = Flux2PromptEncoder._get_qwen3_prompt_embeds(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        if num_images_per_prompt > 1:
            prompt_embeds = mx.repeat(prompt_embeds, num_images_per_prompt, axis=0)
        text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_embeds)
        prompt_embeds, text_ids = RuntimeMemory.materialize_tensors(prompt_embeds, text_ids)

        if prompt_cache is not None:
            prompt_cache[cache_key] = (prompt_embeds, text_ids)
            while len(prompt_cache) > Flux2PromptEncoder.PROMPT_CACHE_MAX_ENTRIES:
                prompt_cache.pop(next(iter(prompt_cache)))
        return prompt_embeds, text_ids

    @staticmethod
    def _get_qwen3_prompt_embeds(
        prompt: str | list[str],
        tokenizer: Tokenizer,
        text_encoder: Qwen3TextEncoder,
        max_sequence_length: int,
        hidden_state_layers: tuple[int, ...],
    ) -> mx.array:
        tokens = tokenizer.tokenize(prompt=prompt, max_length=max_sequence_length)
        return text_encoder.get_prompt_embeds(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            hidden_state_layers=hidden_state_layers,
        )

    @staticmethod
    def prepare_text_ids(x: mx.array, t_coord: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        out_ids = []
        for i in range(batch_size):
            if t_coord is None:
                t = mx.zeros((seq_len,), dtype=mx.int32)
            else:
                t = t_coord[i]
                if t.ndim == 0:
                    t = mx.full((seq_len,), t, dtype=mx.int32)
                elif t.shape[0] != seq_len:
                    t = mx.broadcast_to(t, (seq_len,))
                t = t.astype(mx.int32)
            h = mx.zeros((seq_len,), dtype=mx.int32)
            w = mx.zeros((seq_len,), dtype=mx.int32)
            token_ids = mx.arange(seq_len, dtype=mx.int32)
            coords = mx.stack([t, h, w, token_ids], axis=1)
            out_ids.append(coords)
        return mx.stack(out_ids, axis=0)
