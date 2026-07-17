from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.tokenizer import BaseTokenizer, TokenizerOutput

if TYPE_CHECKING:
    # Annotation-only: an eager transformers import would put torch+transformers (~0.6 s)
    # on the startup path of the ernie CLI and mflux-save.
    from transformers import PreTrainedTokenizer


class ErnieImageTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "longest",
        template: str | None = None,
        use_chat_template: bool = False,
        chat_template_kwargs: dict | None = None,
        add_special_tokens: bool = True,
    ):
        super().__init__(tokenizer, max_length)
        self.padding = padding
        self.template = template
        self.use_chat_template = use_chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.add_special_tokens = add_special_tokens

    def tokenize(
        self,
        prompt: str | list[str],
        images: list[Image.Image] | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> TokenizerOutput:
        max_length = max_length or self.max_length
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        prompts = [p if p is not None else "" for p in prompts]
        prompts = self._format_prompts(prompts)

        encoded_ids = [self._encode_one(p, max_length=max_length) for p in prompts]
        max_len = max(len(ids) for ids in encoded_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = np.full((len(encoded_ids), max_len), pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((len(encoded_ids), max_len), dtype=np.int32)
        for index, ids in enumerate(encoded_ids):
            input_ids[index, : len(ids)] = ids
            attention_mask[index, : len(ids)] = 1

        return TokenizerOutput(
            input_ids=mx.array(input_ids),
            attention_mask=mx.array(attention_mask),
        )

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        if self.template:
            return [self.template.format(p) for p in prompts]
        if self.use_chat_template:
            return [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                for p in prompts
            ]
        return prompts

    def _encode_one(self, prompt: str, max_length: int) -> list[int]:
        tokenized = self.tokenizer(
            prompt,
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        ids = list(tokenized["input_ids"])
        if ids:
            return ids
        bos_token_id = self.tokenizer.bos_token_id
        return [bos_token_id if bos_token_id is not None else 0]
