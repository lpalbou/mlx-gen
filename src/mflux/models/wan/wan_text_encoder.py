from contextlib import ExitStack
from pathlib import Path

import numpy as np

from mflux.models.common.weights.loading.streaming_weight_loader import StreamingWeightLoader


class WanTextEncoder:
    # UMT5 in float32 with its weights read from the safetensors one block at a time: each block's weights are
    # converted to float32 on the device right before it runs and dropped right after, so an encode holds about one
    # block (1 GiB with the allocator's staging) instead of the 22 GiB float32 model. The forward is the reference
    # transformers model's own, built without storage, so the result is the float32 model's.
    EMBEDDING = "shared.weight"
    FINAL_LAYER_NORM = "encoder.final_layer_norm."
    # Packages store bfloat16 weights; the Bernini package keeps the feed-forward output layers in float32. A
    # float32 source is rounded the same way, so a source and its package encode a prompt identically.
    FLOAT32_WEIGHTS = ".wo."

    def __init__(self, text_encoder_path: Path):
        import torch
        from transformers import UMT5Config, UMT5EncoderModel

        config = UMT5Config.from_pretrained(text_encoder_path, local_files_only=True)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._files = sorted(Path(text_encoder_path).glob("*.safetensors"))
        self._tensors = StreamingWeightLoader.tensor_index(self._files)
        with torch.device("meta"):
            self._model = UMT5EncoderModel(config).eval().requires_grad_(False)

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        import torch

        try:
            return self._encode(input_ids, attention_mask)
        finally:
            # Also after a failed encode: the allocator hands the block it staged back to the system.
            if self.device.type == "mps":
                torch.mps.empty_cache()

    def _encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        import torch
        from safetensors import safe_open

        # A padded key gets an attention weight of exactly zero, so the real tokens do not depend on the padding:
        # the batch runs at the length of its longest real row instead of the tokenizer's 512.
        length = max(1, int(attention_mask.sum(axis=1).max()))
        ids = torch.from_numpy(np.ascontiguousarray(input_ids[:, :length])).long()
        mask = torch.from_numpy(np.ascontiguousarray(attention_mask[:, :length])).to(self.device)
        encoder = self._model.encoder
        with ExitStack() as stack, torch.no_grad():
            files = {path: stack.enter_context(safe_open(str(path), framework="pt")) for path in self._files}

            def load(module, prefix: str) -> None:
                weights = {
                    name[len(prefix) :]: self._float32(files[path].get_tensor(name), name)
                    for name, (path, _) in self._tensors.items()
                    if name.startswith(prefix)
                }
                module.load_state_dict(weights, strict=True, assign=True)

            # Whatever happens, no real weight outlives the call.
            stack.callback(self._model.to, "meta")
            for index, block in enumerate(encoder.block):
                pre = block.register_forward_pre_hook(
                    lambda module, args, prefix=f"encoder.block.{index}.": load(module, prefix)
                )
                post = block.register_forward_hook(WanTextEncoder._release)
                stack.callback(pre.remove)
                stack.callback(post.remove)
            load(encoder.final_layer_norm, WanTextEncoder.FINAL_LAYER_NORM)
            table = files[self._tensors[WanTextEncoder.EMBEDDING][0]]
            embeds = self._float32(self._embed(ids, table), WanTextEncoder.EMBEDDING)
            return encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state.cpu().numpy()

    def _float32(self, tensor, name: str):
        import torch

        if WanTextEncoder.FLOAT32_WEIGHTS not in name:
            tensor = tensor.to(torch.bfloat16)
        return tensor.to(self.device).float()

    @staticmethod
    def _release(module, args, output) -> None:
        # A forward hook that returned something would replace the block's output.
        module.to("meta")

    @staticmethod
    def _embed(ids, handle):
        import torch

        # Only the rows of the tokens in the batch are read from the 2 GB embedding table.
        rows, inverse = np.unique(ids.numpy(), return_inverse=True)
        table = handle.get_slice(WanTextEncoder.EMBEDDING)
        gathered = torch.cat([table[int(row) : int(row) + 1] for row in rows])
        return gathered[torch.from_numpy(inverse.reshape(ids.shape))]
