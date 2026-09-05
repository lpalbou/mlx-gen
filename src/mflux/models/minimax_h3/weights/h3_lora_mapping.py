from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class MiniMaxH3LoRAMapping(LoRAMapping):
    """PEFT LoRAs trained on the diffusers transformer (the lightx2v Turbo adapters): attention
    projections and both feed-forward linears of every transformer block and both token refiner blocks."""

    TARGET_MODULES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    _BLOCK_PREFIXES = ("transformer_blocks.{block}", "token_refiner.refiner_blocks.{block}")
    _STATE_DICT_PREFIXES = ("", "transformer.", "diffusion_model.")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        for block_prefix in MiniMaxH3LoRAMapping._BLOCK_PREFIXES:
            for module in MiniMaxH3LoRAMapping.TARGET_MODULES:
                path = f"{block_prefix}.{module}"
                targets.append(
                    LoRATarget(
                        model_path=path,
                        possible_up_patterns=[
                            f"{p}{path}.lora_B.weight" for p in MiniMaxH3LoRAMapping._STATE_DICT_PREFIXES
                        ],
                        possible_down_patterns=[
                            f"{p}{path}.lora_A.weight" for p in MiniMaxH3LoRAMapping._STATE_DICT_PREFIXES
                        ],
                        possible_alpha_patterns=[f"{p}{path}.alpha" for p in MiniMaxH3LoRAMapping._STATE_DICT_PREFIXES],
                    )
                )
        return targets
