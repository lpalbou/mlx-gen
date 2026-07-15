from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms


class WanWeightMapping:
    @staticmethod
    def get_vace_block_mapping(num_vace_blocks: int) -> list[WeightTarget]:
        mapping = [
            WeightTarget(
                to_pattern="vace_patch_embedding.weight",
                from_pattern=["vace_patch_embedding.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(to_pattern="vace_patch_embedding.bias", from_pattern=["vace_patch_embedding.bias"]),
        ]
        for block in range(num_vace_blocks):
            prefix = f"vace_blocks.{block}"
            mapping.extend(
                [
                    WeightTarget(
                        to_pattern=f"{prefix}.scale_shift_table", from_pattern=[f"{prefix}.scale_shift_table"]
                    ),
                    *WanWeightMapping._attention_mapping(f"{prefix}.attn1"),
                    *WanWeightMapping._attention_mapping(f"{prefix}.attn2"),
                    WeightTarget(to_pattern=f"{prefix}.norm2.weight", from_pattern=[f"{prefix}.norm2.weight"]),
                    WeightTarget(to_pattern=f"{prefix}.norm2.bias", from_pattern=[f"{prefix}.norm2.bias"]),
                    WeightTarget(
                        to_pattern=f"{prefix}.ffn.net.0.weight",
                        from_pattern=[f"{prefix}.ffn.net.0.proj.weight"],
                    ),
                    WeightTarget(
                        to_pattern=f"{prefix}.ffn.net.0.bias",
                        from_pattern=[f"{prefix}.ffn.net.0.proj.bias"],
                    ),
                    WeightTarget(to_pattern=f"{prefix}.ffn.net.1.weight", from_pattern=[f"{prefix}.ffn.net.2.weight"]),
                    WeightTarget(to_pattern=f"{prefix}.ffn.net.1.bias", from_pattern=[f"{prefix}.ffn.net.2.bias"]),
                    WeightTarget(to_pattern=f"{prefix}.proj_out.weight", from_pattern=[f"{prefix}.proj_out.weight"]),
                    WeightTarget(to_pattern=f"{prefix}.proj_out.bias", from_pattern=[f"{prefix}.proj_out.bias"]),
                ]
            )
        # Only block 0 carries the input projection in the reference checkpoint.
        mapping.extend(
            [
                WeightTarget(to_pattern="vace_blocks.0.proj_in.weight", from_pattern=["vace_blocks.0.proj_in.weight"]),
                WeightTarget(to_pattern="vace_blocks.0.proj_in.bias", from_pattern=["vace_blocks.0.proj_in.bias"]),
            ]
        )
        return mapping

    @staticmethod
    def get_transformer_mapping(num_layers: int = 30) -> list[WeightTarget]:
        mapping = [
            WeightTarget(to_pattern="scale_shift_table", from_pattern=["scale_shift_table"]),
            WeightTarget(
                to_pattern="patch_embedding.weight",
                from_pattern=["patch_embedding.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(to_pattern="patch_embedding.bias", from_pattern=["patch_embedding.bias"]),
            WeightTarget(
                to_pattern="condition_embedder.time_embedder.linear_1.weight",
                from_pattern=["condition_embedder.time_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.time_embedder.linear_1.bias",
                from_pattern=["condition_embedder.time_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.time_embedder.linear_2.weight",
                from_pattern=["condition_embedder.time_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.time_embedder.linear_2.bias",
                from_pattern=["condition_embedder.time_embedder.linear_2.bias"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.time_proj.weight",
                from_pattern=["condition_embedder.time_proj.weight"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.time_proj.bias", from_pattern=["condition_embedder.time_proj.bias"]
            ),
            WeightTarget(
                to_pattern="condition_embedder.text_embedder.linear_1.weight",
                from_pattern=["condition_embedder.text_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.text_embedder.linear_1.bias",
                from_pattern=["condition_embedder.text_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.text_embedder.linear_2.weight",
                from_pattern=["condition_embedder.text_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="condition_embedder.text_embedder.linear_2.bias",
                from_pattern=["condition_embedder.text_embedder.linear_2.bias"],
            ),
            WeightTarget(to_pattern="proj_out.weight", from_pattern=["proj_out.weight"]),
            WeightTarget(to_pattern="proj_out.bias", from_pattern=["proj_out.bias"]),
        ]

        for layer in range(num_layers):
            prefix = f"blocks.{layer}"
            mapping.extend(
                [
                    WeightTarget(
                        to_pattern=f"{prefix}.scale_shift_table", from_pattern=[f"{prefix}.scale_shift_table"]
                    ),
                    *WanWeightMapping._attention_mapping(f"{prefix}.attn1"),
                    *WanWeightMapping._attention_mapping(f"{prefix}.attn2"),
                    WeightTarget(to_pattern=f"{prefix}.norm2.weight", from_pattern=[f"{prefix}.norm2.weight"]),
                    WeightTarget(to_pattern=f"{prefix}.norm2.bias", from_pattern=[f"{prefix}.norm2.bias"]),
                    WeightTarget(
                        to_pattern=f"{prefix}.ffn.net.0.weight",
                        from_pattern=[f"{prefix}.ffn.net.0.proj.weight"],
                    ),
                    WeightTarget(
                        to_pattern=f"{prefix}.ffn.net.0.bias",
                        from_pattern=[f"{prefix}.ffn.net.0.proj.bias"],
                    ),
                    WeightTarget(to_pattern=f"{prefix}.ffn.net.1.weight", from_pattern=[f"{prefix}.ffn.net.2.weight"]),
                    WeightTarget(to_pattern=f"{prefix}.ffn.net.1.bias", from_pattern=[f"{prefix}.ffn.net.2.bias"]),
                ]
            )
        return mapping

    @staticmethod
    def get_vae_mapping(variant: str = "wan22_ti2v") -> list[WeightTarget]:
        if variant == "wan21":
            return WanWeightMapping.get_wan21_vae_mapping()
        return WanWeightMapping.get_wan22_ti2v_vae_mapping()

    @staticmethod
    def get_wan22_ti2v_vae_mapping() -> list[WeightTarget]:
        mapping = [
            *WanWeightMapping._conv3d("quant_conv", "quant_conv"),
            *WanWeightMapping._conv3d("post_quant_conv", "post_quant_conv"),
            *WanWeightMapping._conv3d("encoder.conv_in", "encoder.conv_in"),
            *WanWeightMapping._conv3d("encoder.conv_out", "encoder.conv_out"),
            WeightTarget(to_pattern="encoder.norm_out.weight", from_pattern=["encoder.norm_out.gamma"]),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.norm.weight",
                from_pattern=["encoder.mid_block.attentions.0.norm.gamma"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_qkv.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_qkv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_qkv.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.proj.weight",
                from_pattern=["encoder.mid_block.attentions.0.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.proj.bias",
                from_pattern=["encoder.mid_block.attentions.0.proj.bias"],
            ),
            *WanWeightMapping._conv3d("decoder.conv_in", "decoder.conv_in"),
            *WanWeightMapping._conv3d("decoder.conv_out", "decoder.conv_out"),
            WeightTarget(to_pattern="decoder.norm_out.weight", from_pattern=["decoder.norm_out.gamma"]),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.norm.weight",
                from_pattern=["decoder.mid_block.attentions.0.norm.gamma"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_qkv.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_qkv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_qkv.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.proj.weight",
                from_pattern=["decoder.mid_block.attentions.0.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.proj.bias",
                from_pattern=["decoder.mid_block.attentions.0.proj.bias"],
            ),
        ]

        for resnet in range(2):
            mapping.extend(
                WanWeightMapping._resnet_mapping(
                    hf_prefix=f"encoder.mid_block.resnets.{resnet}",
                    to_prefix=f"encoder.mid_block.resnets.{resnet}",
                )
            )
            mapping.extend(
                WanWeightMapping._resnet_mapping(
                    hf_prefix=f"decoder.mid_block.resnets.{resnet}",
                    to_prefix=f"decoder.mid_block.resnets.{resnet}",
                )
            )

        for block in range(4):
            for resnet in range(2):
                prefix = f"encoder.down_blocks.{block}.resnets.{resnet}"
                mapping.extend(WanWeightMapping._resnet_mapping(hf_prefix=prefix, to_prefix=prefix))
            if block != 3:
                if block in (1, 2):
                    mapping.extend(
                        WanWeightMapping._conv3d(
                            hf_prefix=f"encoder.down_blocks.{block}.downsampler.time_conv",
                            to_prefix=f"encoder.down_blocks.{block}.downsampler.time_conv",
                        )
                    )
                mapping.extend(
                    WanWeightMapping._conv2d(
                        hf_prefix=f"encoder.down_blocks.{block}.downsampler.resample.1",
                        to_prefix=f"encoder.down_blocks.{block}.downsampler.resample_conv",
                    )
                )
            for resnet in range(3):
                prefix = f"decoder.up_blocks.{block}.resnets.{resnet}"
                mapping.extend(WanWeightMapping._resnet_mapping(hf_prefix=prefix, to_prefix=prefix))
            if block in (0, 1):
                mapping.extend(
                    WanWeightMapping._conv3d(
                        hf_prefix=f"decoder.up_blocks.{block}.upsampler.time_conv",
                        to_prefix=f"decoder.up_blocks.{block}.upsampler.time_conv",
                    )
                )
            mapping.extend(
                WanWeightMapping._conv2d(
                    hf_prefix=f"decoder.up_blocks.{block}.upsampler.resample.1",
                    to_prefix=f"decoder.up_blocks.{block}.upsampler.resample_conv",
                )
            )
        return mapping

    @staticmethod
    def get_wan21_vae_mapping() -> list[WeightTarget]:
        mapping = [
            *WanWeightMapping._conv3d("quant_conv", "quant_conv"),
            *WanWeightMapping._conv3d("post_quant_conv", "post_quant_conv"),
            *WanWeightMapping._conv3d("encoder.conv_in", "encoder.conv_in"),
            *WanWeightMapping._conv3d("encoder.conv_out", "encoder.conv_out"),
            WeightTarget(to_pattern="encoder.norm_out.weight", from_pattern=["encoder.norm_out.gamma"]),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.norm.weight",
                from_pattern=["encoder.mid_block.attentions.0.norm.gamma"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_qkv.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_qkv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_qkv.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.proj.weight",
                from_pattern=["encoder.mid_block.attentions.0.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.proj.bias",
                from_pattern=["encoder.mid_block.attentions.0.proj.bias"],
            ),
            *WanWeightMapping._conv3d("decoder.conv_in", "decoder.conv_in"),
            *WanWeightMapping._conv3d("decoder.conv_out", "decoder.conv_out"),
            WeightTarget(to_pattern="decoder.norm_out.weight", from_pattern=["decoder.norm_out.gamma"]),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.norm.weight",
                from_pattern=["decoder.mid_block.attentions.0.norm.gamma"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_qkv.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_qkv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_qkv.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_qkv.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.proj.weight",
                from_pattern=["decoder.mid_block.attentions.0.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.proj.bias",
                from_pattern=["decoder.mid_block.attentions.0.proj.bias"],
            ),
        ]
        for resnet in range(2):
            mapping.extend(
                WanWeightMapping._residual_block_mapping(
                    hf_prefix=f"encoder.mid_block.resnets.{resnet}",
                    to_prefix=f"encoder.mid_block.resnets.{resnet}",
                )
            )
            mapping.extend(
                WanWeightMapping._residual_block_mapping(
                    hf_prefix=f"decoder.mid_block.resnets.{resnet}",
                    to_prefix=f"decoder.mid_block.resnets.{resnet}",
                )
            )

        encoder_layer = 0
        for block in range(4):
            for _ in range(2):
                mapping.extend(
                    WanWeightMapping._residual_block_mapping(
                        hf_prefix=f"encoder.down_blocks.{encoder_layer}",
                        to_prefix=f"encoder.down_blocks.{encoder_layer}",
                    )
                )
                encoder_layer += 1
            if block != 3:
                mapping.extend(
                    WanWeightMapping._resample_mapping(
                        hf_prefix=f"encoder.down_blocks.{encoder_layer}",
                        to_prefix=f"encoder.down_blocks.{encoder_layer}",
                    )
                )
                encoder_layer += 1

        for block in range(4):
            for resnet in range(3):
                mapping.extend(
                    WanWeightMapping._residual_block_mapping(
                        hf_prefix=f"decoder.up_blocks.{block}.resnets.{resnet}",
                        to_prefix=f"decoder.up_blocks.{block}.resnets.{resnet}",
                    )
                )
            if block != 3:
                mapping.extend(
                    WanWeightMapping._resample_mapping(
                        hf_prefix=f"decoder.up_blocks.{block}.upsamplers.0",
                        to_prefix=f"decoder.up_blocks.{block}.upsamplers.0",
                    )
                )
        return mapping

    @staticmethod
    def _attention_mapping(prefix: str) -> list[WeightTarget]:
        return (
            [
                WeightTarget(to_pattern=f"{prefix}.{name}.weight", from_pattern=[f"{prefix}.{name}.weight"])
                for name in ("to_q", "to_k", "to_v")
            ]
            + [
                WeightTarget(to_pattern=f"{prefix}.{name}.bias", from_pattern=[f"{prefix}.{name}.bias"])
                for name in ("to_q", "to_k", "to_v")
            ]
            + [
                WeightTarget(to_pattern=f"{prefix}.to_out.0.weight", from_pattern=[f"{prefix}.to_out.0.weight"]),
                WeightTarget(to_pattern=f"{prefix}.to_out.0.bias", from_pattern=[f"{prefix}.to_out.0.bias"]),
                WeightTarget(to_pattern=f"{prefix}.norm_q.weight", from_pattern=[f"{prefix}.norm_q.weight"]),
                WeightTarget(to_pattern=f"{prefix}.norm_k.weight", from_pattern=[f"{prefix}.norm_k.weight"]),
            ]
        )

    @staticmethod
    def _resnet_mapping(hf_prefix: str, to_prefix: str) -> list[WeightTarget]:
        mapping = [
            WeightTarget(to_pattern=f"{to_prefix}.norm1.weight", from_pattern=[f"{hf_prefix}.norm1.gamma"]),
            WeightTarget(to_pattern=f"{to_prefix}.norm2.weight", from_pattern=[f"{hf_prefix}.norm2.gamma"]),
            *WanWeightMapping._conv3d(f"{hf_prefix}.conv1", f"{to_prefix}.conv1"),
            *WanWeightMapping._conv3d(f"{hf_prefix}.conv2", f"{to_prefix}.conv2"),
        ]
        needs_shortcut = ".resnets.0" in hf_prefix and (
            "encoder.down_blocks.1" in hf_prefix
            or "encoder.down_blocks.2" in hf_prefix
            or "decoder.up_blocks.2" in hf_prefix
            or "decoder.up_blocks.3" in hf_prefix
        )
        if needs_shortcut:
            mapping.extend(WanWeightMapping._conv3d(f"{hf_prefix}.conv_shortcut", f"{to_prefix}.conv_shortcut"))
        return mapping

    @staticmethod
    def _residual_block_mapping(hf_prefix: str, to_prefix: str) -> list[WeightTarget]:
        return [
            WeightTarget(to_pattern=f"{to_prefix}.norm1.weight", from_pattern=[f"{hf_prefix}.norm1.gamma"]),
            WeightTarget(to_pattern=f"{to_prefix}.norm2.weight", from_pattern=[f"{hf_prefix}.norm2.gamma"]),
            *WanWeightMapping._conv3d(f"{hf_prefix}.conv1", f"{to_prefix}.conv1"),
            *WanWeightMapping._conv3d(f"{hf_prefix}.conv2", f"{to_prefix}.conv2"),
            *WanWeightMapping._conv3d(f"{hf_prefix}.conv_shortcut", f"{to_prefix}.conv_shortcut"),
        ]

    @staticmethod
    def _resample_mapping(hf_prefix: str, to_prefix: str) -> list[WeightTarget]:
        return [
            *WanWeightMapping._conv2d(f"{hf_prefix}.resample.1", f"{to_prefix}.resample_conv"),
            *WanWeightMapping._conv3d(f"{hf_prefix}.time_conv", f"{to_prefix}.time_conv"),
        ]

    @staticmethod
    def _conv3d(hf_prefix: str, to_prefix: str) -> list[WeightTarget]:
        return [
            WeightTarget(
                to_pattern=f"{to_prefix}.conv3d.weight",
                from_pattern=[f"{hf_prefix}.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(to_pattern=f"{to_prefix}.conv3d.bias", from_pattern=[f"{hf_prefix}.bias"]),
        ]

    @staticmethod
    def _conv2d(hf_prefix: str, to_prefix: str) -> list[WeightTarget]:
        return [
            WeightTarget(
                to_pattern=f"{to_prefix}.weight",
                from_pattern=[f"{hf_prefix}.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(to_pattern=f"{to_prefix}.bias", from_pattern=[f"{hf_prefix}.bias"]),
        ]
