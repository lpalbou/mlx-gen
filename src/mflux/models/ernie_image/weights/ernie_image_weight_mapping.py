from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class ErnieImageWeightMapping(WeightMapping):
    @staticmethod
    def get_vae_mapping() -> list[WeightTarget]:
        return Flux2WeightMapping.get_vae_mapping()

    @staticmethod
    def get_transformer_mapping() -> list[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="x_embedder.proj.weight",
                from_pattern=["x_embedder.proj.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(to_pattern="x_embedder.proj.bias", from_pattern=["x_embedder.proj.bias"]),
            WeightTarget(to_pattern="text_proj.weight", from_pattern=["text_proj.weight"]),
            WeightTarget(
                to_pattern="time_embedding.linear_1.weight",
                from_pattern=["time_embedding.linear_1.weight"],
            ),
            WeightTarget(to_pattern="time_embedding.linear_1.bias", from_pattern=["time_embedding.linear_1.bias"]),
            WeightTarget(
                to_pattern="time_embedding.linear_2.weight",
                from_pattern=["time_embedding.linear_2.weight"],
            ),
            WeightTarget(to_pattern="time_embedding.linear_2.bias", from_pattern=["time_embedding.linear_2.bias"]),
            WeightTarget(to_pattern="adaLN_modulation.linear.weight", from_pattern=["adaLN_modulation.1.weight"]),
            WeightTarget(to_pattern="adaLN_modulation.linear.bias", from_pattern=["adaLN_modulation.1.bias"]),
            WeightTarget(to_pattern="final_norm.linear.weight", from_pattern=["final_norm.linear.weight"]),
            WeightTarget(to_pattern="final_norm.linear.bias", from_pattern=["final_norm.linear.bias"]),
            WeightTarget(to_pattern="final_linear.weight", from_pattern=["final_linear.weight"]),
            WeightTarget(to_pattern="final_linear.bias", from_pattern=["final_linear.bias"]),
            WeightTarget(to_pattern="layers.{layer}.adaLN_sa_ln.weight", from_pattern=["layers.{layer}.adaLN_sa_ln.weight"]),
            WeightTarget(
                to_pattern="layers.{layer}.adaLN_mlp_ln.weight",
                from_pattern=["layers.{layer}.adaLN_mlp_ln.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_q.weight",
                from_pattern=["layers.{layer}.self_attention.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_k.weight",
                from_pattern=["layers.{layer}.self_attention.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_v.weight",
                from_pattern=["layers.{layer}.self_attention.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.to_out.0.weight",
                from_pattern=["layers.{layer}.self_attention.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.norm_q.weight",
                from_pattern=["layers.{layer}.self_attention.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attention.norm_k.weight",
                from_pattern=["layers.{layer}.self_attention.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.gate_proj.weight",
                from_pattern=["layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(to_pattern="layers.{layer}.mlp.up_proj.weight", from_pattern=["layers.{layer}.mlp.up_proj.weight"]),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.linear_fc2.weight",
                from_pattern=["layers.{layer}.mlp.linear_fc2.weight"],
            ),
        ]

    @staticmethod
    def get_text_encoder_mapping() -> list[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=["language_model.model.embed_tokens.weight"],
            ),
            WeightTarget(
                to_pattern="norm.weight",
                from_pattern=["language_model.model.norm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.input_layernorm.weight",
                from_pattern=["language_model.model.layers.{layer}.input_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.post_attention_layernorm.weight",
                from_pattern=["language_model.model.layers.{layer}.post_attention_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.q_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.k_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.v_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.o_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.self_attn.o_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.gate_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.up_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.down_proj.weight",
                from_pattern=["language_model.model.layers.{layer}.mlp.down_proj.weight"],
            ),
        ]
