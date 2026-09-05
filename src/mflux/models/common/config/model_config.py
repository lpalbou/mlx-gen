from functools import lru_cache

import mlx.core as mx

from mflux.models.common.resolution.config_resolution import ConfigResolution

WAN_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# The seven architecture-shape keys of the Wan 2.2 TI2V-5B transformer. SwiftVR is a
# fine-tune of exactly this model - all 825 checkpoint tensors match stock Wan in name,
# shape and dtype - so both catalog entries read the numbers from here rather than
# restating them and risking a silent divergence.
WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE = {
    "in_channels": 48,
    "out_channels": 48,
    "num_layers": 30,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "ffn_dim": 14336,
    "patch_size": [1, 2, 2],
}

# Shared by the ByteDance source entry and the AbstractFramework BF16 repack
# entry; per-entry keys (component sources, revision pins, download byte
# expectations) are layered on top by each catalog entry.
BERNINI_R_1_3B_SHARED_TRANSFORMER_OVERRIDES = {
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 30,
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "ffn_dim": 8960,
    "patch_size": [1, 2, 2],
    "expand_timesteps": False,
    "has_transformer_2": False,
    "boundary_ratio": None,
    # The pinned checkpoint records shift=3, but the official renderer
    # CLI overrides it to 5 for every published inference recipe.
    "flow_shift": 5.0,
    "unipc_flow_sigma_schedule": "diffusers-0.35.2",
    "vae_variant": "wan21",
    "vae_config": {
        "base_dim": 96,
        "decoder_base_dim": None,
        "z_dim": 16,
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": 1,
        "scale_factor_spatial": 8,
        "scale_factor_temporal": 4,
        "is_residual": False,
    },
    "task": "reference-to-video",
    "supports_image_to_video": False,
    "supports_video_to_video": True,
    "supports_bernini_renderer": True,
    "use_source_id_rotary_embedding": True,
    "interpolate_source_ids": True,
    "max_trained_source_id": 5,
    "max_reference_images": 8,
    "max_condition_size": 848,
    "max_condition_size_limit": 1280,
    "expected_renderer_config": {
        "model_type": "bernini_renderer",
        "skip_transformer_1": False,
        "skip_transformer_2": True,
        "max_sequence_length": 512,
        "shift": 3.0,
        "use_unipc": True,
        "use_src_id_rotary_emb": True,
    },
    "download_headroom_bytes": 2 * 1024**3,
    "default_width": 848,
    "default_height": 480,
    "default_frames": 81,
    "default_steps": 40,
    "default_fps": 16,
    "default_guidance": 4.0,
    "default_reference_guidance": 4.5,
    "default_source_guidance": 1.25,
    "default_apg_eta": 0.5,
    "default_apg_norm_threshold": 50.0,
    "default_apg_momentum": 0.0,
    "default_negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
    "default_solver": "unipc",
}

BERNINI_R_1_3B_TEXT_ENCODER_OVERRIDES = {
    "model_type": "umt5",
    "d_model": 4096,
    "d_ff": 10240,
    "num_layers": 24,
    "num_heads": 64,
    "vocab_size": 256384,
}


class ModelConfig:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
        self,
        priority: int,
        aliases: list[str],
        model_name: str,
        base_model: str | None,
        controlnet_model: str | None,
        custom_transformer_model: str | None,
        num_train_steps: int | None,
        max_sequence_length: int | None,
        supports_guidance: bool | None,
        requires_sigma_shift: bool | None,
        transformer_overrides: dict | None = None,
        text_encoder_overrides: dict | None = None,
        inference_aliases: list[str] | None = None,
        sigma_base_shift: float = 0.5,
        sigma_max_shift: float = 1.15,
        sigma_base_seq_len: int = 256,
        sigma_max_seq_len: int = 4096,
        sigma_shift_terminal: float | None = None,
    ):
        self.aliases = aliases
        self.model_name = model_name
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.custom_transformer_model = custom_transformer_model
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.supports_guidance = supports_guidance
        self.requires_sigma_shift = requires_sigma_shift
        self.priority = priority
        self.transformer_overrides = transformer_overrides or {}
        self.text_encoder_overrides = text_encoder_overrides or {}
        self.inference_aliases = aliases if inference_aliases is None else inference_aliases
        self.sigma_base_shift = sigma_base_shift
        self.sigma_max_shift = sigma_max_shift
        self.sigma_base_seq_len = sigma_base_seq_len
        self.sigma_max_seq_len = sigma_max_seq_len
        self.sigma_shift_terminal = sigma_shift_terminal

    @staticmethod
    @lru_cache
    def dev() -> "ModelConfig":
        return AVAILABLE_MODELS["dev"]

    @staticmethod
    @lru_cache
    def schnell() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell"]

    @staticmethod
    @lru_cache
    def dev_kontext() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-kontext"]

    @staticmethod
    @lru_cache
    def dev_fill() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill"]

    @staticmethod
    @lru_cache
    def dev_redux() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-redux"]

    @staticmethod
    @lru_cache
    def dev_depth() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-depth"]

    @staticmethod
    @lru_cache
    def dev_controlnet_canny() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-controlnet-canny"]

    @staticmethod
    @lru_cache
    def schnell_controlnet_canny() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell-controlnet-canny"]

    @staticmethod
    @lru_cache
    def dev_controlnet_upscaler() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-controlnet-upscaler"]

    @staticmethod
    @lru_cache
    def dev_fill_catvton() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill-catvton"]

    @staticmethod
    @lru_cache
    def krea_dev() -> "ModelConfig":
        return AVAILABLE_MODELS["krea-dev"]

    @staticmethod
    @lru_cache
    def flux2_klein_4b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-4b"]

    @staticmethod
    @lru_cache
    def flux2_klein_9b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-9b"]

    @staticmethod
    @lru_cache
    def flux2_klein_base_4b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-base-4b"]

    @staticmethod
    @lru_cache
    def flux2_klein_base_9b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-base-9b"]

    @staticmethod
    @lru_cache
    def bonsai_image_ternary() -> "ModelConfig":
        return AVAILABLE_MODELS["bonsai-image-ternary"]

    @staticmethod
    @lru_cache
    def bonsai_image_binary() -> "ModelConfig":
        return AVAILABLE_MODELS["bonsai-image-binary"]

    @staticmethod
    @lru_cache
    def qwen_image() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image"]

    @staticmethod
    @lru_cache
    def qwen_image_edit() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image-edit"]

    @staticmethod
    @lru_cache
    def qwen_image_edit_2509() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image-edit-2509"]

    @staticmethod
    @lru_cache
    def fibo() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo"]

    @staticmethod
    @lru_cache
    def fibo_lite() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo-lite"]

    @staticmethod
    @lru_cache
    def fibo_edit() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo-edit"]

    @staticmethod
    @lru_cache
    def fibo_edit_rmbg() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo-edit-rmbg"]

    @staticmethod
    @lru_cache
    def z_image_turbo() -> "ModelConfig":
        return AVAILABLE_MODELS["z-image-turbo"]

    @staticmethod
    @lru_cache
    def z_image() -> "ModelConfig":
        return AVAILABLE_MODELS["z-image"]

    @staticmethod
    @lru_cache
    def ernie_image_turbo() -> "ModelConfig":
        return AVAILABLE_MODELS["ernie-image-turbo"]

    @staticmethod
    @lru_cache
    def seedvr2_3b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-3b"]

    @staticmethod
    @lru_cache
    def seedvr2_7b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-7b"]

    @staticmethod
    @lru_cache
    def seedvr2_7b_sharp() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-7b-sharp"]

    @staticmethod
    @lru_cache
    def wan2_2_ti2v_5b() -> "ModelConfig":
        return AVAILABLE_MODELS["wan2.2-ti2v-5b"]

    @staticmethod
    @lru_cache
    def wan2_2_t2v_a14b() -> "ModelConfig":
        return AVAILABLE_MODELS["wan2.2-t2v-a14b"]

    @staticmethod
    @lru_cache
    def minimax_h3() -> "ModelConfig":
        return AVAILABLE_MODELS["minimax-h3"]

    @staticmethod
    @lru_cache
    def minimax_h3_turbo() -> "ModelConfig":
        return AVAILABLE_MODELS["minimax-h3-turbo"]

    @staticmethod
    @lru_cache
    def minimax_h3_turbo_544p() -> "ModelConfig":
        return AVAILABLE_MODELS["minimax-h3-turbo-544p"]

    @staticmethod
    @lru_cache
    def wan2_2_i2v_a14b() -> "ModelConfig":
        return AVAILABLE_MODELS["wan2.2-i2v-a14b"]

    @staticmethod
    @lru_cache
    def bernini_r_1_3b() -> "ModelConfig":
        return AVAILABLE_MODELS["bernini-r-1.3b"]

    @staticmethod
    @lru_cache
    def swiftvr() -> "ModelConfig":
        return AVAILABLE_MODELS["swiftvr"]

    def x_embedder_input_dim(self) -> int:
        if "Fill" in self.model_name:
            return 384
        if "Depth" in self.model_name:
            return 128
        else:
            return 64

    def is_canny(self) -> bool:
        return self.controlnet_model is not None and "Canny" in self.controlnet_model

    @staticmethod
    def from_name(
        model_name: str,
        base_model: str | None = None,
    ) -> "ModelConfig":
        return ConfigResolution.resolve(model_name=model_name, base_model=base_model)


AVAILABLE_MODELS = {
    "dev": ModelConfig(
        priority=0,
        aliases=["dev"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=3000,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "schnell": ModelConfig(
        priority=1,
        aliases=["schnell"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-kontext": ModelConfig(
        priority=2,
        aliases=["dev-kontext"],
        model_name="black-forest-labs/FLUX.1-Kontext-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=3000,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-fill": ModelConfig(
        priority=3,
        aliases=["dev-fill"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=3000,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-redux": ModelConfig(
        priority=4,
        aliases=["dev-redux"],
        model_name="black-forest-labs/FLUX.1-Redux-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=3000,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-depth": ModelConfig(
        priority=5,
        aliases=["dev-depth"],
        model_name="black-forest-labs/FLUX.1-Depth-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-controlnet-canny": ModelConfig(
        priority=6,
        aliases=["dev-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "schnell-controlnet-canny": ModelConfig(
        priority=7,
        aliases=["schnell-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-controlnet-upscaler": ModelConfig(
        priority=8,
        aliases=["dev-controlnet-upscaler"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="jasperai/Flux.1-dev-Controlnet-Upscaler",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-fill-catvton": ModelConfig(
        priority=9,
        aliases=["dev-fill-catvton"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model="xiaozaa/catvton-flux-beta",
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "krea-dev": ModelConfig(
        priority=10,
        aliases=["krea-dev", "dev-krea"],
        model_name="black-forest-labs/FLUX.1-Krea-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "flux2-klein-4b": ModelConfig(
        priority=11,
        aliases=["flux2-klein-4b", "flux2-klein-4B", "flux2-klein", "klein-4b", "klein-4B"],
        model_name="black-forest-labs/FLUX.2-klein-4B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
        text_encoder_overrides={
            "hidden_size": 2560,
            "intermediate_size": 9728,
        },
    ),
    "flux2-klein-9b": ModelConfig(
        priority=12,
        aliases=["flux2-klein-9b", "flux2-klein-9B", "klein-9b", "klein-9B"],
        model_name="black-forest-labs/FLUX.2-klein-9B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        },
        text_encoder_overrides={
            "hidden_size": 4096,
            "intermediate_size": 12288,
        },
    ),
    "flux2-klein-base-4b": ModelConfig(
        priority=13,
        aliases=[
            "flux2-klein-base-4b",
            "flux2-klein-base-4B",
            "flux2-base-4b",
            "flux2-base-4B",
            "klein-base-4b",
            "klein-base-4B",
        ],
        model_name="black-forest-labs/FLUX.2-klein-base-4B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
        text_encoder_overrides={
            "hidden_size": 2560,
            "intermediate_size": 9728,
        },
    ),
    "flux2-klein-base-9b": ModelConfig(
        priority=14,
        aliases=[
            "flux2-klein-base-9b",
            "flux2-klein-base-9B",
            "flux2-base-9b",
            "flux2-base-9B",
            "klein-base-9b",
            "klein-base-9B",
        ],
        model_name="black-forest-labs/FLUX.2-klein-base-9B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        },
        text_encoder_overrides={
            "hidden_size": 4096,
            "intermediate_size": 12288,
        },
    ),
    "bonsai-image-ternary": ModelConfig(
        priority=15,
        aliases=[
            "bonsai",
            "bonsai-image",
            "bonsai-image-ternary",
            "bonsai-image-2bit",
            "bonsai-ternary",
            "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
            "prism-ml/bonsai-image-ternary-4b-mlx-2bit",
        ],
        model_name="prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,
        requires_sigma_shift=False,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
        text_encoder_overrides={
            "hidden_size": 2560,
            "intermediate_size": 9728,
        },
    ),
    "bonsai-image-binary": ModelConfig(
        priority=16,
        aliases=[
            "bonsai-image-binary",
            "bonsai-image-1bit",
            "bonsai-binary",
            "prism-ml/bonsai-image-binary-4B-mlx-1bit",
            "prism-ml/bonsai-image-binary-4b-mlx-1bit",
        ],
        model_name="prism-ml/bonsai-image-binary-4B-mlx-1bit",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,
        requires_sigma_shift=False,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
        text_encoder_overrides={
            "hidden_size": 2560,
            "intermediate_size": 9728,
        },
    ),
    "qwen-image": ModelConfig(
        priority=17,
        aliases=["qwen-image", "qwen"],
        model_name="Qwen/Qwen-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
    ),
    "qwen-image-edit": ModelConfig(
        priority=16,
        aliases=["qwen-image-edit", "qwen-edit"],
        model_name="Qwen/Qwen-Image-Edit",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
    ),
    "qwen-image-edit-2509": ModelConfig(
        priority=17,
        aliases=["qwen-image-edit-2509", "qwen-edit-2509", "qwen-edit-plus", "qwen-edit-plus-2509"],
        model_name="Qwen/Qwen-Image-Edit-2509",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
        transformer_overrides={
            "qwen_edit_plus": True,
        },
    ),
    "qwen-image-edit-2511": ModelConfig(
        priority=18,
        aliases=["qwen-image-edit-2511", "qwen-edit-2511"],
        model_name="Qwen/Qwen-Image-Edit-2511",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
        transformer_overrides={
            "qwen_edit_plus": True,
            "zero_cond_t": True,
        },
    ),
    "fibo": ModelConfig(
        priority=17,
        aliases=["fibo"],
        model_name="briaai/FIBO",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "fibo-lite": ModelConfig(
        priority=18,
        aliases=["fibo-lite", "fibo_lite"],
        model_name="briaai/Fibo-lite",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "fibo-edit": ModelConfig(
        priority=19,
        aliases=["fibo-edit", "fiboedit"],
        model_name="briaai/Fibo-Edit",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "fibo-edit-rmbg": ModelConfig(
        priority=24,
        aliases=["fibo-edit-rmbg", "fiboedit-rmbg"],
        model_name="briaai/Fibo-Edit-RMBG",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "z-image": ModelConfig(
        priority=20,
        aliases=["z-image", "zimage"],
        model_name="Tongyi-MAI/Z-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "z-image-turbo": ModelConfig(
        priority=21,
        aliases=["z-image-turbo", "zimage-turbo"],
        model_name="Tongyi-MAI/Z-Image-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,  # Turbo model uses guidance_scale=0
        requires_sigma_shift=True,
    ),
    "ernie-image-turbo": ModelConfig(
        priority=22,
        aliases=["ernie-image-turbo", "ernie-image", "ernie"],
        model_name="baidu/ERNIE-Image-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "seedvr2-3b": ModelConfig(
        priority=23,
        aliases=["seedvr2-3b", "seedvr2"],
        model_name="ByteDance-Seed/SeedVR2-3B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
        transformer_overrides={
            "rope_freqs_for": "lang",
            "text_rope_freqs_for": "lang",
        },
    ),
    "seedvr2-7b": ModelConfig(
        priority=24,
        aliases=["seedvr2-7b", "seedvr2-7B"],
        model_name="ByteDance-Seed/SeedVR2-7B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
        transformer_overrides={
            "vid_dim": 3072,
            "heads": 24,
            "num_layers": 36,
            "mm_layers": 36,
            "rope_dim": 64,
            "rope_on_text": False,
            "rope_freqs_for": "pixel",
            "text_attention_mode": "global_text",
            "mlp_type": "normal",
            "use_output_ada": False,
            "last_layer_vid_only": False,
        },
    ),
    "seedvr2-7b-sharp": ModelConfig(
        priority=25,
        aliases=["seedvr2-7b-sharp", "seedvr2-7b-sharp-fp16"],
        model_name="ByteDance-Seed/SeedVR2-7B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
        transformer_overrides={
            "vid_dim": 3072,
            "heads": 24,
            "num_layers": 36,
            "mm_layers": 36,
            "rope_dim": 64,
            "rope_on_text": False,
            "rope_freqs_for": "pixel",
            "text_attention_mode": "global_text",
            "mlp_type": "normal",
            "use_output_ada": False,
            "last_layer_vid_only": False,
        },
    ),
    "minimax-h3": ModelConfig(
        priority=30,
        aliases=["minimax-h3", "minimaxai/minimax-h3", "h3"],
        model_name="MiniMaxAI/MiniMax-H3",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=8192,
        supports_guidance=False,
        requires_sigma_shift=False,
        inference_aliases=["minimax-h3"],
        transformer_overrides={
            "task": "text-to-video",
            "generates_audio": True,
            "supports_image_to_video": False,
            "default_frames": 124,
            "default_steps": 50,
            "default_fps": 24,
            "default_width": 1344,
            "default_height": 768,
            "default_video_shift": 12.0,
            "default_audio_shift": 3.0,
        },
        text_encoder_overrides={"model_type": "qwen3_vl", "hidden_size": 5120, "num_layers": 50},
    ),
    "minimax-h3-turbo": ModelConfig(
        priority=31,
        aliases=["minimax-h3-turbo", "h3-turbo"],
        model_name="MiniMaxAI/MiniMax-H3",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=8192,
        supports_guidance=False,
        requires_sigma_shift=False,
        inference_aliases=["minimax-h3-turbo"],
        transformer_overrides={
            "task": "text-to-video",
            "generates_audio": True,
            "supports_image_to_video": False,
            "default_frames": 124,
            # lightx2v FL2VA Turbo 8-step v1.0, trained at 768p (1344x768) with shifts 6 / 3.
            "default_steps": 8,
            "default_fps": 24,
            "default_width": 1344,
            "default_height": 768,
            "default_video_shift": 6.0,
            "default_audio_shift": 3.0,
            "turbo_lora": "hf:lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
        },
        text_encoder_overrides={"model_type": "qwen3_vl", "hidden_size": 5120, "num_layers": 50},
    ),
    "minimax-h3-turbo-544p": ModelConfig(
        priority=32,
        aliases=["minimax-h3-turbo-544p", "h3-turbo-544p"],
        model_name="MiniMaxAI/MiniMax-H3",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=8192,
        supports_guidance=False,
        requires_sigma_shift=False,
        inference_aliases=["minimax-h3-turbo-544p"],
        transformer_overrides={
            "task": "text-to-video",
            "generates_audio": True,
            "supports_image_to_video": False,
            "default_frames": 124,
            # lightx2v FL2VA Turbo 8-step v1.0, trained at 544p on mixed aspect ratios with the base shifts 12 / 3.
            "default_steps": 8,
            "default_fps": 24,
            "default_width": 960,
            "default_height": 544,
            "default_video_shift": 12.0,
            "default_audio_shift": 3.0,
            "turbo_lora": "hf:lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        },
        text_encoder_overrides={"model_type": "qwen3_vl", "hidden_size": 5120, "num_layers": 50},
    ),
    "wan2.2-ti2v-5b": ModelConfig(
        priority=26,
        aliases=[
            "wan2.2-ti2v-5b",
            "wan2-2-ti2v-5b",
            "wan-ti2v",
        ],
        model_name="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        transformer_overrides={
            **WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE,
            "expand_timesteps": True,
            "has_transformer_2": False,
            "boundary_ratio": None,
            "flow_shift": 5.0,
            "vae_variant": "wan22_ti2v",
            "vae_config": {
                "base_dim": 160,
                "decoder_base_dim": 256,
                "z_dim": 48,
                "in_channels": 12,
                "out_channels": 12,
                "patch_size": 2,
                "scale_factor_spatial": 16,
                "scale_factor_temporal": 4,
                "is_residual": True,
            },
            "task": "text-image-to-video",
            "supports_image_to_video": True,
            "default_width": 1280,
            "default_height": 704,
            "default_frames": 81,
            "default_steps": 50,
            "default_fps": 24,
            "default_guidance": 5.0,
            "default_negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
            "default_solver": "unipc",
        },
        text_encoder_overrides={
            "model_type": "umt5",
            "d_model": 4096,
            "d_ff": 10240,
            "num_layers": 24,
            "num_heads": 64,
            "vocab_size": 256384,
        },
    ),
    "wan2.2-t2v-a14b": ModelConfig(
        priority=26,
        aliases=[
            "wan2.2-t2v-a14b",
            "wan2-2-t2v-a14b",
            "wan-t2v-a14b",
            "wan-a14b-t2v",
        ],
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        transformer_overrides={
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 40,
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "ffn_dim": 13824,
            "patch_size": [1, 2, 2],
            "expand_timesteps": False,
            "has_transformer_2": True,
            "boundary_ratio": 0.875,
            "flow_shift": 3.0,
            "vae_variant": "wan21",
            "vae_config": {
                "base_dim": 96,
                "decoder_base_dim": None,
                "z_dim": 16,
                "in_channels": 3,
                "out_channels": 3,
                "patch_size": 1,
                "scale_factor_spatial": 8,
                "scale_factor_temporal": 4,
                "is_residual": False,
            },
            "task": "text-to-video",
            "supports_image_to_video": False,
            "supports_video_to_video": True,
            "default_width": 1280,
            "default_height": 720,
            "default_frames": 81,
            "default_steps": 40,
            "default_fps": 16,
            "default_guidance": 4.0,
            "default_guidance_2": 3.0,
            "default_negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
            "default_solver": "unipc",
        },
        text_encoder_overrides={
            "model_type": "umt5",
            "d_model": 4096,
            "d_ff": 10240,
            "num_layers": 24,
            "num_heads": 64,
            "vocab_size": 256384,
        },
    ),
    "wan2.1-vace-1.3b": ModelConfig(
        priority=28,
        aliases=[
            "wan2.1-vace-1.3b",
            "wan2-1-vace-1-3b",
            "wan-vace-1.3b",
            "wan-vace",
            "vace-1.3b",
        ],
        model_name="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        transformer_overrides={
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 30,
            "num_attention_heads": 12,
            "attention_head_dim": 128,
            "ffn_dim": 8960,
            "patch_size": [1, 2, 2],
            "expand_timesteps": False,
            "has_transformer_2": False,
            "boundary_ratio": None,
            "flow_shift": 3.0,
            "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
            "vace_in_channels": 96,
            "vae_variant": "wan21",
            "vae_config": {
                "base_dim": 96,
                "decoder_base_dim": None,
                "z_dim": 16,
                "in_channels": 3,
                "out_channels": 3,
                "patch_size": 1,
                "scale_factor_spatial": 8,
                "scale_factor_temporal": 4,
                "is_residual": False,
            },
            # VACE genuinely supports pure text-to-video (zeros conditioning); the planner builds
            # its video-video capability from supports_video_to_video below.
            "task": "text-to-video",
            "supports_image_to_video": False,
            "supports_video_to_video": True,
            "supports_vace": True,
            "default_width": 832,
            "default_height": 480,
            "default_frames": 81,
            "default_steps": 30,
            "default_fps": 16,
            "default_guidance": 5.0,
            "default_negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
            "default_solver": "unipc",
        },
        text_encoder_overrides={
            "model_type": "umt5",
            "d_model": 4096,
            "d_ff": 10240,
            "num_layers": 24,
            "num_heads": 64,
            "vocab_size": 256384,
        },
    ),
    "bernini-r-1.3b": ModelConfig(
        priority=29,
        aliases=[
            "bernini-r-1.3b",
            "bernini-r-1-3b",
            "wan-bernini-r-1.3b",
            "wan-bernini",
            "bernini-r",
        ],
        inference_aliases=[],
        model_name="ByteDance/Bernini-R-1.3B-Diffusers",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model="ByteDance/Bernini-R-1.3B-Diffusers",
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        transformer_overrides={
            **BERNINI_R_1_3B_SHARED_TRANSFORMER_OVERRIDES,
            # Exact public-example parity requires the official Bernini text
            # encoder and tokenizer. The VAE is byte-identical to Wan VACE, but
            # the public Bernini text encoder weights are not.
            "component_base_model": "ByteDance/Bernini-R-1.3B-Diffusers",
            "expected_component_base_revision": "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce",
            "expected_transformer_revision": "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce",
            "expected_component_base_download_bytes": 23_252_742_299,
            "expected_transformer_download_bytes": 5_676_148_056,
        },
        text_encoder_overrides=dict(BERNINI_R_1_3B_TEXT_ENCODER_OVERRIDES),
    ),
    # BF16 repack of the pinned ByteDance/Bernini-R-1.3B-Diffusers revision:
    # identical runtime dtypes (the loader casts the FP32 original to exactly
    # these), ~13.7 GiB download instead of ~27 GiB. Published and maintained
    # under the AbstractFramework namespace.
    "bernini-r-1.3b-bf16": ModelConfig(
        priority=30,
        aliases=[
            "bernini-r-1.3b-bf16",
            "bernini-r-1-3b-bf16",
            "bernini-bf16",
        ],
        inference_aliases=["bernini-r-1.3b-diffusers-bf16"],
        model_name="AbstractFramework/bernini-r-1.3B-diffusers-bf16",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model="AbstractFramework/bernini-r-1.3B-diffusers-bf16",
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        transformer_overrides={
            **BERNINI_R_1_3B_SHARED_TRANSFORMER_OVERRIDES,
            "component_base_model": "AbstractFramework/bernini-r-1.3B-diffusers-bf16",
            "expected_component_base_revision": "09e82f824f708b43af142b398e1acd4068f12596",
            "expected_transformer_revision": "09e82f824f708b43af142b398e1acd4068f12596",
            "expected_component_base_download_bytes": 13_904_188_569,
            "expected_transformer_download_bytes": 2_844_405_640,
        },
        text_encoder_overrides=dict(BERNINI_R_1_3B_TEXT_ENCODER_OVERRIDES),
    ),
    "wan2.2-i2v-a14b": ModelConfig(
        priority=27,
        aliases=[
            "wan2.2-i2v-a14b",
            "wan2-2-i2v-a14b",
            "wan-i2v-a14b",
            "wan-a14b-i2v",
        ],
        model_name="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        transformer_overrides={
            "in_channels": 36,
            "out_channels": 16,
            "num_layers": 40,
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "ffn_dim": 13824,
            "patch_size": [1, 2, 2],
            "expand_timesteps": False,
            "has_transformer_2": True,
            "boundary_ratio": 0.9,
            "flow_shift": 3.0,
            "vae_variant": "wan21",
            "vae_config": {
                "base_dim": 96,
                "decoder_base_dim": None,
                "z_dim": 16,
                "in_channels": 3,
                "out_channels": 3,
                "patch_size": 1,
                "scale_factor_spatial": 8,
                "scale_factor_temporal": 4,
                "is_residual": False,
            },
            "task": "image-to-video",
            "supports_image_to_video": True,
            "default_width": 1280,
            "default_height": 720,
            "default_frames": 81,
            "default_steps": 40,
            "default_fps": 16,
            "default_guidance": 3.5,
            "default_guidance_2": 3.5,
            "default_negative_prompt": WAN_DEFAULT_NEGATIVE_PROMPT,
            "default_solver": "unipc",
        },
        text_encoder_overrides={
            "model_type": "umt5",
            "d_model": 4096,
            "d_ff": 10240,
            "num_layers": 24,
            "num_heads": 64,
            "vocab_size": 256384,
        },
    ),
    # SwiftVR one-step video restoration. The transformer is a fine-tune of Wan 2.2
    # TI2V-5B and is tensor-identical to it, so the shape keys come from the shared
    # constant above; everything Wan-specific that SwiftVR does not have - the 3D VAE,
    # the flow-matching sampler, guidance, the negative prompt, image-to-video - is
    # deliberately absent rather than inherited. base_model stays None: three separate
    # string-matching surfaces (prepare backend selection, download patterns, family
    # inference) build a token key from aliases + model_name + base_model, and naming
    # Wan there would silently reroute SwiftVR into the Wan family.
    "swiftvr": ModelConfig(
        priority=31,
        aliases=[
            "swiftvr",
            "swiftvr-5b",
        ],
        model_name="H-oliday/SwiftVR",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        # One forward pass at a constant timestep: no sampler, no CFG, no negative prompt.
        supports_guidance=False,
        requires_sigma_shift=False,
        transformer_overrides={
            **WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE,
            "text_dim": 4096,
            "freq_dim": 256,
            "rope_max_seq_len": 1024,
            "cross_attn_norm": True,
            "eps": 1e-06,
            "added_kv_proj_dim": None,
            # Mask-free shifted-window self-attention. The window size is a SwiftVR code
            # default, not checkpoint metadata: transformer/config.json carries no
            # enable_swa or self_attn_window_hw key, so it is pinned here to stay auditable.
            "swiftvr_window_size": [16, 16],
            "swiftvr_shift_alternate_layers": True,
            "swiftvr_inference_timestep": 1000.0,
            # ReAE replaces the Wan 3D VAE entirely: 40.95M parameters against 704.69M,
            # reproducing the same 48-channel, 16x spatial, 4x temporal latent contract.
            "reae_config": {
                "patch_size": 2,
                "latent_channels": 48,
                "width_mult": 2,
                "decoder_time_upscale": [True, True],
                "decoder_space_upscale": [True, True, True],
            },
            # Run defaults the CLI and SwiftVR.restore_video_to_path read through
            # SwiftVRInitializer.runtime_settings; there is no code-side copy of either.
            # The padded-canvas multiple is deliberately NOT here: it is a structural
            # consequence of the ReAE and patch-embed geometry, fixed by the weights, and
            # lives once in SwiftVRUtil.SPATIAL_PAD_MULTIPLE.
            "default_clip_len": 24,
            "default_dit_overlap": 0,
            "task": "video-to-video",
            "supports_video_to_video": True,
            "supports_image_to_video": False,
            "expected_download_bytes": 20_167_236_128,
            "download_headroom_bytes": 2 * 1024**3,
        },
        # No text encoder at all: a frozen 512-token prompt embedding ships with the
        # checkpoint and cross-attention runs against it.
        text_encoder_overrides={},
    ),
}
