from transformers import LlamaConfig, PretrainedConfig


class SarashinaVisionEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        depth: int = 32,
        embed_dim: int = 1280,
        hidden_size: int = 3584,
        hidden_act: str = "quick_gelu",
        mlp_ratio: float = 4,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size


class SarashinaVisionConfig(PretrainedConfig):
    model_type = "sarashina2_vision"
    sub_configs = {
        "text_config": LlamaConfig,
        "vision_config": SarashinaVisionEncoderConfig,
    }

    def __init__(
        self,
        text_config: dict | LlamaConfig | None = None,
        vision_config: dict | SarashinaVisionEncoderConfig | None = None,
        image_token_index: int = 14,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Keep the checkpoint's fractional MLP ratio for Sarashina2.2.
        self.vision_config = (
            vision_config
            if isinstance(vision_config, SarashinaVisionEncoderConfig)
            else SarashinaVisionEncoderConfig(**(vision_config or {}))
        )
        self.text_config = (
            text_config
            if isinstance(text_config, LlamaConfig)
            else LlamaConfig(**(text_config or {}))
        )
        self.image_token_index = image_token_index
