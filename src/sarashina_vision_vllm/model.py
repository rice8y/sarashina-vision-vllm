from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_vl import Qwen2VisionTransformer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from .processing import (
    IMAGE_TOKEN,
    SarashinaDummyInputsBuilder,
    SarashinaMultiModalProcessor,
    SarashinaProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    SarashinaMultiModalProcessor,
    info=SarashinaProcessingInfo,
    dummy_inputs=SarashinaDummyInputsBuilder,
)
class Sarashina2VisionForCausalLM(nn.Module, SupportsMultiModal):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality != "image":
            raise ValueError(f"Unsupported modality: {modality}")
        return IMAGE_TOKEN

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if vllm_config.model_config.tokenizer_mode != "sarashina":
            raise ValueError("Sarashina Vision requires tokenizer_mode='sarashina'")
        config = vllm_config.model_config.hf_config
        text = config.text_config
        vision = config.vision_config
        self.merge_area = vision.spatial_merge_size**2
        self.configure_mm_token_handling(text.vocab_size, [config.image_token_index])

        with self._mark_tower_model(vllm_config, "image"):
            self.visual = Qwen2VisionTransformer(
                vision, prefix=maybe_prefix(prefix, "visual")
            )
            if vision.hidden_act == "gelu_pytorch_tanh":
                for block in self.visual.blocks:
                    block.mlp.act = nn.GELU(approximate="tanh")
            elif vision.hidden_act != "quick_gelu":
                raise ValueError(f"Unsupported vision activation: {vision.hidden_act}")
            self.norm = nn.LayerNorm(text.hidden_size)

        with self._mark_language_model(vllm_config):
            self.llm = init_vllm_registered_model(
                vllm_config,
                hf_config=text,
                architectures=["LlamaForCausalLM"],
                prefix=maybe_prefix(prefix, "llm"),
            )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        pixels = cast(torch.Tensor | None, kwargs.get("pixel_values"))
        if pixels is None:
            return ()
        grid = cast(torch.Tensor, kwargs["image_grid_thw"])
        features = self.norm(self.visual(pixels, grid_thw=grid.tolist()))
        return features.split((grid.prod(-1) // self.merge_area).tolist())

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.llm.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.llm.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="llm", tower_model="visual", connector="norm"
        )
