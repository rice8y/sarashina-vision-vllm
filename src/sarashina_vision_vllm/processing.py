import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import isqrt
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import normalize, rescale
from transformers.image_utils import ChannelDimension
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.utils import IMAGE_PROCESSOR_NAME, cached_file
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from .configuration import SarashinaVisionConfig

IMAGE_TOKEN = "<|file|>"


@dataclass(frozen=True)
class ImagePreprocessing:
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    rescale_factor: float

    @property
    def resize_factor(self) -> int:
        return self.patch_size * self.merge_size

    def preprocess(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        height, width = smart_resize(
            image.height,
            image.width,
            self.resize_factor,
            self.min_pixels,
            self.max_pixels,
        )
        # Sarashina interpolates normalized float pixels rather than resizing bytes.
        channel_dim = ChannelDimension.LAST
        pixels = normalize(
            rescale(
                np.asarray(image.convert("RGB")),
                self.rescale_factor,
                input_data_format=channel_dim,
            ),
            self.image_mean,
            self.image_std,
            input_data_format=channel_dim,
        )
        pixels = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
        pixels = F.interpolate(
            pixels, size=(height, width), mode="bicubic", align_corners=False
        )
        patch, merge, temporal = (
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
        )
        grid_h, grid_w = height // patch, width // patch
        channels = pixels.shape[1]
        # Order patches by spatial merge group, duplicating the still frame in time.
        patches = (
            pixels.expand(temporal, -1, -1, -1)
            .reshape(
                temporal,
                channels,
                grid_h // merge,
                merge,
                patch,
                grid_w // merge,
                merge,
                patch,
            )
            .permute(2, 5, 3, 6, 1, 0, 4, 7)
            .reshape(grid_h * grid_w, channels * temporal * patch**2)
        )
        return patches, (1, grid_h, grid_w)


class SarashinaProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> SarashinaVisionConfig:
        return self.ctx.get_hf_config(SarashinaVisionConfig)

    @cached_property
    def image_preprocessing(self) -> ImagePreprocessing:
        config = self.ctx.get_hf_image_processor_config()
        if not config:
            # Some Transformers versions skip the legacy file when processor_config.json exists.
            model = self.ctx.model_config
            path = cached_file(
                model.model,
                IMAGE_PROCESSOR_NAME,
                revision=model.revision,
                token=model.hf_token,
            )
            if path is None:
                raise ValueError(
                    f"Image preprocessing configuration not found: {model.model}"
                )
            config = json.loads(Path(path).read_text(encoding="utf-8"))
        vision = self.get_hf_config().vision_config
        return ImagePreprocessing(
            patch_size=vision.patch_size,
            temporal_patch_size=vision.temporal_patch_size,
            merge_size=vision.spatial_merge_size,
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
            image_mean=tuple(config["image_mean"]),
            image_std=tuple(config["image_std"]),
            rescale_factor=config["rescale_factor"],
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        processor = self.image_preprocessing
        return {"image": processor.max_pixels // processor.resize_factor**2}


class SarashinaDummyInputsBuilder(BaseDummyInputsBuilder[SarashinaProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        processor = self.info.image_preprocessing
        factor = processor.resize_factor
        side = isqrt(processor.max_pixels) // factor * factor
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class SarashinaMultiModalProcessor(BaseMultiModalProcessor[SarashinaProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer.encode(prompt, **tok_kwargs)
        output: dict[str, object] = {"input_ids": [input_ids]}
        if images := cast(Sequence[Image.Image] | None, mm_data.get("images")):
            patches, grids = zip(
                *(self.info.image_preprocessing.preprocess(image) for image in images)
            )
            output.update(
                pixel_values=torch.cat(patches), image_grid_thw=torch.tensor(grids)
            )
        return BatchFeature(output)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        grids = hf_inputs.get("image_grid_thw")
        patch_counts = (
            grids.prod(-1) if grids is not None else torch.empty(0, dtype=torch.long)
        )
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", patch_counts
            ),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        image_token_id = config.image_token_index
        merge_area = config.vision_config.spatial_merge_size**2

        def replacement(index: int) -> list[int]:
            grid = out_mm_kwargs["image"][index]["image_grid_thw"].data
            assert isinstance(grid, torch.Tensor)
            return [image_token_id] * (int(grid.prod()) // merge_area)

        return [PromptReplacement("image", [image_token_id], replacement)]
