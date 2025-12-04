from typing import List, Union, Optional, Dict, Any

from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    ProcessorMixin,
)

from .configuration_pathology_llava import PathologyLLaVAConfig
from .constants import DEFAULT_VISION_TOWER, DEFAULT_LLM_NAME


class PathologyLLaVAProcessor(ProcessorMixin):
    """
    HF Processor for Pathology-LLaVA wrapper.

    It combines:
      - CLIP-style image processor from PLIP,
      - tokenizer from the underlying LLaMA model.

    Typical usage:
        processor = PathologyLLaVAProcessor.from_pretrained(...)
        inputs = processor(images=image, text="Describe the patch", return_tensors="pt")
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: CLIPImageProcessor,
        tokenizer,
        **kwargs,
    ):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.current_processor = self.image_processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "PathologyLLaVAProcessor":
        """
        For simplicity, this loads:
          - image_processor from PLIP,
          - tokenizer from the LLaMA model used by the config.
        """
        config = PathologyLLaVAConfig.from_pretrained(pretrained_model_name_or_path)

        vision_id = config.vision_tower_name_or_path or DEFAULT_VISION_TOWER
        llm_id = config.llm_name_or_path or DEFAULT_LLM_NAME

        image_processor = CLIPImageProcessor.from_pretrained(vision_id)
        tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            use_fast=True,
            padding_side="right",
        )
        return cls(image_processor=image_processor, tokenizer=tokenizer, **kwargs)

    def __call__(
        self,
        images: Union[Image.Image, torch.Tensor, List[Image.Image], List[torch.Tensor]],
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Normalize to list for both images and text.
        if isinstance(images, (Image.Image, torch.Tensor)):
            images = [images]
        if isinstance(text, str):
            text = [text]

        image_outputs = self.image_processor(
            images=images,
            return_tensors="pt",
        )

        text_outputs = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        data = {
            "pixel_values": image_outputs["pixel_values"],
            "input_ids": text_outputs["input_ids"],
            "attention_mask": text_outputs["attention_mask"],
        }

        if return_tensors is None or return_tensors == "pt":
            return data

        raise ValueError(f"Unsupported return_tensors={return_tensors!r} for this processor.")
