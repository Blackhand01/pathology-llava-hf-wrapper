from typing import Optional, List, Union, Any, Dict

import torch
from torch import nn, Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    CLIPModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_pathology_llava import PathologyLLaVAConfig
from .utils import (
    resolve_vision_tower_path,
    resolve_llm_path,
    resolve_projector_path,
    encode_plip_for_projector,
    build_multimodal_inputs,
)


class PathologyLLaVAForConditionalGeneration(PreTrainedModel):
    """
    Hugging Face-compatible multimodal wrapper for PA-LLaVA.

    This class:
      - loads PLIP as vision tower,
      - loads a LLaMA-based AutoModelForCausalLM,
      - loads the projector / Q-Former model from the official Pathology-LLaVA
        distribution,
      - exposes a `generate` method via the underlying language model.

    No third-party weights are distributed with this package.
    """

    config_class = PathologyLLaVAConfig

    def __init__(self, config: PathologyLLaVAConfig):
        super().__init__(config)

        # Vision tower (PLIP / CLIP-style).
        vision_id_or_path = resolve_vision_tower_path(config.vision_tower_name_or_path)
        self.vision_model = CLIPModel.from_pretrained(
            vision_id_or_path,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else None,
        )

        # Language model (LLaMA).
        llm_id_or_path = resolve_llm_path(config.llm_name_or_path)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            llm_id_or_path,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else None,
        )

        # Projector / Q-Former from Pathology-LLaVA official weights.
        projector_dir = resolve_projector_path(
            pathology_llava_root=config.pathology_llava_root,
            projector_subfolder=config.projector_subfolder,
            pathology_llava_repo_id=config.pathology_llava_repo_id,
        )
        # We rely on the projector being exported as an AutoModel with custom code.
        self.projector = AutoModel.from_pretrained(
            projector_dir,
            trust_remote_code=True,
            torch_dtype=getattr(torch, config.torch_dtype) if config.torch_dtype else None,
        )

        # Ensure LLM does not use cache by default (better for training / gradients).
        if hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False

        # Tie weights if needed.
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "PathologyLLaVAForConditionalGeneration":
        """
        Standard HF loading path. This method is mainly kept to be explicit.
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def _encode_images(self, pixel_values: Tensor) -> Tensor:
        """
        Run PLIP + projector to obtain image embeddings in LLM hidden space.

        Parameters
        ----------
        pixel_values:
            Tensor of shape (B, 3, H, W).

        Returns
        -------
        image_embeds: Tensor
            (B, I, H_lm) image embeddings aligned to the LLM hidden size.
        """
        ori_pixel_embeds, patch_pixel_embeds = encode_plip_for_projector(
            self.vision_model,
            pixel_values.to(self.vision_model.dtype),
        )

        # Simple attention mask: consider all patches as valid tokens.
        device = patch_pixel_embeds.device
        image_atts = torch.ones(
            patch_pixel_embeds.size()[:-1],
            dtype=torch.long,
            device=device,
        )

        # The projector is expected to follow the PA-LLaVA Q-Former API:
        # forward(ori_pixel, patch_pixel, image_atts=None)
        projector_out = self.projector(
            ori_pixel=ori_pixel_embeds,
            patch_pixel=patch_pixel_embeds,
            image_atts=image_atts,
        )
        # We accept both dict-like and tensor outputs.
        if isinstance(projector_out, dict):
            image_embeds = projector_out.get("last_hidden_state", None)
            if image_embeds is None:
                # Fallback: try attribute.
                image_embeds = getattr(projector_out, "last_hidden_state", None)
        else:
            image_embeds = getattr(projector_out, "last_hidden_state", None)

        if image_embeds is None:
            raise RuntimeError(
                "Projector did not return `last_hidden_state`. "
                "Check the exported Pathology-LLaVA projector."
            )

        # Ensure dtype/device compatibility with the LLM.
        image_embeds = image_embeds.to(self.language_model.dtype).to(
            self.language_model.device
        )
        return image_embeds

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass that optionally incorporates image features.

        If `pixel_values` is provided, images are encoded and injected as
        prefix tokens in the LLM hidden space.
        """
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pixel_values are given.")
            if input_ids.device != self.language_model.device:
                input_ids = input_ids.to(self.language_model.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.language_model.device)

            image_embeds = self._encode_images(pixel_values)
            mm_inputs = build_multimodal_inputs(
                self.language_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds=image_embeds,
            )

            if labels is not None:
                labels = labels.to(self.language_model.device)
                mm_inputs["labels"] = labels

            outputs = self.language_model(**mm_inputs, **kwargs)
        else:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # Wrap result as CausalLMOutputWithPast for HF compatibility.
        if isinstance(outputs, CausalLMOutputWithPast):
            return outputs

        # Minimal adapter in case underlying model returns plain logits.
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = getattr(outputs, "loss", None)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
