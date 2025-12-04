from typing import Dict, Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel


def build_multimodal_inputs(
    llm: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    image_embeds: Tensor,
) -> Dict[str, Tensor]:
    """
    Build multimodal inputs for the LLM by concatenating image embeddings
    as prefix tokens in the hidden space.

    This does NOT reproduce the exact PA-LLaVA training recipe, but provides
    a simple, working multimodal path:

      - Embed text tokens via the LLM token embedding matrix.
      - Concatenate image_embeds (B, I, H) in front of the text embeddings.
      - Build an extended attention mask.

    Parameters
    ----------
    llm:
        LLaMA model (AutoModelForCausalLM) with get_input_embeddings().
    input_ids:
        (B, T) token ids.
    attention_mask:
        (B, T) attention mask or None.
    image_embeds:
        (B, I, H) image embeddings projected into LLM hidden space.

    Returns
    -------
    inputs:
        Dict ready to feed into llm(**inputs).
    """
    text_embeds = llm.get_input_embeddings()(input_ids)  # (B, T, H)
    if text_embeds.dtype != image_embeds.dtype:
        image_embeds = image_embeds.to(text_embeds.dtype)
    if text_embeds.device != image_embeds.device:
        image_embeds = image_embeds.to(text_embeds.device)

    multimodal_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # (B, I+T, H)
    bsz, seq_len = input_ids.shape
    img_len = image_embeds.shape[1]

    if attention_mask is None:
        attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=input_ids.device)

    img_mask = torch.ones((bsz, img_len), dtype=attention_mask.dtype, device=attention_mask.device)
    extended_mask = torch.cat([img_mask, attention_mask], dim=1)  # (B, I+T)

    return {
        "inputs_embeds": multimodal_embeds,
        "attention_mask": extended_mask,
    }
