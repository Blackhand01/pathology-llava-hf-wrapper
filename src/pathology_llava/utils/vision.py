from typing import Tuple

import torch
from torch import Tensor
from transformers import CLIPModel


def encode_plip_for_projector(
    vision_model: CLIPModel,
    pixel_values: Tensor,
    select_layer: int = -2,
) -> Tuple[Tensor, Tensor]:
    """
    Encode images with PLIP and extract features suitable for the Q-Former projector.

    This is a simplified approximation of the feature selection logic used in PA-LLaVA:
    - We take hidden states from an intermediate vision layer.
    - We treat the first token as 'global' query, the rest as patch tokens.

    Parameters
    ----------
    vision_model:
        CLIPModel instance loaded from the PLIP checkpoint.
    pixel_values:
        Tensor of shape (batch, 3, H, W).
    select_layer:
        Index of the hidden state layer to use (e.g. -2 for penultimate).

    Returns
    -------
    ori_pixel_embeds: Tensor
        Global token embeddings, shape (batch, 1, hidden_dim).
    patch_pixel_embeds: Tensor
        Patch token embeddings, shape (batch, num_patches, hidden_dim).
    """
    outputs = vision_model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[select_layer]  # (B, seq_len, hidden)
    # Typically [CLS] + patches; use that convention.
    ori_pixel_embeds = hidden_states[:, :1, :]          # (B, 1, D)
    patch_pixel_embeds = hidden_states[:, 1:, :]        # (B, P, D)
    return ori_pixel_embeds, patch_pixel_embeds
