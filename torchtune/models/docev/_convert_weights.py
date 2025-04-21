# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch

from torchtune.models.convert_weights import get_mapped_key

_FROM_HF = {
    "image_newline": "encoder.image_newline",
    "language_model.model.norm.weight": "decoder.norm.scale",
    "language_model.lm_head.weight": "decoder.output.weight",
    "language_model.model.embed_tokens.weight": "decoder.tok_embeddings.weight",
    "language_model.model.layers.{}.input_layernorm.weight" : "decoder.layers.{}.sa_norm.scale",
    "language_model.model.layers.{}.mlp.down_proj.weight" : "decoder.layers.{}.mlp.w2.weight",
    "language_model.model.layers.{}.mlp.gate_proj.weight" : "decoder.layers.{}.mlp.w1.weight",
    "language_model.model.layers.{}.mlp.up_proj.weight": "decoder.layers.{}.mlp.w3.weight",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "decoder.layers.{}.mlp_norm.scale",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "decoder.layers.{}.attn.k_proj.weight",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "decoder.layers.{}.attn.output_proj.weight",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "decoder.layers.{}.attn.q_proj.weight",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "decoder.layers.{}.attn.v_proj.weight",
    "multi_modal_projector.mlp.{}.bias" : "encoder.connector.mlp.{}.bias",
    "multi_modal_projector.mlp.{}.weight" : "encoder.connector.mlp.{}.weight",
    "multi_modal_projector.peg.{}.bias" : "encoder.connector.peg.{}.bias",
    "multi_modal_projector.peg.{}.weight" : "encoder.connector.peg.{}.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias" : "encoder.clip.conv.bias",
    "vision_tower.vision_model.embeddings.patch_embedding.weight" : "encoder.clip.conv.weight",
    "vision_tower.vision_model.embeddings.position_embedding.weight" : "encoder.clip.token_pos_embedding.positional_embedding",
    "vision_tower.vision_model.encoder.layers.{}.layer_norm1.bias": "encoder.clip.layers.{}.sa_norm.bias",
    "vision_tower.vision_model.encoder.layers.{}.layer_norm1.weight" : "encoder.clip.layers.{}.sa_norm.weight",
    "vision_tower.vision_model.encoder.layers.{}.layer_norm2.bias" : "encoder.clip.layers.{}.mlp_norm.bias",
    "vision_tower.vision_model.encoder.layers.{}.layer_norm2.weight" : "encoder.clip.layers.{}.mlp_norm.weight",
    "vision_tower.vision_model.encoder.layers.{}.mlp.fc1.bias" : "encoder.clip.layers.{}.mlp.w1.bias",
    "vision_tower.vision_model.encoder.layers.{}.mlp.fc1.weight" : "encoder.clip.layers.{}.mlp.w1.weight",
    "vision_tower.vision_model.encoder.layers.{}.mlp.fc2.bias" : "encoder.clip.layers.{}.mlp.w2.bias",
    "vision_tower.vision_model.encoder.layers.{}.mlp.fc2.weight" : "encoder.clip.layers.{}.mlp.w2.weight",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj.bias" : "encoder.clip.layers.{}.attn.k_proj.bias",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj.weight" : "encoder.clip.layers.{}.attn.k_proj.weight",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj.bias" : "encoder.clip.layers.{}.attn.output_proj.bias",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj.weight" : "encoder.clip.layers.{}.attn.output_proj.weight",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj.bias" : "encoder.clip.layers.{}.attn.q_proj.bias",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj.weight" : "encoder.clip.layers.{}.attn.q_proj.weight",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj.bias" : "encoder.clip.layers.{}.attn.v_proj.bias",
    "vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj.weight" : "encoder.clip.layers.{}.attn.v_proj.weight",
    "vision_tower.vision_model.head.attention.in_proj_bias" : None,
    "vision_tower.vision_model.head.attention.in_proj_weight" : None,
    "vision_tower.vision_model.head.attention.out_proj.bias" : None,
    "vision_tower.vision_model.head.attention.out_proj.weight" : None,
    "vision_tower.vision_model.head.layernorm.bias" : None,
    "vision_tower.vision_model.head.layernorm.weight" : None,
    "vision_tower.vision_model.head.mlp.fc1.bias" : None,
    "vision_tower.vision_model.head.mlp.fc1.weight" : None,
    "vision_tower.vision_model.head.mlp.fc2.bias" : None,
    "vision_tower.vision_model.head.mlp.fc2.weight" : None,
    "vision_tower.vision_model.head.probe" : None,
    "vision_tower.vision_model.post_layernorm.bias" : "encoder.clip.ln_post.bias",
    "vision_tower.vision_model.post_layernorm.weight" : "encoder.clip.ln_post.weight",
}

def docev_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    dim: int = 4096,
    head_dim: int = None,
    vocab_size: int = 64000,
    # Vision Encoder Paramters
    encoder_dim: int = 1152,
    tile_size: int = 560,
    num_tiles: int = 9,
    supported_aspect_ratios: List[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a Hugging Face DocEV model state dictionary to a torchtune-compatible format.

    This conversion process handles multiple transformations required for alignment between
    the two model architectures:
    - Maps keys from HF naming format to torchtune format using the _FROM_HF mapping dictionary
    - Skips loading rotary position embeddings since they're computed on the fly
    - Ignores vision encoder head layer which isn't used in torchtune implementation
    - Reshapes query and key projection weights for attention mechanisms
    - Splits token embeddings into vocabulary embeddings and learned fusion embeddings

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        The Hugging Face model state dictionary to convert
    num_heads : int, default=32
        Number of attention heads in the decoder
    num_kv_heads : int, default=8
        Number of key/value heads in the decoder (for grouped-query attention)
    dim : int, default=4096
        Hidden dimension size of the model
    head_dim : int, optional
        Dimension of each attention head. If None, computed as dim / num_heads
    vocab_size : int, default=64000
        Size of the model vocabulary for proper embedding separation
    encoder_dim : int, default=1152
        Hidden dimension size of the vision encoder
    tile_size : int, default=560
        Size of each image tile processed by the vision encoder
    num_tiles : int, default=9
        Maximum number of image tiles supported by the model
    supported_aspect_ratios : List[Tuple[int, int]], optional
        List of supported image aspect ratios for positioning embeddings

    Returns
    -------
    Dict[str, torch.Tensor]
        The converted state dictionary with torchtune-compatible parameter names and shapes
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2).contiguous()
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue
        if "vision_tower.vision_model.head" in key: # Skip loading the vision encoder head  layer
            continue
        new_key = get_mapped_key(key, _FROM_HF)
        if "language_model" in key:
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)
            elif new_key == "decoder.tok_embeddings.weight":
                # Split embedding between learnable embeddings and original text embedding
                learned_embedding = "decoder.tok_embeddings.fusion_embedding.weight"
                converted_state_dict[learned_embedding] = value[vocab_size:]
                value = value[:vocab_size]
        converted_state_dict[new_key] = value
    return converted_state_dict

def docev_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    dim: int = 4096,
    head_dim: int = None,
    vocab_size: int = 64000,
    # Vision Encoder Parameters
    encoder_dim: int = 1152,
    tile_size: int = 560,
    num_tiles: int = 9,
    supported_aspect_ratios: List[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a torchtune DocEV model state dictionary to a Hugging Face-compatible format.

    This conversion process handles multiple transformations required for alignment between
    the two model architectures:
    - Maps keys from torchtune naming format to HF format using inverted _FROM_HF mapping
    - Handles fusion layer and cross-attention layer numbering
    - Reshapes query and key projection weights for attention mechanisms
    - Combines vocabulary embeddings and learned fusion embeddings into single tensor
    - Processes positional embeddings for the vision encoder

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        The torchtune model state dictionary to convert
    num_heads : int, default=32
        Number of attention heads in the decoder
    num_kv_heads : int, default=8
        Number of key/value heads in the decoder (for grouped-query attention)
    dim : int, default=4096
        Hidden dimension size of the model
    head_dim : int, optional
        Dimension of each attention head. If None, computed as dim / num_heads
    vocab_size : int, default=64000
        Size of the model vocabulary
    cross_attention_layers : Optional[List[int]], default=None
        List of layer indices that contain cross-attention mechanisms
    encoder_dim : int, default=1152
        Hidden dimension size of the vision encoder
    tile_size : int, default=560
        Size of each image tile processed by the vision encoder
    num_tiles : int, default=9
        Maximum number of image tiles supported by the model
    supported_aspect_ratios : List[Tuple[int, int]], optional
        List of supported image aspect ratios for positioning embeddings

    Returns
    -------
    Dict[str, torch.Tensor]
        The converted state dictionary with Hugging Face-compatible parameter names and shapes
    """
    converted_state_dict = {}
    # Create inverted mapping from torchtune keys to HF keys
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items() if v is not None}

    # Add missing keys not in _FROM_HF due to naming collisions
    missing_keys = {
        "decoder.tok_embeddings.fusion_embedding.weight": None,
    }
    inverted_mapping_dict.update(missing_keys)

    if head_dim is None:
        head_dim = dim // num_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, head_dim // 2, 2, dim)
            .transpose(1, 2).contiguous()
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if new_key is None:
            continue

        if "decoder" in key:
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)
            elif key == "decoder.tok_embeddings.weight":
                # Combine main embeddings with fusion embeddings
                if "decoder.tok_embeddings.fusion_embedding.weight" in state_dict:
                    fusion_embedding = state_dict["decoder.tok_embeddings.fusion_embedding.weight"]
                    value = torch.cat([value, fusion_embedding])
            elif key == "decoder.tok_embeddings.fusion_embedding.weight":
                continue  # Skip as it's handled with decoder.tok_embeddings.weight

        converted_state_dict[new_key] = value
    return converted_state_dict
