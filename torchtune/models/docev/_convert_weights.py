# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
from torchtune import training
from torchtune.models.convert_weights import get_mapped_key
from safetensors.torch import save as save_safetensors, save_file
from torchtune.utils import get_logger
from torchtune.training.checkpointing._utils import ADAPTER_MODEL_FNAME

logger = get_logger("DEBUG")

_VISION_ENCODER={
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

_LDPV2_CONNECTOR={
    "image_newline": "encoder.image_newline",
    "multi_modal_projector.mlp.{}.bias" : "encoder.connector.mlp.{}.bias",
    "multi_modal_projector.mlp.{}.weight" : "encoder.connector.mlp.{}.weight",
    "multi_modal_projector.peg.{}.bias" : "encoder.connector.peg.{}.bias",
    "multi_modal_projector.peg.{}.weight" : "encoder.connector.peg.{}.weight",
}


_SOLAR_MINI = {
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
}


_PHI3_MINI = {
    "language_model.model.embed_tokens.weight": "decoder.tok_embeddings.weight",
    "language_model.model.layers.{}.self_attn.qkv_proj.weight": "decoder.layers.{}.attn.q_proj.weight",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "decoder.layers.{}.attn.output_proj.weight",
    "language_model.model.layers.{}.mlp.gate_up_proj.weight": "decoder.layers.{}.mlp.w1.weight",
    "language_model.model.layers.{}.mlp.down_proj.weight": "decoder.layers.{}.mlp.w2.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "decoder.layers.{}.sa_norm.scale",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "decoder.layers.{}.mlp_norm.scale",
    "language_model.model.norm.weight": "decoder.norm.scale",
    "language_model.lm_head.weight": "decoder.output.weight",
}
# add vision encoder and connector
_SOLAR_MINI.update(_VISION_ENCODER)
_SOLAR_MINI.update(_LDPV2_CONNECTOR)
_PHI3_MINI.update(_VISION_ENCODER)
_PHI3_MINI.update(_LDPV2_CONNECTOR)



# Mapping from torchtune LoRA module names to PEFT LoRA module names
_TO_PEFT_KEYS = {
    "lora_a": "lora_A",
    "lora_b": "lora_B",
    "magnitude": "lora_magnitude_vector",
}

def docev_solar_mini_tune_to_peft_adapter_weights(
    state_dict: Dict[str, torch.Tensor],
    # Decoder Parameters
    num_heads: int = 32,
    num_kv_heads: int = 8,
    dim: int = 4096,
    head_dim: int = None,
    # Checkpointer Parameters
    epoch: int = 0,
    output_dir: str = None,
    fs: Optional[Any] = None,
    safe_serialization: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert torchtune LoRA adapter weights into PEFT adapter format and save a checkpoint.

    This routine:
      1. Builds a key‐mapping from torchtune’s LoRA naming to PEFT’s LoRA naming.
      2. Applies a special permutation to the B‐matrices of q/k projections for grouped‐query attention.
      3. Prepends “base_model.model.” to each PEFT key.
      4. Writes out a safetensors (.safetensors) or standard PyTorch (.bin) checkpoint under
         `{output_dir}/epoch_{epoch}/{ADAPTER_MODEL_FNAME}`, creating directories via `fs.mkdirs`.
      5. Logs the final file size.

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        Torchtune adapter weights (LoRA A, LoRA B, magnitude vectors) keyed by torchtune names.
    num_heads : int, default=32
        Number of decoder attention heads (for q/k/v projections).
    num_kv_heads : int, default=8
        Number of key/value heads (for grouped‐query attention).
    dim : int, default=4096
        Total hidden size of the decoder.
    head_dim : int, optional
        Dimension per attention head; if None, computed as `dim // num_heads`.
    epoch : int, default=0
        Epoch index used in the output path (`epoch_{epoch}`).
    output_dir : str
        Base directory where the converted adapter checkpoint will be written.
    fs : Any, optional
        Filesystem abstraction (e.g., an `fsspec` filesystem) providing `.mkdirs()`, `.open()`, and `.size()`.
    safe_serialization : bool, default=True
        If True, save using `safetensors`; otherwise, fall back to `torch.save(..., .bin)`.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary mapping PEFT‐style adapter keys to their (possibly permuted) tensors.
    """

    converted_state_dict = {}
    full_mapping = {}
    # Rather than recreate a separate mapping for LoRA adapter weights, we re-use the
    # _FROM_HF mapping for base model weights. The mapping is adapted to account for:
    # LoRA A matrices, LoRA B matrices and the dora magnitude parameter.
    for peft_key, peft_val in _TO_PEFT_KEYS.items():
        for hf_key, hf_val in _SOLAR_MINI.items():
            if hf_val is None:
                continue

            if peft_key == "magnitude":
                # e.g. attn.q_proj.magnitude -> attn.q_proj.lora_magnitude_vector
                adapter_key = hf_val.replace(".weight", f".{peft_key}")
                adapter_val = hf_key.replace(".weight", f".{peft_val}")
            else:
                # e.g. attn.q_proj.lora_a.weight -> attn.q_proj.lora_A.weight
                adapter_key = hf_val.replace(".weight", f".{peft_key}.weight")
                adapter_val = hf_key.replace(".weight", f".{peft_val}.weight")

            full_mapping.update({adapter_key: adapter_val})

    if head_dim is None:
        head_dim = dim // num_heads

    def _permute_lora_matrix(t, n_heads):
        """
        Permute a LoRA B matrix to match the grouped-query attention (GQA) interleaved head layout.

        In GQA, the base Q/K weight rows are stored by splitting each head's rows into two
        sub-blocks (of size head_dim//2) and interleaving them across all heads. To apply a
        LoRA update correctly, its B matrix must be rearranged in the same pattern.

        This function performs three steps:
        1. **Reshape** `t` from shape `(n_heads * head_dim, rank)` into
            `(n_heads, head_dim//2, 2, rank)`, dividing each head’s rows into two halves.
        2. **Transpose** the middle two dimensions (1 ↔ 2) to interleave those halves:
            each head’s sub-blocks are swapped into the order expected by the base weight.
        3. **Flatten** back to `(n_heads * head_dim, rank)`, producing a row ordering
            that aligns with the model’s internal group-interleaved storage.

        Args:
            t (torch.Tensor): LoRA B matrix of shape `(n_heads * head_dim, rank)`.
            n_heads (int): Number of heads used for this projection (e.g. `num_heads` for Q,
                        `num_kv_heads` for K).

        Returns:
            torch.Tensor: Permuted LoRA B matrix of shape `(n_heads * head_dim, rank)`,
                        ready to be added to the base Q/K weight with group-interleaving.
        """
        rank = t.shape[-1]
        return (
            t.view(n_heads, head_dim // 2, 2, rank)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), rank)
        )

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, full_mapping)
        if "language_model" in key:
            if "q_proj" in new_key and "lora_B" in new_key:
                value = _permute_lora_matrix(value, num_heads)
            elif "k_proj" in new_key and "lora_B" in new_key:
                value = _permute_lora_matrix(value, num_kv_heads)
        converted_state_dict["base_model.model." + new_key] = value

    output_path = os.path.join(
        output_dir, f"epoch_{epoch}", ADAPTER_MODEL_FNAME
    )
    fs.mkdirs(os.path.dirname(output_path), exist_ok=True)
    if not safe_serialization:
        output_path = output_path + ".bin"
        with fs.open(output_path, "wb") as f:
            torch.save(converted_state_dict, f)
    else:
        output_path = output_path + ".safetensors"
        with fs.open(output_path, "wb") as f:
            save_bytes = save_safetensors(
                converted_state_dict,
                metadata={"format": "pt"},
            )
            f.write(save_bytes)
    logger.info(
        "Adapter checkpoint of size "
        f"{fs.size(output_path) / 1024**3:.2f} GiB "
        f"saved to {output_path}"
    )
    return converted_state_dict


def docev_solar_mini_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 8,
    dim: int = 4096,
    head_dim: int = None,
    vocab_size: int = 64000,

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
        new_key = get_mapped_key(key, _SOLAR_MINI)
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

def docev_solar_mini_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    # Decoder Parameters
    num_heads: int = 32,
    num_kv_heads: int = 8,
    dim: int = 4096,
    head_dim: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a torchtune DocEV model state dictionary to a Hugging Face-compatible format.

    This conversion process handles multiple transformations required for alignment between
    the two model architectures:
    - Maps keys from torchtune naming format to HF format using inverted _FROM_HF mapping
    - Handles fusion embedding and combines it back into the main token embedding
    - Reshapes query and key projection weights for attention mechanisms
    - Filters out any torchtune‐only keys (e.g., fusion weight key after merging)

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        The torchtune model state dictionary to convert.
    num_heads : int, default=32
        Number of attention heads in the decoder.
    num_kv_heads : int, default=8
        Number of key/value heads for grouped‐query attention.
    dim : int, default=4096
        Hidden dimension size of the model.
    head_dim : int, optional
        Per‐head dimension (if None, computed as `dim // num_heads`).

    Returns
    -------
    Dict[str, torch.Tensor]
        The converted state dictionary with Hugging Face–compatible names and shapes.
    """
    converted_state_dict = {}
    # Create inverted mapping from torchtune keys to HF keys
    inverted_mapping_dict = {v: k for k, v in _SOLAR_MINI.items() if v is not None}

    # Add missing keys not in _SOLAR_MINI due to naming collisions
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

def docev_phi4_mini_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 24,
    num_kv_heads: int = 8,
    dim: int = 3072,
    head_dim: int = None,
    vocab_size: int = 200064,

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

    Returns
    -------
    Dict[str, torch.Tensor]
        The converted state dictionary with torchtune-compatible parameter names and shapes
    """
    converted_state_dict = {}


    if dim is not None:
        if num_heads is None or num_kv_heads is None:
            raise ValueError(
                "Phi models with GQA require dim, num_heads and num_kv_heads to be specified"
            )
        q_dim = dim
        k_dim = q_dim * num_kv_heads // num_heads
        v_dim = q_dim * num_kv_heads // num_heads
    else:
        q_dim, k_dim, v_dim = None, None, None

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue
        if "vision_tower.vision_model.head" in key: # Skip loading the vision encoder head  layer
            continue
        new_key = get_mapped_key(key, _PHI3_MINI)
        if "language_model" in key:
            if "qkv" in key:
                if q_dim is not None:
                    q, k, v = torch.split(value, [q_dim, k_dim, v_dim], dim=0)
                else:
                    (
                        q,
                        k,
                        v,
                    ) = value.chunk(3, dim=0)
                converted_state_dict[new_key] = q
                converted_state_dict[new_key.replace("q_proj", "k_proj")] = k
                converted_state_dict[new_key.replace("q_proj", "v_proj")] = v
            elif "gate" in key:
                w1, w3 = value.chunk(2, dim=0)
                converted_state_dict[new_key] = w1
                converted_state_dict[new_key.replace("w1", "w3")] = w3
            elif new_key == "decoder.tok_embeddings.weight":
                # Split embedding between learnable embeddings and original text embedding
                learned_embedding = "decoder.tok_embeddings.fusion_embedding.weight"
                converted_state_dict[learned_embedding] = value[vocab_size:]
                value = value[:vocab_size]

        converted_state_dict[new_key] = value
    return converted_state_dict


def docev_phi4_mini_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    # Decoder Parameters
    num_heads: int = 24,
    num_kv_heads: int = 8,
    dim: int = 3072,
    head_dim: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Converts a torchtune DocEV model state dictionary to a Hugging Face-compatible format.

    This conversion process handles multiple transformations required for alignment between
    the two model architectures:
    - Maps keys from torchtune naming format to HF format using inverted _FROM_HF mapping
    - Handles fusion embedding and combines it back into the main token embedding
    - Reshapes query and key projection weights for attention mechanisms
    - Filters out any torchtune‐only keys (e.g., fusion weight key after merging)

    Parameters
    ----------
    state_dict : Dict[str, torch.Tensor]
        The torchtune model state dictionary to convert.
    num_heads : int, default=32
        Number of attention heads in the decoder.
    num_kv_heads : int, default=8
        Number of key/value heads for grouped‐query attention.
    dim : int, default=4096
        Hidden dimension size of the model.
    head_dim : int, optional
        Per‐head dimension (if None, computed as `dim // num_heads`).

    Returns
    -------
    Dict[str, torch.Tensor]
        The converted state dictionary with Hugging Face–compatible names and shapes.
    """
    converted_state_dict = {}
    # Create inverted mapping from torchtune keys to HF keys
    inverted_mapping_dict = {v: k for k, v in _PHI3_MINI.items() if v is not None}

    # Add missing keys not in _SOLAR_MINI due to naming collisions
    missing_keys = {
        "decoder.tok_embeddings.fusion_embedding.weight": None,
    }
    inverted_mapping_dict.update(missing_keys)

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if new_key is None:
            continue
        if "decoder" in key:
            if "k_proj" in key or "v_proj" in key or "w3" in key:
                # these keys are accounted for separately and should be skipped
                continue

            if "q_proj" in key:
                q = value
                k = state_dict[key.replace("q_proj", "k_proj")]
                v = state_dict[key.replace("q_proj", "v_proj")]
                qkv = torch.cat([q, k, v], dim=0)
                # q_proj maps to qkv_proj; no need to string replace
                converted_state_dict[new_key] = qkv

            elif "w1" in key:
                gate_proj = value
                up_proj = state_dict[key.replace("w1", "w3")]
                gate_up_proj = torch.cat([gate_proj, up_proj], dim=0)
                # w1 maps to gate_up_proj; no need to string replace
                converted_state_dict[new_key] = gate_up_proj
        converted_state_dict[new_key] = value
    return converted_state_dict