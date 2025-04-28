# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from functools import partial
from typing import List, Optional, Dict, Tuple, Literal, Union

from torchtune.models.docev._component_builders import (  # noqa
    docev_encoder_with_connector,
    docev_solar_decoder,
    lora_docev_encoder_with_connector,
    lora_docev_solar_decoder,
    LoRATrainable,
)

from torchtune.models.docev._transform import DocEVTransform
from torchtune.models.docev._early_fusion import EarlyFusionModel
from torchtune.modules.peft import LORA_ATTN_MODULES

def docev_preview_transform(
    model_name_or_path: str,
    image_token: str,
    stop_tokens: List[str],
    tile_size: int,
    patch_size: int,
    max_num_tiles: int,
    min_num_tiles: int,
    vision_feature_select_strategy: Literal["default", "full"],
    sampling_ratio: List[int],
    apply_random_sampling_ratio: bool,
    max_seq_len: int,
    chat_template: Optional[str],
    ufx_type: Literal["instruction", "pretraining"],
) -> DocEVTransform:
    """
    Data Transforms (including Tokenizer) for DocEV.

    Args:
        model_name_or_path (str): Path or name of the tokenizer model
        image_token (str): Token used to represent images in the input
        stop_tokens (List[str]): List of tokens that signal the end of generation
        tile_size (int): Size of image tiles for processing document images
        patch_size (int): Size of patches within each tile for vision processing
        max_num_tiles (int): Maximum number of tiles to process from an image
        min_num_tiles (int): Minimum number of tiles to process from an image
        vision_feature_select_strategy (Literal["default", "full"]): Strategy for selecting vision features
        sampling_ratio (List[int]): Ratios for sampling tiles from different regions
        apply_random_sampling_ratio (bool): Whether to apply random sampling ratios
        max_seq_len (int): Maximum sequence length for tokenizing input
        chat_template (Optional[str]): Template for formatting chat messages

    Returns:
        DocEVTransform: Instantiation of the DocEV transform
    """

    return DocEVTransform(
        model_name_or_path=model_name_or_path,
        image_token=image_token,
        stop_tokens=stop_tokens,
        tile_size=tile_size,
        patch_size=patch_size,
        max_num_tiles=max_num_tiles,
        min_num_tiles=min_num_tiles,
        vision_feature_select_strategy=vision_feature_select_strategy,
        sampling_ratio=sampling_ratio,
        apply_random_sampling_ratio=apply_random_sampling_ratio,
        max_seq_len=max_seq_len,
        chat_template=chat_template,
        ufx_type=ufx_type,
    )


def docev_preview(
    image_token_id: int,
    decoder_trainable: bool,
    encoder_trainable: Union[bool, Dict[str, bool]],
    fusion_trainable: bool,
) -> EarlyFusionModel:
    """DocEV Preview model based on Llama 3.2 Vision architecture

    Args:
        image_token_id (int): Token ID used to represent images in the input
        decoder_trainable (bool): Whether to make decoder params trainable
        encoder_trainable (Union[bool, Dict[str, bool]]): Whether to make encoder params trainable.
            Can be a boolean or a dictionary specifying trainability for different components
        fusion_trainable (bool): Whether to make fusion params trainable

    Returns:
        EarlyFusionModel: Instantiation of the DocEV Preview model
    """
    # siglip so-400m & ldp-v2 hyper-parameters
    encoder = docev_encoder_with_connector(
        patch_size=14,
        num_heads=16,
        attn_bias=True, # q,k,v,out proj bias
        use_rope=False,
        activation=nn.GELU,
        clip_embed_dim=1152,
        clip_hidden_dim=4304,
        clip_num_layers=27,
        clip_hidden_states=[24, 25, 26], # possible range: 0 ~ clip_num_layers-1
        cls_output_dim=1152,
        append_cls_token=False,
        output_cls_projection=False,
        decoder_embed_dim=4096,
        vision_select_layer=-2,
        connector_type="ldp_v2",
        vision_feature_select_strategy="full",
        tile_size=560,
        max_num_tiles=9,
        in_channels=3,
    )
    # solar mini hyper-parameters
    decoder = docev_solar_decoder(
        vocab_size=64000,
        fusion_vocab_size=64,
        num_layers=48,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=65536,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
    )
    return EarlyFusionModel(
        decoder=decoder,
        encoder=encoder,
        image_token_id=image_token_id,
        decoder_trainable=decoder_trainable,
        encoder_trainable=encoder_trainable,
        fusion_trainable=fusion_trainable,
    )

def lora_docev_preview(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    image_token_id: int,
    decoder_trainable: str = "frozen",
    encoder_trainable: str = "lora",
    fusion_trainable: str = "lora",
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> EarlyFusionModel:
    """
    Return a version of DocEV Preview model (an instance of EarlyFusionModel)
    with LoRA applied based on the passed in configuration.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        decoder_trainable (str): Option to set decoder params as fully trainble (full), lora trainable (lora),
            or frozen (frozen). The default is "frozen".
        encoder_trainable (str): Option to set encoder params as fully trainble (full), lora trainable (lora),
            or frozen (frozen). The default is "lora".
        fusion_trainable (str): Option to set fusion params as fully trainble (full), lora trainable (lora),
            or frozen (frozen). The default is "lora".
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA (weight-decomposed low-rank adaptation) instead of LoRA.
            Default: False
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        EarlyFusionModel: Instantiation of DocEV Preview model with LoRA applied to
        a subset of the attention projections in each layer.
    """
    decoder_type = LoRATrainable(decoder_trainable.lower())
    encoder_type = LoRATrainable(encoder_trainable.lower())
    fusion_type = LoRATrainable(fusion_trainable.lower())
    assert LoRATrainable.FULL not in [
        decoder_type,
        encoder_type,
        fusion_type,
    ], "We've temporarily removed support for mixed LoRA + Full Finetuning yet. Please don't use the 'full' option and use llama3_2_vision_11b if you need full finetuning"

    encoder = lora_docev_encoder_with_connector(
        encoder_lora=encoder_type == LoRATrainable.LORA,
        fusion_lora=fusion_type == LoRATrainable.LORA,
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        patch_size=14,
        num_heads=16,
        attn_bias=True,
        use_rope=False,
        activation=nn.GELU,
        clip_embed_dim=1152,
        clip_hidden_dim=4304,
        clip_num_layers=27,
        clip_hidden_states=[24, 25, 26],
        cls_output_dim=1152,
        append_cls_token=False,
        output_cls_projection=False,
        decoder_embed_dim=4096,
        vision_select_layer=-2,
        tile_size=560,
        max_num_tiles=9,
        in_channels=3,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        connector_type="ldp_v2",
        vision_feature_select_strategy="full",
    )

    decoder = lora_docev_solar_decoder(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=64000,
        fusion_vocab_size=64,
        num_layers=48,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=65536,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

    return EarlyFusionModel(
        encoder=encoder,
        decoder=decoder,
        image_token_id=image_token_id,
        encoder_trainable=encoder_type != LoRATrainable.FROZEN,
        decoder_trainable=decoder_type != LoRATrainable.FROZEN,
        fusion_trainable=fusion_type != LoRATrainable.FROZEN,
    )


qlora_docev_preview = partial(lora_docev_preview, quantize_base=True)

qlora_docev_preview.__doc__ = """
Builder for creating a Llama3.2 vision 11B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_docev_1_0_preview` for full API arguments.
"""