# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import partial
from typing import List, Optional, Literal, Callable

from torch import nn
from torchtune.models.clip._component_builders import (
    clip_vision_encoder,
    lora_clip_vision_encoder,
    lora_clip_mlp,
)

from torchtune.models.docev._encoder import (
    DocEVEncoderWithConnector,
    DocEVLDPv2Connector,
    SiglipVisionTransformer,
    TokenPositionalEmbedding,
)
from torchtune.models.llama3._component_builders import (
    llama3_mlp,
    lora_llama3_self_attention,
    lora_llama3_mlp,
)

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import (
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    VisionRotaryPositionalEmbeddings
)
from torchtune.modules.vision_transformer import CLSProjection
from torchtune.models.clip import (
    clip_mlp,
)
from torchtune.modules.common_utils import (
    _register_reparametrize_state_dict_hooks,
    reparametrize_as_dtype_state_dict_post_hook,
)
from torchtune.modules.model_fusion import FusionEmbedding
from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear


class LoRATrainable(Enum):
    """
    Enumeration of different trainable modes for LoRA-based models.

    Attributes:
        FULL: Train all parameters of the model.
        LORA: Train only the LoRA parameters.
        FROZEN: Freeze all parameters, no training.
    """
    FULL = "full"
    LORA = "lora"
    FROZEN = "frozen"


def docev_ldp_v2_connector(
    *,
    clip_embed_dim: int,
    decoder_embed_dim: int,
) -> DocEVLDPv2Connector:
    """
    Build the DocEV LDPv2 Connector that maps the output of the CLIP encoder
    to the decoder input.

    This connector consists of two main components:
    1. An MLP layer that projects CLIP embeddings to decoder dimension
    2. A positional encoding generator (PEG) implemented with a depthwise convolution

    Args:
        clip_embed_dim (int): Embedding dimension for the CLIP encoder.
        decoder_embed_dim (int): Embedding dimension for the decoder.

    Returns:
        DocEVLDPv2Connector: A connector module that transforms CLIP embeddings into
        a format suitable for the text decoder, enhancing vision-language alignment.

    Example:
        >>> connector = docev_ldp_v2_connector(
        ...     clip_embed_dim=1024,
        ...     decoder_embed_dim=4096
        ... )
        >>> clip_features = torch.randn(1, 100, 1024)  # [batch, seq_len, dim]
        >>> decoder_features = connector(clip_features)
    """
    mlp_layer = nn.Sequential(
        nn.Linear(clip_embed_dim, decoder_embed_dim),
        nn.GELU(),
        nn.Linear(decoder_embed_dim, decoder_embed_dim)
    )
    peg_layer = nn.Sequential(
        nn.Conv2d(decoder_embed_dim, decoder_embed_dim, 3, 1, 1, bias=True, groups=decoder_embed_dim)
    )

    return DocEVLDPv2Connector(
        mlp_layer=mlp_layer,
        peg_layer=peg_layer,
    )


def lora_docev_ldp_v2_connector(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # Connector args
    clip_embed_dim: int,
    decoder_embed_dim: int,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    **quantization_kwargs,
) -> DocEVLDPv2Connector:
    """
    Build the DocEV LDPv2 Connector with LoRA (Low-Rank Adaptation) or DoRA (Decomposed
    Low-Rank Adaptation) applied to the linear projections.

    This connector uses parameter-efficient fine-tuning techniques to adapt pre-trained
    models by adding small trainable matrices (low-rank updates) to the weight matrices
    of the linear layers. This approach allows for efficient adaptation while keeping
    most of the original model parameters frozen.

    The connector consists of:
    1. A projection from CLIP embedding dimension to decoder dimension with LoRA/DoRA
    2. A GELU activation
    3. A second projection with LoRA/DoRA
    4. A positional encoding generator using depthwise convolution

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): List of which linear layers
            LoRA should be applied to. Options are ``{"q_proj", "k_proj", "v_proj",
            "output_proj"}``.
        clip_embed_dim (int): Embedding dimension for the CLIP encoder.
        decoder_embed_dim (int): Embedding dimension for the decoder.
        lora_rank (int): Rank of each low-rank approximation.
        lora_alpha (float): Scaling factor for the low-rank approximation.
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.

    Returns:
        DocEVLDPv2Connector: A connector module with LoRA/DoRA adaptations that transforms
        CLIP embeddings into a format suitable for the text decoder.

    Example:
        >>> lora_connector = lora_docev_ldp_v2_connector(
        ...     lora_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        ...     clip_embed_dim=1024,
        ...     decoder_embed_dim=4096,
        ...     lora_rank=8,
        ...     lora_alpha=16,
        ...     lora_dropout=0.05,
        ...     use_dora=False,
        ...     quantize_base=False
        ... )
        >>> clip_features = torch.randn(1, 100, 1024)  # [batch, seq_len, dim]
        >>> decoder_features = lora_connector(clip_features)
    """

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    # TODO: quantize_base is not applied to final output_proj currently.
    adapter_cls = DoRALinear if use_dora else LoRALinear
    clip_proj = adapter_cls(clip_embed_dim, decoder_embed_dim, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, use_bias=True)
    intermediate_proj = adapter_cls(decoder_embed_dim, decoder_embed_dim, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, use_bias=True)

    mlp_layer = nn.Sequential(clip_proj, nn.GELU(), intermediate_proj)
    peg_layer = nn.Sequential(nn.Conv2d(decoder_embed_dim, decoder_embed_dim, 3, 1, 1, bias=True, groups=decoder_embed_dim))

    return DocEVLDPv2Connector(
        mlp_layer=mlp_layer,
        peg_layer=peg_layer,
    )


def docev_vision_encoder(
    tile_size: int,
    patch_size: int,
    num_layers: int,
    embed_dim: int,
    hidden_dim: int,
    num_heads: int,
    activation: Callable,
    cls_output_dim: int,
    attn_bias: bool,
    use_rope: bool,
    out_indices: Optional[List[int]],
    output_cls_projection: bool ,
    max_num_tiles: int,
    in_channels: int,
    append_cls_token: bool,
) -> SiglipVisionTransformer:
    """
    Builds the vision encoder associated with the DocEV model, based on SiglipVisionTransformer architecture.

    This encoder processes image tiles into embeddings through patch extraction, positional encoding,
    and a sequence of transformer layers. It can optionally include a CLS token for classification tasks
    and supports returning intermediate layer outputs.

    The vision encoder handles spatial understanding of document images by:
    - Dividing input tiles into patches
    - Embedding patches into a consistent dimension
    - Processing embeddings through transformer layers
    - Optionally producing specialized CLS token representations

    Args:
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        embed_dim (int): The dimensionality of each patch embedding (token).
        hidden_dim (int): The dimensionality of the intermediate layer in the MLP.
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        activation (Callable): The activation function to use in the MLP layer.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        attn_bias (bool): Boolean for if to use bias in the attention module. Default True.
        use_rope (bool): If True, include 2D rope in attention in each transformer layer. Default: False
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        append_cls_token (bool): If True, adds CLS token embedding to the end of the sequence in the vision transformer.
            Default is False, which adds CLS token to the beginning of the sequence.

    Returns:
        SiglipVisionTransformer: A vision transformer model that processes image tiles into token
        representations suitable for multimodal tasks. The output shape depends on configuration, but
        typically provides tensor(s) of shape [batch_size, num_patches, embed_dim] or, with
        output_cls_projection=True, a tensor of shape [batch_size, cls_output_dim].

    Example:
        >>> vision_encoder = docev_vision_encoder(
        ...     tile_size=224,
        ...     patch_size=16,
        ...     num_layers=12,
        ...     embed_dim=768,
        ...     hidden_dim=3072,
        ...     num_heads=12,
        ...     activation=nn.GELU,
        ...     cls_output_dim=512,
        ...     attn_bias=True,
        ...     use_rope=True,
        ...     out_indices=None,
        ...     output_cls_projection=False,
        ...     max_num_tiles=4,
        ...     in_channels=3,
        ...     append_cls_token=False
        ... )
        >>> images = torch.randn(1, 3, 224, 224)
        >>> output = vision_encoder(images)

    Raises:
        ValueError: If embed_dim is not divisible by num_heads.
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    cls_projection = (
        CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim)
        if output_cls_projection
        else None
    )
    rope = (
        VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            max_num_tiles=max_num_tiles,
            dim=head_dim // 2,
            base=10_000,
            append_cls_token=append_cls_token,
        )
        if use_rope
        else None
    )

    # transformer layer
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        pos_embeddings=rope,
        attn_dropout=0.0,
        is_causal=False,
    )

    mlp = clip_mlp(
        in_dim=embed_dim,
        hidden_dim=hidden_dim,
        out_dim=embed_dim,
        activation=activation(approximate='tanh'),
    )

    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm= nn.LayerNorm(embed_dim, eps=1e-06),
        mlp_norm= nn.LayerNorm(embed_dim, eps=1e-06),
        sa_scale=None,
        mlp_scale=None,
    )

    # position embeddings
    token_pos_embedding = TokenPositionalEmbedding(
        embed_dim=embed_dim, patch_size=patch_size, tile_size=tile_size
    )

    return SiglipVisionTransformer(
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        cls_projection=cls_projection,
        out_indices=out_indices,
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        append_cls_token=append_cls_token,
    )



def lora_docev_vision_encoder(
    lora_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    *,
    # clip encoder parameters
    tile_size: int,
    patch_size: int,
    num_layers: int,
    embed_dim: int,
    hidden_dim: int,
    num_heads: int,
    activation: Callable,
    cls_output_dim: int,
    attn_bias: bool,
    use_rope: bool,
    out_indices: Optional[List[int]],
    output_cls_projection: bool ,
    max_num_tiles: int,
    in_channels: int,
    append_cls_token: bool,
    # LoRA parameters
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    **quantization_kwargs,
) -> SiglipVisionTransformer:
    """
    Builds a LoRA implementation of the vision encoder associated with the DocEV model, based on SiglipVisionTransformer architecture.

    This encoder processes image tiles into embeddings through patch extraction, positional encoding,
    and a sequence of transformer layers. It can optionally include a CLS token for classification tasks
    and supports returning intermediate layer outputs.

    The vision encoder handles spatial understanding of document images by:
    - Dividing input tiles into patches
    - Embedding patches into a consistent dimension
    - Processing embeddings through transformer layers
    - Optionally producing specialized CLS token representations

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        embed_dim (int): The dimensionality of each patch embedding (token).
        hidden_dim (int): The dimensionality of the intermediate layer in the MLP.
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        activation (Callable): The activation function to use in the MLP layer.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        attn_bias (bool): Boolean for if to use bias in the attention module. Default True.
        use_rope (bool): If True, include 2D rope in attention in each transformer layer. Default: False
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        append_cls_token (bool): If True, adds CLS token embedding to the end of the sequence in the vision transformer.
            Default is False, which adds CLS token to the beginning of the sequence.
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        SiglipVisionTransformer: A vision transformer model that processes image tiles into token
        representations suitable for multimodal tasks. The output shape depends on configuration, but
        typically provides tensor(s) of shape [batch_size, num_patches, embed_dim] or, with
        output_cls_projection=True, a tensor of shape [batch_size, cls_output_dim].

    Raises:
        ValueError: If embed_dim is not divisible by num_heads.
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    cls_projection = (
        CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim)
        if output_cls_projection
        else None
    )
    rope = (
        VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            max_num_tiles=max_num_tiles,
            dim=head_dim // 2,
            base=10_000,
            append_cls_token=append_cls_token,
        )
        if use_rope
        else None
    )

    # transformer layer
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        pos_embeddings=rope,
        attn_dropout=0.0,
        is_causal=False,
    )

    if apply_lora_to_mlp:
        mlp = lora_clip_mlp(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            activation=activation(approximate='tanh'),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            quantize_base=quantize_base,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            **quantization_kwargs,
        )
    else:
        mlp = clip_mlp(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            activation=activation(approximate='tanh'),
        )

    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm= nn.LayerNorm(embed_dim, eps=1e-06),
        mlp_norm= nn.LayerNorm(embed_dim, eps=1e-06),
        sa_scale=None,
        mlp_scale=None,
    )

    # position embeddings
    token_pos_embedding = TokenPositionalEmbedding(
        embed_dim=embed_dim, patch_size=patch_size, tile_size=tile_size
    )

    model = SiglipVisionTransformer(
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        cls_projection=cls_projection,
        out_indices=out_indices,
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        append_cls_token=append_cls_token,
    )
    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )
    return model


def docev_encoder_with_connector(
    # clip encoder parameters
    *,
    patch_size: int,
    num_heads: int,
    attn_bias: bool,
    use_rope: bool,
    activation: Callable,
    clip_embed_dim: int,
    clip_hidden_dim: int,
    clip_num_layers: int,
    clip_hidden_states: Optional[List[int]],
    cls_output_dim: int,
    append_cls_token: bool,
    output_cls_projection: bool,
    # projection parameters
    decoder_embed_dim: int,
    vision_select_layer: int,
    connector_type: Literal["ldp_v2"],
    vision_feature_select_strategy: Literal["default", "full"],
    # image parameters
    tile_size: int,
    max_num_tiles: int,
    in_channels: int,

) -> DocEVEncoderWithConnector:
    """
    Build the DocEV encoder by combining the CLIP image model with an additional
    connector component for visual-textual alignment.

    This encoder architecture handles the end-to-end process of:
    - Processing image tiles through a vision transformer
    - Extracting features from specific transformer layers
    - Connecting vision features to the appropriate dimension for the language decoder
    - Managing spatial relationships across tiles

    The encoder forms the visual understanding component of the DocEV multimodal system,
    enabling information from document images to be effectively integrated with text generation.

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        num_heads (int): The number of attention heads in each transformer layer.
        attn_bias (bool): Boolean for if to use bias in the attention module.
        use_rope (bool): If True, include 2D rope in attention in each transformer layer.
        activation (Callable): The activation function to use in transformer layers.
        clip_embed_dim (int): The dimensionality of each patch embedding in CLIP.
        clip_hidden_dim (int): The dimensionality of the hidden layer in CLIP MLPs.
        clip_num_layers (int): The number of transformer layers in CLIP.
        clip_hidden_states (Optional[List[int]]): The indices of CLIP hidden layers to return
            to return to the encoder projection head. It will return the intermediate results
            of the vision transformer layers which will be concatenated with the CLIP output
            and input into the projection head. For example, ``clip_hidden_states=[0,3]`` will
            return the embeddings before they go through the first and fourth layers.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        append_cls_token (bool): If True, adds CLS token embedding to the end of the sequence in the vision transformer.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens.
        decoder_embed_dim (int): The dimensionality of the final output embeddings for the decoder.
        vision_select_layer (int): The index of the layer to select the vision feature.
        connector_type (Literal["ldp_v2"]): The type of connector to use to bridge vision and language models.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.

    Returns:
        DocEVEncoderWithConnector: A combined encoder that processes images into features suitable for
        conditioning the language decoder. The encoder handles both the visual feature extraction
        and the necessary transformations to align with the language model's expected input format.

    Example:
        >>> encoder = docev_encoder_with_connector(
        ...     patch_size=16,
        ...     num_heads=12,
        ...     attn_bias=True,
        ...     use_rope=True,
        ...     activation=nn.GELU,
        ...     clip_embed_dim=768,
        ...     clip_hidden_dim=3072,
        ...     clip_num_layers=12,
        ...     clip_hidden_states=[0, 3, 6, 9],
        ...     cls_output_dim=512,
        ...     append_cls_token=False,
        ...     output_cls_projection=False,
        ...     decoder_embed_dim=4096,
        ...     vision_select_layer=11,
        ...     connector_type="ldp_v2",
        ...     tile_size=224,
        ...     max_num_tiles=4,
        ...     in_channels=3
        ... )
        >>> images = torch.randn(1, 3, 224, 224)
        >>> vision_features = encoder(images)
    """

    clip = docev_vision_encoder(
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=clip_embed_dim,
        hidden_dim=clip_hidden_dim,
        num_layers=clip_num_layers,
        num_heads=num_heads,
        activation=activation,
        cls_output_dim=cls_output_dim,
        attn_bias=attn_bias,
        use_rope=use_rope,
        out_indices=clip_hidden_states,
        max_num_tiles=max_num_tiles,
        in_channels=in_channels,
        append_cls_token=append_cls_token,
        output_cls_projection=output_cls_projection,
    )
    connector_type_map = {
        "ldp_v2": docev_ldp_v2_connector,
    }
    connector = connector_type_map[connector_type](
        clip_embed_dim=clip_embed_dim,
        decoder_embed_dim=decoder_embed_dim
    )

    return DocEVEncoderWithConnector(
        clip=clip,
        connector=connector,
        tile_size=tile_size,
        patch_size=patch_size,
        max_num_tiles=max_num_tiles,
        min_num_tiles=1,
        llm_hidden_size=decoder_embed_dim,
        vision_feature_layer=vision_select_layer,
        vision_feature_select_strategy=vision_feature_select_strategy
    )



def lora_docev_encoder_with_connector(
    encoder_lora: bool,
    fusion_lora: bool,
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # clip encoder parameters
    patch_size: int,
    num_heads: int,
    attn_bias: bool,
    use_rope: bool,
    activation: Callable,
    clip_embed_dim: int,
    clip_hidden_dim: int,
    clip_num_layers: int,
    clip_hidden_states: Optional[List[int]],
    cls_output_dim: int,
    append_cls_token: bool,
    output_cls_projection: bool,
    # projection parameters
    decoder_embed_dim: int,
    vision_select_layer: int,
    connector_type: Literal["ldp_v2"],
    vision_feature_select_strategy: Literal["default", "full"],
    # image parameters
    tile_size: int,
    max_num_tiles: int,
    in_channels: int = 3,
    # LoRA parameters
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    **quantization_kwargs,
) -> DocEVEncoderWithConnector:
    """
    Build the DocEV encoder with LoRA (Low-Rank Adaptation) or DoRA (Decomposed
    Low-Rank Adaptation) applied to selected components.

    This function creates a parameter-efficient version of the DocEV encoder where
    either the vision encoder, connector, or both can be adapted using LoRA/DoRA
    while keeping most of the original model parameters frozen. This approach is
    especially useful for fine-tuning large pre-trained models efficiently.

    The encoder with LoRA handles:
    - Selective application of LoRA to vision encoder and/or connector components
    - Quantization of base weights for memory efficiency (QLoRA) when specified
    - Configuration of LoRA/DoRA rank, scaling, and dropout

    Args:
        encoder_lora (bool): Whether to apply LoRA to the CLIP encoder.
        fusion_lora (bool): Whether to apply LoRA to the projection head.
        lora_attn_modules (List[LORA_ATTN_MODULES]): List of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): Whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): Whether to apply LoRA to the model's decoder and encoder output projection.
            Default: False
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
        num_heads (int): The number of attention heads in each transformer layer.
        attn_bias (bool): Boolean for if to use bias in the attention module. Default False.
        use_rope (bool): If True, include 2D rope in attention in each transformer layer. Default: True
        clip_embed_dim (int): The dimensionality of each patch embedding in CLIP.
        clip_hidden_dim (int): The dimensionality of the intermediate layer in the CLIP MLP.
        clip_num_layers (int): The number of transformer layers.
        clip_hidden_states (Optional[List[int]]): The indices of CLIP hidden layers to return.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        append_cls_token (bool): If True, adds CLS token embedding to the end of the sequence in the vision transformer.
            Default is False, which adds CLS token to the beginning of the sequence.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        decoder_embed_dim (int): The dimensionality of the final output embeddings for the decoder.
        vision_select_layer (int): The index of the layer to select the vision feature.
        connector_type (Literal["ldp_v2"]): The type of connector to use.
        tile_size (int): The size of your image tiles.
        max_num_tiles (int): The maximum number of tiles that can be processed.
        in_channels (int): The number of image input channels.
        lora_rank (int): Rank of each low-rank approximation.
        lora_alpha (float): Scaling factor for the low-rank approximation.
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base (bool): Whether to quantize base model weights or not. Default: False

    Returns:
        DocEVEncoderWithConnector: A combined encoder with LoRA/DoRA adaptations that
        processes images into features suitable for conditioning the language decoder.

    Example:
        >>> lora_encoder = lora_docev_encoder_with_connector(
        ...     encoder_lora=True,
        ...     fusion_lora=True,
        ...     lora_attn_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        ...     apply_lora_to_mlp=False,
        ...     patch_size=16,
        ...     num_heads=12,
        ...     clip_embed_dim=768,
        ...     clip_hidden_dim=3072,
        ...     clip_num_layers=12,
        ...     clip_hidden_states=[0, 6],
        ...     cls_output_dim=512,
        ...     decoder_embed_dim=4096,
        ...     vision_select_layer=11,
        ...     tile_size=224,
        ...     max_num_tiles=4,
        ...     lora_rank=8,
        ...     lora_alpha=16,
        ...     quantize_base=True
        ... )
        >>> images = torch.randn(1, 3, 224, 224)
        >>> vision_features = lora_encoder(images)
    """
    lora_options = {
        "lora_modules": lora_attn_modules,
        "apply_lora_to_mlp": apply_lora_to_mlp,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_dora": use_dora,
        "quantize_base": quantize_base,
        **quantization_kwargs,
    }

    # clip encoder
    clip_options = {
        "tile_size": tile_size,
        "patch_size": patch_size,
        "embed_dim": clip_embed_dim,
        "hidden_dim": clip_hidden_dim,
        "num_layers": clip_num_layers,
        "num_heads": num_heads,
        "activation": nn.GELU,
        "out_indices": clip_hidden_states,
        "max_num_tiles": max_num_tiles,
        "in_channels": in_channels,
        "attn_bias": attn_bias,
        "use_rope": use_rope,
        "append_cls_token": append_cls_token,
        "output_cls_projection": output_cls_projection,
        "cls_output_dim": cls_output_dim,
    }
    if encoder_lora:
        # 통합된 옵션 딕셔너리 생성 (clip_options가 우선순위)
        combined_options = {**lora_options, **clip_options}
        clip = lora_docev_vision_encoder(**combined_options)
    else:
        clip = docev_vision_encoder(**clip_options)

    # Connector
    connector_type_map = {
        "ldp_v2": lora_docev_ldp_v2_connector,
    }
    connector = connector_type_map[connector_type](
        clip_embed_dim=clip_embed_dim,
        decoder_embed_dim=decoder_embed_dim
    )

    encoder = DocEVEncoderWithConnector(
        clip=clip,
        connector=connector,
        tile_size=tile_size,
        patch_size=patch_size,
        max_num_tiles=max_num_tiles,
        min_num_tiles=1,
        llm_hidden_size=decoder_embed_dim,
        vision_feature_layer=vision_select_layer,
        vision_feature_select_strategy=vision_feature_select_strategy
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        encoder._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return encoder


def docev_solar_decoder(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    fusion_vocab_size: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
) -> TransformerDecoder:
    """
    Build the transformer decoder component of the DocEV model based on the Solar architecture.

    This decoder is responsible for generating text based on input token embeddings, optionally
    conditioned on vision features. It incorporates:
    - Token embeddings with fusion vocabulary support
    - Multi-layer transformer with self-attention and feed-forward blocks
    - Rotary positional embeddings
    - Final projection to vocabulary space

    The decoder architecture is designed to handle both standard text generation tasks and
    vision-conditioned generation through a fusion vocabulary mechanism.

    Args:
        vocab_size (int): Number of tokens in vocabulary.
        num_layers (int): Number of layers in the transformer decoder.
        num_heads (int): Number of query heads. For MHA this is also the
            number of heads for key and value.
        num_kv_heads (int): Number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): Embedding dimension for self-attention.
        max_seq_len (int): Maximum sequence length the model will be run with, as used
            by KVCache mechanism.
        fusion_vocab_size (int): Number of tokens in the fusion vocabulary for multimodal inputs.
        attn_dropout (float): Dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        rope_base (int): Base for the rotary positional embeddings. Default: 500_000
        intermediate_dim (Optional[int]): Intermediate dimension for MLP. If not specified,
            this is computed using scale_hidden_dim_for_mlp function.
        norm_eps (float): Epsilon in RMS norms for numerical stability. Default: 1e-5

    Returns:
        TransformerDecoder: A decoder model that processes token sequences and generates
        text output. When used in the DocEV architecture, this decoder can be conditioned
        on visual features to generate document-grounded text.

    Example:
        >>> decoder = docev_solar_decoder(
        ...     vocab_size=32000,
        ...     num_layers=24,
        ...     num_heads=32,
        ...     num_kv_heads=8,  # Grouped-query attention
        ...     embed_dim=4096,
        ...     max_seq_len=4096,
        ...     fusion_vocab_size=256,
        ...     attn_dropout=0.0,
        ...     rope_base=500000,
        ...     norm_eps=1e-5
        ... )
        >>> # Input token IDs of shape [batch_size, seq_len]
        >>> tokens = torch.randint(0, 32000, (1, 128))
        >>> output = decoder(tokens)
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = FusionEmbedding(vocab_size, fusion_vocab_size, embed_dim)
    output_dim = vocab_size + fusion_vocab_size if fusion_vocab_size > 0 else vocab_size
    output_proj = nn.Linear(embed_dim, output_dim, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
        # output_hidden_states=[i for i in range(num_layers)] # for debugging
    )

def lora_docev_solar_decoder(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # llama3 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    fusion_vocab_size: int,
    intermediate_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 500_000,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Build the Solar transformer decoder with LoRA (Low-Rank Adaptation) or DoRA (Decomposed
    Low-Rank Adaptation) applied to selected components.

    This function creates a parameter-efficient version of the Solar decoder for DocEV
    where specific attention components and optionally MLP layers and output projections
    can be adapted using LoRA/DoRA while keeping most parameters frozen. This approach
    enables efficient fine-tuning of large language models.

    The decoder architecture includes:
    - Selective application of LoRA to attention projection matrices
    - Optional LoRA adaptation for MLP layers
    - Optional LoRA adaptation for final output projection
    - Optional quantization of base weights (QLoRA) for memory efficiency
    - Support for multimodal fusion vocabulary

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): List of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): Whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): Whether to apply LoRA to the model's final output projection.
            Default: False
        vocab_size (int): Number of tokens in vocabulary.
        num_layers (int): Number of layers in the transformer decoder.
        num_heads (int): Number of query heads.
        num_kv_heads (int): Number of key and value heads.
        embed_dim (int): Embedding dimension for self-attention.
        max_seq_len (int): Maximum sequence length for positional encodings and KV cache.
        fusion_vocab_size (int): Number of tokens in the fusion vocabulary.
        intermediate_dim (Optional[int]): Intermediate dimension for MLP.
        attn_dropout (float): Dropout value for attention. Default: 0.0
        norm_eps (float): Epsilon in RMS norms. Default: 1e-5
        rope_base (int): Base for the rotary positional embeddings. Default: 500_000
        lora_rank (int): Rank of each low-rank approximation.
        lora_alpha (float): Scaling factor for the low-rank approximation.
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA instead of LoRA. Default: False
        quantize_base (bool): Whether to quantize base model weights. Default: False

    Returns:
        TransformerDecoder: A decoder model with LoRA/DoRA adaptations that processes
        token sequences and generates text output, potentially conditioned on
        visual features through the fusion mechanism.

    Example:
        >>> lora_decoder = lora_docev_solar_decoder(
        ...     lora_attn_modules=["q_proj", "k_proj", "v_proj"],
        ...     apply_lora_to_mlp=True,
        ...     apply_lora_to_output=True,
        ...     vocab_size=32000,
        ...     num_layers=24,
        ...     num_heads=32,
        ...     num_kv_heads=8,
        ...     embed_dim=4096,
        ...     max_seq_len=4096,
        ...     fusion_vocab_size=256,
        ...     lora_rank=16,
        ...     lora_alpha=32,
        ...     lora_dropout=0.05,
        ...     use_dora=True,
        ...     quantize_base=True
        ... )
        >>> tokens = torch.randint(0, 32000, (1, 128))
        >>> output = lora_decoder(tokens)
    """
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = lora_llama3_self_attention(
            lora_modules=lora_attn_modules,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            rope_base=rope_base,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize_base=quantize_base,
            use_dora=use_dora,
        )

        if apply_lora_to_mlp:
            mlp = lora_llama3_mlp(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
            )
        else:
            mlp = llama3_mlp(
                dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base
            )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = FusionEmbedding(vocab_size, fusion_vocab_size, embed_dim)

    # TODO: quantize_base is not applied to final output_proj currently.
    adapter_cls = DoRALinear if use_dora else LoRALinear
    output_proj = (
        adapter_cls(
            embed_dim,
            vocab_size,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, vocab_size, bias=False)
    )
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        _register_reparametrize_state_dict_hooks(model)

    return model