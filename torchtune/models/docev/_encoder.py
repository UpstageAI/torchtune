# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Optional, Tuple, Any
import math
import torch
import numpy as np
from torch import nn
from torchtune.modules.model_fusion import register_fusion_module
from torchtune.modules import Fp32LayerNorm
from torchtune.modules.transformer import _get_clones
from torchtune.models.docev._utils import get_anyres_image_grid_shape, unpad_image

class DocEVLDPv2Connector(nn.Module):
    """
    A connector module for DocEV (Document Enhanced Vision) models with the LDPv2 (Latent Document Perception v2) architecture.

    This module serves as a bridge between vision and language components in multimodal models,
    transforming feature representations from a vision encoder into a format suitable for language models.

    The connector performs several key transformations:
    1. Applies a Multi-Layer Perceptron (MLP) to transform feature representations
    2. Reshapes the features into a spatial grid representation
    3. Performs spatial downsampling via average pooling to reduce sequence length
    4. Applies a Position Embedding with Gating (PEG) layer to enhance spatial awareness
    5. Reshapes the output into a sequence format compatible with language models

    This connectivity architecture is specifically designed to maintain important visual relationships
    while optimizing for computational efficiency when processing document images.

    Args:
        mlp_layer (nn.Module): The Multi-Layer Perceptron module that transforms the input features.
            This typically projects features to the desired embedding dimension.
        peg_layer (nn.Module): The Position Embedding with Gating convolutional layer applied
            after pooling. Enhances features with positional information while maintaining the
            ability to flow gradients efficiently.

    Shape:
        - Input: `x` - Tensor of shape [batch_size, num_images, num_tiles, num_features_per_tile, embed_dim]
        - Input: `sampling_ratio` - Integer controlling the downsampling factor
        - Output: Tensor of shape [batch_size*num_images, tiles*pooled_hw, embed_dim]
          where pooled_hw is the spatial dimension after pooling

    Examples:
        >>> mlp = nn.Linear(768, 768)
        >>> peg = nn.Conv2d(768, 768, kernel_size=3, padding=1, groups=768)
        >>> connector = DocEVLDPv2Connector(mlp, peg)
        >>> x = torch.randn(2, 3, 4, 196, 768)  # [batch, images, tiles, features, dim]
        >>> out = connector(x, sampling_ratio=2)
        >>> out.shape  # [batch*images, tiles*pooled_features, dim]
    """
    def __init__(
        self,
        mlp_layer: nn.Module,
        peg_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.mlp = mlp_layer
        self.peg = peg_layer

    def forward(
        self,
        x: torch.Tensor,
        sampling_ratio: int,
    ) -> torch.Tensor:
        """
        Forward pass of the connector.

        Args:
            x (torch.Tensor): Input tensor representing image features with shape
                `[b x i x t x e x d]`.
            sampling_ratio (int): The factor by which to downsample the spatial dimensions
                of the features using average pooling. The kernel size and stride for pooling
                will be equal to this value.

        Returns:
            torch.Tensor: Output tensor reshaped into a sequence of embeddings with shape
                `[bsz*imgs, sequence_length, dim]`, where `sequence_length` is
                `tiles * pooled_hw` (pooled height * pooled width).

        Notation used for tensor shapes:
            - b: batch size
            - i: number of images per sample
            - t: number of tiles per image
            - e: number of features (embeddings) per tile (e.g., patch embeddings).
                 Must be a perfect square.
            - d: embedding dimension
            - pooled_hw: number of features per tile after pooling.
        """
        x = self.mlp(x)
        bsz, imgs, tiles, num_feature_per_tile, dim = x.shape
        # check if the number of tiles is a square number
        assert num_feature_per_tile ** 0.5 == int(num_feature_per_tile ** 0.5), "The number of tiles must be a square number"
        side_len = int(num_feature_per_tile ** 0.5)
        x = x.transpose(3,4).view(bsz*imgs*tiles, dim, side_len, side_len)

        # reproducibility: avg_pool2d (deterministic) instead of adaptive_avg_pool2d (non-deterministic)
        x = torch.nn.functional.avg_pool2d(
            x, kernel_size=sampling_ratio, stride=sampling_ratio, padding=0
        ) # (bsz*imgs*tiles, dim, pooled_h, pooled_w)
        x = self.peg(x) + x
        # (bsz*imgs*tiles, dim, pooled_h, pooled_w) -> (bsz*imgs*tiles, pooled_hw, dim)
        x = x.flatten(2).transpose(1, 2)
        x = x.view(bsz*imgs, -1, dim) # (bsz*imgs, tiles*pooled_hw, dim)
        return x

class DocEVEncoderWithConnector(nn.Module):
    """
    Document Enhanced Vision (DocEV) encoder with connector for multimodal document understanding.

    This class combines a pretrained vision encoder (typically a CLIP or SigLIP model) with a
    learnable connector module that bridges vision and language representations. The model is
    specifically designed for processing document images through a multi-scale, tile-based approach
    that maintains document structure while enabling efficient processing of high-resolution inputs.

    The encoder works by:
    1. Processing image tiles through the vision encoder to extract features
    2. Obtaining hidden states from a specific layer of the vision model
    3. Using the connector to transform these features into a format suitable for language models
    4. Reshaping and rearranging the features to maintain spatial relationships
    5. Adding special token embeddings (like newlines) for document structure awareness

    This architecture is particularly effective for tasks requiring understanding of both
    visual elements and textual content in documents, such as document VQA, information extraction,
    and multimodal reasoning.

    Args:
        clip (nn.Module): CLIP or similar vision encoder model that processes image tiles and
            produces feature representations. Despite the name, this can be any vision encoder
            that returns hidden states compatible with the connector.

        connector (nn.Module): Connector module that transforms vision features into a format
            compatible with language models. Takes embeddings with dimension `encoder_dim` as
            input and outputs embeddings of size `llm_hidden_size`. Typically a
            DocEVLDPv2Connector or similar architecture.

        tile_size (int): The size (height/width in pixels) of each square image tile. Documents
            are divided into tiles of this size during preprocessing for more efficient processing
            of high-resolution inputs.

        patch_size (int): The size of each patch within a tile. Patches are the smallest unit
            processed by the vision transformer, where `tile_size/patch_size` determines the grid
            resolution of features per tile.

        max_num_tiles (int): The maximum number of tiles allowed per image. Used to constrain
            the possible resolutions when processing variable-sized images. Helps control
            computational complexity.

        min_num_tiles (int): The minimum number of tiles required per image. Used along with
            max_num_tiles to define valid tiling configurations.

        llm_hidden_size (int): The hidden dimension size of the language model that will
            consume these features. The connector will ensure output features match this
            dimension.

        vision_feature_layer (int, optional): The specific layer of the vision model from which
            to extract features. Default is -1 (the last layer). Earlier layers capture more
            low-level visual features while later layers have more semantic information.

        vision_feature_select_strategy (Literal["default", "full"], optional): Strategy for
            selecting features from the vision model:
            - "default": Ignores the CLS token (index 0) and only uses patch tokens
            - "full": Uses all tokens including the CLS token
            Default is "full".

    Shape:
        - Input:
            - images: Tensor of shape [batch_size, num_images, num_tiles, channels, tile_height, tile_width]
            - image_sizes: Tensor of shape [batch_size, num_images, 2] containing actual (H, W) sizes
            - sampling_ratio: Tensor of shape [1] indicating downsampling factor for features

        - Output:
            - Tensor of shape [total_feature_length, llm_hidden_size] containing all processed
              visual features ready to be consumed by a language model

    Example:
        >>> # Initialize components
        >>> clip_model = SiglipVisionTransformer(...)
        >>> connector = DocEVLDPv2Connector(...)
        >>> docev = DocEVEncoderWithConnector(
        ...     clip=clip_model,
        ...     connector=connector,
        ...     tile_size=400,
        ...     patch_size=40,
        ...     max_num_tiles=64,
        ...     min_num_tiles=1,
        ...     llm_hidden_size=4096,
        ...     vision_feature_layer=-1,
        ...     vision_feature_select_strategy="full"
        ... )
        >>>
        >>> # Process images
        >>> images = torch.randn(2, 3, 4, 3, 400, 400)  # [batch, imgs, tiles, channels, height, width]
        >>> image_sizes = torch.tensor([[[1200, 800]], [[1600, 1200]]])  # actual sizes
        >>> sampling_ratio = torch.tensor([2])  # downsample by factor of 2
        >>>
        >>> visual_features = docev(images, image_sizes, sampling_ratio)
    """

    def __init__(
        self,
        clip: nn.Module,
        connector: nn.Module,
        tile_size: int,
        patch_size: int,
        max_num_tiles: int,
        min_num_tiles: int,
        llm_hidden_size: int,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: Literal["default", "full"] = "full",
    ) -> None:
        super().__init__()
        self.clip = clip
        self.connector = connector
        register_fusion_module(self.connector)
        self.tile_size = tile_size
        self.patch_size = patch_size
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        embed_std = 1 / math.sqrt(llm_hidden_size)
        self.image_newline = nn.Parameter(torch.randn(llm_hidden_size) * embed_std)
        tile_range = range(max_num_tiles + 1)
        self.possible_resolutions = [
            [tile_size * i, tile_size * j]
            for i in tile_range
            for j in tile_range
            if min_num_tiles <= i * j and i * j <= max_num_tiles
        ]



    def pack_image_features(self, image_features, image_sizes, sampling_ratio):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_images, num_tiles, vision_feature_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all tiles.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            sampling_ratio (int): The ratio of the sampling of the hidden states.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                down_scaled_feature_len = self.tile_size // self.patch_size // sampling_ratio
                if down_scaled_feature_len * down_scaled_feature_len != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                # num_patch_height = best_resolution_height // tile_size
                # num_patch_width = best_resolution_width // tile_size
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(image_sizes[image_idx], self.possible_resolutions, self.tile_size)
                image_feature = image_feature.view(num_patch_height, num_patch_width, down_scaled_feature_len, down_scaled_feature_len, -1) # (nph, npw, h, w, d)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # (d, nph, h, npw, w)
                image_feature = image_feature.flatten(1, 2).flatten(2, 3) # (d, nph*h, npw*w)
                image_feature = unpad_image(image_feature, image_sizes[image_idx]) # (d, H', W')

                if self.image_newline is not None:
                    # image_newline : (d, H', 1)
                    # image_feature : (d, H', W')
                    # concat them together: (d, H', W' + 1)
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1) # (H' * (W' + 1), d)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0) # (single image feature length, d)
            else:
                raise ValueError("All image features should have at least one feature in LLava-Next architecture.")
                # TODO : check if this is correct
                image_feature = image_feature[0] # (down_scaled_feature_height * down_scaled_feature_width, d)
                if self.image_newline is not None:
                    # image_newline : (1, d)
                    # image_feature : (down_scaled_feature_height * down_scaled_feature_width, d)
                    # concat them together: (down_scaled_feature_height * down_scaled_feature_width + 1, d)
                    image_feature = torch.cat((image_feature, self.image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature) # (single image feature length, d) * n
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0) # all_image_feature_len, d
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens


    def forward(
        self, images: torch.Tensor, image_sizes: torch.Tensor, sampling_ratio: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b, i, t, c, w, h]
            image_sizes (torch.Tensor): The size of each image [b, i, 2].
            sampling_ratio (torch.Tensor): The ratio of the sampling of the vision feature. [1]

        Returns:
            Tensor: output tensor of a sequence of embedings [b*s, d]
                where sequence length is num_imgs*num_tiles+num_embeds

         Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*pooled_hw
            - d: embed dim
        """

        bsz, imgs, tiles, channel, tile_size_h, tile_size_w = images.shape
        _, hidden_states = self.clip(images)

        x = hidden_states[self.vision_feature_layer] # [bsz, n_imgs, num_tiles, num_patches_per_tile, embed_dim]

        if self.vision_feature_select_strategy == "default":
            x = x[:, :, :, 1:, :] # ignore the CLS token
        if isinstance(sampling_ratio, torch.Tensor):
            sampling_ratio = sampling_ratio.item()
        x = x.contiguous()
        x = self.connector(x, sampling_ratio) # [bsz*imgs, tiles*pooled_hw, dim]
        dim = x.shape[-1]
        x = x.view(bsz*imgs, tiles, -1, dim)
        image_sizes = image_sizes.view(bsz*imgs, 2)
        # Insert image newline and flatten all image features
        x, _ = self.pack_image_features(
            x,
            image_sizes,
            sampling_ratio
        ) # [all_feat_len, dim]

        return x

class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, embed_dim: int, tile_size: int, patch_size: int) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        n_tokens_per_tile = patch_grid_size**2
        scale = embed_dim**-0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((n_tokens_per_tile, embed_dim))
        )

    def forward(self, x: torch.Tensor, *args: Tuple[Any]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): torch.Tensor with shape (..., n_tokens_per_tile, embed_dim)
            *args (Tuple[Any]): Optional args.

        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding

class SiglipVisionTransformer(nn.Module):
    """
    Vision Transformer model based on the SigLIP architecture for processing image tiles through transformer layers.

    This module implements a Vision Transformer that:
    1. Divides input image tiles into patches using a convolutional layer
    2. Embeds patches into tokens and applies positional encoding
    3. Optionally adds a CLS token for each tile
    4. Processes tokens through a sequence of transformer layers
    5. Optionally returns hidden states from specified layers
    6. Optionally projects the CLS token for downstream tasks

    The model is designed to handle batched inputs with multiple images per sample and multiple tiles per image.

    Architecture details:
    - Input images are divided into fixed-size patches using a convolutional layer
    - Each patch becomes a token with dimension `embed_dim`
    - Positional embeddings are added to provide spatial information
    - A CLS token can be prepended or appended to the sequence of patch tokens
    - The tokens are processed through `num_layers` transformer layers
    - Layer normalization is applied to the final output
    - Optionally, only the CLS token can be projected and returned

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each. Must be an integer divisor of `tile_size`.

        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
            Must be divisible by `patch_size` to ensure an integer number of patches per tile.

        num_layers (int): The number of transformer layers used to process the token sequence.
            Each layer typically includes self-attention and feed-forward components.

        embed_dim (int): The dimensionality of each patch embedding (token). This determines the
            size of the feature vector representing each patch and the dimensionality of the
            transformer layers.

        layer (nn.Module): The transformer layer module to be replicated `num_layers` times.
            Should accept input of shape (batch_size * n_tiles, n_tokens, embed_dim) and
            return output of the same shape.

        token_pos_embedding (nn.Module): The token positional embedding module that adds
            positional information to the patch tokens. May also add CLS tokens depending
            on implementation.

        cls_projection (Optional[nn.Module]): The CLS projection module. It should take an input tensor
            of shape (bsz * n_tiles, n_tokens, embed_dim) and output a tensor of shape
            (bsz * n_tiles, cls_output_dim). If provided, only the CLS token projection will be
            outputted, instead of all tokens. Useful for classification or feature extraction tasks.

        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers. This is
            useful for feature extraction at different levels of abstraction.

        in_channels (int): The number of image input channels (e.g., 3 for RGB, 1 for grayscale).
            Default is 3.

        append_cls_token (bool): If True, adds CLS token to the end of the sequence.
            Default is False, which adds CLS token to the beginning of the sequence.
            This parameter affects how the CLS token is added and may impact models
            that expect the CLS token in a specific position.

    Shapes:
        - Input: (bsz, n_imgs, n_tiles, in_channels, tile_size, tile_size)
            - bsz: Batch size
            - n_imgs: Number of images per sample
            - n_tiles: Number of tiles per image
            - in_channels: Number of input channels (e.g., 3 for RGB)
            - tile_size: Height and width of each tile

        - Output: Tuple containing:
            1. Final output tensor with shape:
               - If cls_projection is None: (bsz, n_imgs, n_tiles, n_tokens, embed_dim)
               - If cls_projection is provided: Shape depends on cls_projection implementation
            2. List of hidden states from layers specified in out_indices, each with shape:
               (bsz, n_imgs, n_tiles, n_tokens, embed_dim)

    Raises:
        ValueError:
            - If `tile_size` is not greater than 0
            - If `patch_size` is not greater than 0
            - If `out_indices` is provided and `len(out_indices)` is greater than `num_layers`
            - Implicitly if `tile_size` is not divisible by `patch_size`

    Examples:
        >>> # Initialize the required modules (simplified example)
        >>> patch_size, tile_size = 40, 400
        >>> embed_dim, num_layers = 1024, 12
        >>> transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=16)
        >>> pos_embed = TokenPositionalEmbedding(embed_dim, tile_size, patch_size)
        >>> cls_embed = CLSEmbedding(embed_dim)
        >>> token_pos_embedding = nn.Sequential(pos_embed, cls_embed)
        >>>
        >>> # Create the vision transformer
        >>> model = SiglipVisionTransformer(
        ...     patch_size=patch_size,
        ...     tile_size=tile_size,
        ...     num_layers=num_layers,
        ...     embed_dim=embed_dim,
        ...     layer=transformer_layer,
        ...     token_pos_embedding=token_pos_embedding,
        ...     out_indices=[3, 6, 9],
        ...     in_channels=3
        ... )
        >>>
        >>> # Create input tensor: 2 samples, 1 image per sample, 2 tiles, RGB, 400x400 pixels
        >>> images = torch.rand(2, 1, 2, 3, 400, 400)
        >>>
        >>> # Forward pass
        >>> outputs, hidden_states = model(images)
        >>>
        >>> # Expected output shapes
        >>> # outputs: (2, 1, 2, 101, 1024) - batch, images, tiles, tokens (100 patches + 1 CLS), features
        >>> # hidden_states: list of 3 tensors, each with shape (2, 1, 2, 101, 1024)
    """

    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        num_layers: int,
        embed_dim: int,
        layer: nn.Module,
        token_pos_embedding: nn.Module,
        cls_projection: Optional[nn.Module] = None,
        out_indices: Optional[List[int]] = None,
        in_channels: int = 3,
        append_cls_token: bool = False,
    ) -> None:
        super().__init__()

        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if out_indices and (len(out_indices) > num_layers):
            raise ValueError(
                f"len(out_indices) must be <= num_layers. Got {out_indices=} and {num_layers=}"
            )

        # constants
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.out_indices = out_indices
        if not out_indices:
            self.out_indices = []

        # input modules
        self.token_pos_embedding = token_pos_embedding

        self.cls_projection = cls_projection
        self.layers = _get_clones(layer, num_layers)

        # other modules
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding="valid",
        )

        self.ln_post = Fp32LayerNorm(embed_dim)

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile + 1  # +1 for CLS token

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension n_imgs to the input. This is useful when a single
        sample contains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, n_imgs=2, num_tiles, n_channels, tile_size, tile_size).

        Notice that to batch it, you will have to pad n_imgs to max_n_imgs and num_tiles to max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, n_imgs, n_tiles, n_channels, tile_size, tile_size).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple (x, hidden_states):
                - x: The output tensor from the final layer.
                    - If `cls_projection` is `None`, the shape is (bsz, n_imgs, n_tiles, n_tokens, embed_dim).
                    - If `cls_projection` is provided, it returns the projected representation of the CLS token,
                      and the shape might be, for example, (bsz, n_imgs, n_tiles, cls_output_dim).
                      (The exact shape depends on the cls_projection module).
                    - Here, `n_tokens` is the number of patches (tokens) per tile, potentially plus 1
                      if a CLS token is included (e.g., by the `token_pos_embedding` module).
                - hidden_states: A list of hidden states from the layers specified in `out_indices`.
                  Each tensor has the shape (bsz, n_imgs, n_tiles, n_tokens, embed_dim).

        Examples:

            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> # Assume we are using SiglipVisionTransformer. Actual initialization requires layer, token_pos_embedding, etc.
            >>> # from torchtune.models.docev._encoder import SiglipVisionTransformer
            >>>
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 400
            >>> patch_size=40
            >>> embed_dim = 32
            >>> num_layers = 6
            >>> # Need to define other required modules (layer, token_pos_embedding, etc.)
            >>> # Setting only necessary args for example. Full module setup needed for actual use.
            >>> # model = SiglipVisionTransformer(
            ... #           patch_size=patch_size,
            ... #           tile_size=tile_size,
            ... #           num_layers=num_layers,
            ... #           embed_dim=embed_dim,
            ... #           layer=..., # Actual Transformer layer module
            ... #           token_pos_embedding=..., # Actual token/position embedding module
            ... #           out_indices=[1, 2, 3, 4, 5],
            ... #           in_channels=num_channels,
            ... # )
            >>>
            >>> # create a random image
            >>> image = torch.rand(num_channels, image_size[0], image_size[1])
            >>>
            >>> # (num_tiles, nch, h, w) -> (2, 3, 400, 400)
            >>> tile_cropped_image = tile_crop(image, tile_size)
            >>>
            >>> # make it a batch of 1 image
            >>> batch_image = tile_cropped_image.unsqueeze(0) # (bsz=1, num_tiles=2, nch=3, h=400, w=400)
            >>>
            >>> # make it have only 1 image per sample
            >>> batch_image = batch_image.unsqueeze(1) # (bsz=1, n_imgs=1, num_tiles=2, nch=3, h=400, w=400)
            >>>
            >>> # Run the model (requires actual model instance)
            >>> # x, hidden_states = model(images=batch_image)
            >>>
            >>> # Expected output shape (assuming cls_projection=None and CLS token is included)
            >>> # n_tokens = (tile_size // patch_size)**2 + 1 = (400 // 40)**2 + 1 = 100 + 1 = 101
            >>> # print(x.shape)
            # torch.Size([1, 1, 2, 101, 32]) # (bsz, n_imgs, num_tiles, n_tokens, embed_dim)
            >>>
            >>> # print(len(hidden_states))
            # 5
            >>> # print(hidden_states[0].shape)
            # torch.Size([1, 1, 2, 101, 32]) # (bsz, n_imgs, num_tiles, n_tokens, embed_dim)
        """
        hidden_states = []

        # parse inputs
        bsz, n_imgs, n_tiles, nch, w, h = images.shape
        bsz_and_n_imgs = bsz * n_imgs

        images = images.reshape(bsz_and_n_imgs * n_tiles, nch, w, h)

        # patch embeddings (tokens)
        # A tile becomes a grid of patch_grid_size X patch_grid_size patches
        # these patches are flatenned, and called tokens from here on.

        # out: (bsz * n_imgs * n_tiles, embed_dim, patch_grid_size, patch_grid_size)
        # type cast image to conv weight dtype
        x = self.conv(images.to(self.conv.weight.dtype))

        # out: (bsz * n_imgs, n_tiles, n_tokens, embed_dim)
        # Here n_tokens is the number of patches (self.patches_per_tile) before CLS token is added.
        x = x.reshape(bsz_and_n_imgs, n_tiles, -1, self.patches_per_tile).permute(
            0, 1, 3, 2
        )
        # Assume token_pos_embedding module adds CLS token and applies positional embeddings.
        # After application, x shape becomes (bsz_and_n_imgs, n_tiles, n_tokens_with_cls, embed_dim).
        x = self.token_pos_embedding(x)
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape # n_tokens now potentially includes CLS

        # transformer with optional hidden layer outputs
        x = x.reshape(bsz_and_n_imgs*n_tiles, n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.layers):
            if layer_idx in self.out_indices:
                # Reshape back to (bsz, n_imgs, n_tiles, ...) when saving hidden_states
                h = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
                hidden_states.append(h)
            x = transformer_layer(x)

        if layer_idx in self.out_indices:
             # Can also save hidden_states after the last layer's output
            h = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
            hidden_states.append(h)

        # norm
        x = self.ln_post(x) # shape: (bsz_and_n_imgs*n_tiles, n_tokens, embed_dim)

        # reshape output
        # Reshape final output x back to (bsz, n_imgs, n_tiles, ...)
        x = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)

        # cls token projection. n_tokens becomes 1 (or cls_output_dim)
        if self.cls_projection:
            # Applying cls_projection might change the shape of x
            x = self.cls_projection(x)

        return x, hidden_states


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile in an image.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
        append_cls_token (bool): If True, adds CLS token to the end of the sequence.
            Default is False, which adds CLS token to the beginning of the sequence.
    """

    def __init__(self, embed_dim: int, append_cls_token: bool = False) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.weight = nn.Parameter(scale * torch.randn(embed_dim))
        self.append_cls_token = append_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # add 1 CLS token to every tile
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_n_imgs, n_tiles, 1, embed_dim)
        return (
            torch.cat([x, cls_emb], dim=2)
            if self.append_cls_token
            else torch.cat([cls_emb, x], dim=2)
        )
