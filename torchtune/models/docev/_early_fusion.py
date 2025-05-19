# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion._fusion_utils import get_fusion_params
from torchtune.modules.peft._utils import get_adapter_params, set_trainable_params


class EarlyFusionModel(nn.Module):
    """EarlyFusion is a type of fused model architecture where pretrained encoder(s) are combined
    with a pretrained decoder (LLM) at the model input and not in internal layers. This is a popular architecture
    for multimodal models, with a full overview available in `The Evolution of Multimodal Model Architectures
    <https://arxiv.org/abs/2405.17927>`_. This module works both for decoders in which the encoder tokens are
    inside the vocab and outside the vocab.

    This module has the same methods and forward signature as :class:`~torchtune.modules.TransformerDecoder` and can be used
    interchangeably where :class:`~torchtune.modules.TransformerDecoder` is. It combines the encoder with the decoder as a
    single module for checkpointing and finetuning. It is expected that the encoder and decoder
    are already defined with any extra learnable ``fusion_params``: learnable parameters to help
    adapt the pre-trained encoder to the pre-trained decoder.

    Note: Once the decoder is wrapped in this module, the decoder's ``tok_embeddings`` module is moved
    to the parent EarlyFusionModel's ``tok_embeddings``. You should not forward pass the decoder individually.
    Instead, use EarlyFusionModel's forward pass with ``encoder_input=None`` to get decoder-only outputs.
    State dicts will automatically be updated on save and load to account for this change.

    Example:
        >>> # decoder is a text-only TransformerDecoder (e.g. llama3_8b) with no modifications
        >>> decoder = llama3_8b()
        >>>
        >>> # encoder is pre-trained encoder (e.g. clip_vit_224) with an added projection head
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)
        >>>
        >>> # EarlyFusionModel combines the encoder and decoder
        >>> model = EarlyFusionModel(decoder, encoder, image_token_id=128256)
        >>>
        >>> # Load full fused checkpoints
        >>> model.load_state_dict(...)
        >>>
        >>> # Forward pass
        >>> encoder_input = {...}
        >>> output = model(tokens, mask=mask, encoder_input=encoder_input, input_pos=input_pos)
        >>>
        >>> # Forward pass decoder only
        >>> output = model(tokens, mask=mask, input_pos=input_pos)

    Args:
        decoder (TransformerDecoder): decoder module
        encoder (nn.Module): encoder module.
        image_token_id (int): special token ID indicating where in the text sequence
            the encoder embedding outputs should be injected.
        decoder_trainable (bool): whether to train or freeze the decoder. Default is False.
        encoder_trainable (bool): whether to train or freeze the encoder. Default is False.
        fusion_trainable (bool): whether to train the fusion parameters. Default is True.

    Raises:
        ValueError: if ``encoder`` and ``encoder_trainable`` keys do not match
    """

    def __init__(
        self,
        decoder: TransformerDecoder,
        encoder: nn.Module,
        image_token_id: int,
        decoder_trainable: bool = False,
        encoder_trainable: bool = False,
        fusion_trainable: bool = True,
    ):
        super().__init__()


        self.decoder = decoder
        self.encoder = encoder
        self.image_token_id = image_token_id
        self.encoder_trainable = encoder_trainable

        trainable_params = set()
        if encoder_trainable:
            adapter_params = get_adapter_params(self.encoder)
            if adapter_params:
                # lora
                trainable_params |= {
                    f"encoder.{n}" for n, p in adapter_params.items()
                }
            else:
                # full
                trainable_params |= {
                    f"encoder.{n}" for n, p in self.encoder.named_parameters()
                }
        if decoder_trainable:
            adapter_params = get_adapter_params(self.decoder)
            if adapter_params:
                # lora
                trainable_params |= {
                    f"decoder.{n}" for n, p in adapter_params.items()
                }
            else:
                # full
                trainable_params |= {
                    f"decoder.{n}" for n, p in self.decoder.named_parameters()
                }
        if fusion_trainable:
            trainable_params |= set(get_fusion_params(self))
        else:
            trainable_params -= set(get_fusion_params(self))

        set_trainable_params(self, trainable_params)


    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Used to save memory in combination with :class:`~torchtune.modules.loss.CEWithChunkedOutputLoss`.
        This should be called before the first forward pass, in the recipe."""
        self.decoder.set_num_output_chunks(num_output_chunks)

    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Setup key value caches for attention calculation.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
        """
        self.decoder.setup_caches(batch_size, dtype)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """Reset the key value caches."""
        self.decoder.reset_caches()

    def _decoder_embed(self, tokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed the text-only tokens with the decoder's tok_embeddings"""
        encoder_token_ids = torch.tensor([self.image_token_id], device=tokens.device)
        # [bsz, seq_len], True indicates the token is not an encoder special token
        is_text = ~torch.isin(tokens, encoder_token_ids)
        text_tokens = torch.masked_select(tokens, is_text)
        # [num_text, embed_dim]
        text_embeds = self.decoder.tok_embeddings(text_tokens)
        return is_text, text_embeds

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[Dict[str, Any]] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Note: This module assumes that there will be enough encoder inputs (i.e., total number of images in the batch)
        for the number of encoder tokens in the batch.

        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
                with shape ``[b x s x s]``. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            encoder_input (Optional[Dict[str, Any]]): Optional input kwargs for the encoder.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
            **kwargs (Dict[str, Any]): additional keyword arguments. This is solely used to match the
                :class:`~torchtune.modules.TransformerDecoder` forward and does not have any effect.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            torch.Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """

        # bsz, seq_len = tokens.shape
        fused_embeds = self.decoder.tok_embeddings(tokens)
        embed_dim = fused_embeds.shape[-1]
        if encoder_input is not None:
            # [bsz, seq_len, 1]
            encoder_mask = (tokens == self.image_token_id).unsqueeze(-1)

            # [bsz, num_encoder_tokens, embed_dim]
            encoder_embeds = self.encoder(**encoder_input)

            # [bsz * num_encoder_tokens, embed_dim]
            encoder_embeds = encoder_embeds.view(-1, embed_dim)
            if encoder_embeds.shape[0] != encoder_mask.sum():
                raise ValueError(
                    f"encoder_embeds.shape[0] != encoder_mask.sum(): {encoder_embeds.shape[0]} != {encoder_mask.sum()}"
                )

            # At locations where encoder token is found, replace with encoder embedding
            fused_embeds = fused_embeds.masked_scatter(encoder_mask, encoder_embeds)

        output = self.decoder(
            tokens=None, mask=mask, input_pos=input_pos, input_embeds=fused_embeds
        )
        return output
