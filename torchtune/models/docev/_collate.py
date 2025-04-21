# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

def left_pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0,
) -> torch.Tensor:
    """
    This function is identical to :func:`torch.nn.utils.rnn.pad_sequence`, but
    instead pads a list of variable length Tensors from the left to the length
    of the longest sequence.

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (List[torch.Tensor]): list of variable length sequences.
        batch_first (bool): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise. Default False.
        padding_value (float): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise

    Example:
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6, 7])
        >>> c = torch.tensor([8, 9, 10, 11, 12])
        >>> left_pad_sequence([a, b, c], batch_first=True, padding_value=0)
        tensor([[ 0,  0,  1,  2,  3],
                [ 0,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12]])
    """
    return pad_sequence(
        map(lambda x: torch.flip(x, dims=[0]), sequences),
        batch_first=batch_first,
        padding_value=padding_value,
    ).flip(dims=[int(batch_first)])

def padded_collate_sft(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_to_multiple_of: int = 1,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        pad_to_multiple_of (int): If > 1, pad the sequence to a multiple of this number.
            This is useful for proper sharding with e.g. SequenceParallel.

    Returns:
        Dict[str, torch.Tensor]: Collated input and label tensors.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(
            labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx
        )
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )

    # Pad to multiple of N
    if pad_to_multiple_of > 1:
        input_ids = F.pad(
            input_ids,
            (0, pad_to_multiple_of - (input_ids_seq_len % pad_to_multiple_of)),
            value=padding_idx,
        )
        labels = F.pad(
            labels,
            (0, pad_to_multiple_of - (labels_seq_len % pad_to_multiple_of)),
            value=ignore_idx,
        )
    return {"tokens": input_ids.long(), "labels": labels.long()}


# TODO: Generalize this to support any type of encoder input, right now this assumes
# a specific encoder_input signature
def padded_collate_tiled_images_and_mask(
    batch: List[Dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_direction: str = "right",
    pad_max_tiles: Optional[int] = None,
    pad_max_images: Optional[int] = None,
    pad_to_multiple_of: int = 1,
) -> Dict[str, torch.Tensor]:
    """Pad a batch of text sequences and image tensors with their metadata.
    This can be used for both training and inference.

    ``batch`` is expected to be a list of sample dicts containing the following::
        - "tokens": List[int] of length text_seq_len, varies across samples
        - "labels": List[int] of length text_seq_len, varies across samples (optional for inference)
        - "encoder_input": Dict[str, Any]
            - "images": List[torch.Tensor], containing image tensors
            - "image_sizes": List[torch.Tensor], each with shape (2,) to indicate image dimensions
            - "sampling_ratio": float, sampling ratio for the images

    Shape notation:
        - c = channel dim
        - h = height dim
        - w = width dim

    This collater does the following:
        (1) Pad text sequence to the longest sequence length in the batch
        (2) Stack image tensors along with their metadata
        (3) Handle left or right padding for tokens based on pad_direction

    Args:
        batch (List[Dict[str, Any]]): A list of sample dicts containing tokens,
            labels, and encoder_input with images, image_sizes, and sampling_ratio.
        padding_idx (int): Padding index for input token ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        pad_direction (str): whether to pad entries from the left, or right. If ``pad_direction="right"``, we use
            :func:`torch.nn.utils.rnn.pad_sequence`, otherwise if ``pad_direction="left"``,
            we use :func:`torchtune.data.left_pad_sequence`. For training, we typically want to pad from the right.
            For inference, we typically want to pad from the left. Defaults to "right".
        pad_max_tiles (Optional[int]): Maximum number of tiles to pad to. If None, will pad to the largest number of tiles
            in the batch. Defaults to None.
        pad_max_images (Optional[int]): Maximum number of images to pad to. If None, will pad to the largest number of images
            in the batch. Defaults to None.
        pad_to_multiple_of (int): If > 1, pad the sequence to a multiple of this number.

    Returns:
        Dict[str, Any]: Collated data including:
            - tokens: Tensor of shape (bsz, max_seq_len)
            - labels: Tensor of shape (bsz, max_seq_len) (if provided in input)
            - encoder_input: Dict containing:
                - images: Tensor with stacked image data
                - image_sizes: Tensor with stacked image size data
                - sampling_ratio: Tensor with sampling ratios

    Raises:
        ValueError:
            If ``pad_direction`` is not one of "left" or "right", **or**
            if pad_max_tiles is set to a value less than the largest number of tiles in an image, **or**
            if ``pad_direction`` is "left" and ``pad_to_multiple_of`` is not None.

    Example:
        >>> batch = [
        ...     {
        ...         "tokens": [1, 2, 1, 3],
        ...         "labels": [4, 5, 6, 7],
        ...         "encoder_input": {
        ...             "images": [torch.ones(3, 224, 224)],
        ...             "image_sizes": [torch.tensor([224, 224])],
        ...             "sampling_ratio": 2,
        ...         },
        ...     },
        ...     {
        ...         "tokens": [1, 4],
        ...         "labels": [8, 9],
        ...         "encoder_input": {
        ...             "images": [torch.ones(3, 224, 224)],
        ...             "image_sizes": [torch.tensor([224, 224])],
        ...             "sampling_ratio": 2,
        ...         },
        ...     },
        ... ]
        >>> model_inputs = padded_collate_tiled_images_and_mask(batch=batch)
        >>> print(model_inputs["tokens"])
        tensor([[1, 2, 1, 3],
                [1, 4, 0, 0]])
        >>> print(model_inputs["labels"])
        tensor([[4, 5, 6, 7],
                [8, 9, -100, -100]])
        >>> print(model_inputs["encoder_input"]["images"].shape)
        torch.Size([2, 3, 224, 224])
        >>> print(model_inputs["encoder_input"]["image_sizes"].shape)
        torch.Size([2, 2])
        >>> print(model_inputs["encoder_input"]["sampling_ratio"].shape)
        torch.Size([2, 1])
    """
    if pad_direction not in ["left", "right"]:
        raise ValueError(
            f"pad_direction should be one of 'left' or 'right' but found {pad_direction}"
        )

    # Text tokens can be handled independently by existing collaters
    if pad_direction == "right":
        text_only = [
            {"tokens": sample["tokens"], "labels": sample["labels"]} for sample in batch
        ]
        collated_text = padded_collate_sft(
            text_only, padding_idx, ignore_idx, pad_to_multiple_of=pad_to_multiple_of
        )
    # For inference, we don't need to handle labels
    elif pad_direction == "left":
        if pad_to_multiple_of > 1:
            raise ValueError(
                f"pad_to_multiple_of={pad_to_multiple_of} is not supported for pad_direction='left'"
            )
        collated_text = {
            "tokens": left_pad_sequence(
                [torch.tensor(x["tokens"]) for x in batch],
                batch_first=True,
                padding_value=padding_idx,
            )
        }
    # (bsz, max_num_images, max_num_tiles, c, h, w)
    collated_images = torch.stack([torch.concat(sample["encoder_input"]["images"]) for sample in batch])
    # (bsz, max_num_images, 2)
    collated_image_sizes = torch.stack([torch.stack(sample["encoder_input"]["image_sizes"]) for sample in batch])
    # (bsz, max_num_images)
    collated_sampling_ratio = torch.stack([torch.tensor([sample["encoder_input"]["sampling_ratio"]]) for sample in batch])

    batch_dict = {
        "tokens": collated_text["tokens"],
        "encoder_input": {
            "images": collated_images,
            "image_sizes": collated_image_sizes,
            "sampling_ratio": collated_sampling_ratio,
        }
    }

    if "labels" in collated_text:
        batch_dict["labels"] = collated_text["labels"]

    return batch_dict