# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, List

from torchtune.data._messages import Message
from torchtune.datasets._sft import SFTDataset, SFTTransform
from torchtune.modules.transforms import Transform
from torchtune.data._utils import load_image


class UfxToMessages(Transform):
    """
    Construct messages from a sample formatted similarly to `UFX dataset`.

    Image placeholders are prepended to the text in the ``Message`` content. Images in the
    dataset are expected to be a list of a single PIL image, so they are simply passed through
    to the model transform with an optional column remapping if ``column_map`` is specified.

    For example, a dataset row:
    UFX format :
        {
            "id": str,
            "name": str,
            "context": List[{"role":str, "content":str}],
            "image_files": List[str],
            "meta": str,
        }

    UfxToMessages function supports two types of context format:
    context format type 1 : Content type is string, If image is present, placeholder for image is prepended to the text
    context format type 2 : Content type is list, If image is present, image is appended to the list
    For example:
    context format type 1 = [
        {
            "role": "user",
            "content": "<image> What is shown in this image?"
        },
        {
            "role": "assistant",
            "content": "There is a red stop sign in the image."
        },
    ]

    context format type 2 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "There is a red stop sign in the image."},
            ],
        },
    ]

    will be converted to::

        [
            Message(
                role = "user",
                content = [
                    {"type": "image", "content": <PIL.Image.Image>},
                    {"type": "text", "content": "What is shown in this image?"},
                ],
            ),
            Message(
                role = "assistant",
                content = [
                    {"type": "text", "content": "There is a red stop sign in the image."},
                ],
            ),
            ...
        ]

    Args:
        image_special_token (str): Special token used to mark image placeholders in text. Default is "<image>".
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "context" and "image_files"
            column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Default is None.

    Raises:
        ValueError: If ``column_map`` is provided and "context" or "image_files" is not in ``column_map``.
    """

    def __init__(
        self,
        image_special_token: str = "<image>",
        column_map: Optional[Dict[str, str]] = None,
        new_system_prompt: Optional[str] = None,
        min_image_size: Tuple[int, int] = (14, 14),
    ):
        self.image_special_token = image_special_token
        self.new_system_prompt = new_system_prompt
        self.min_image_size = min_image_size
        self._column_map = column_map or {"context": "context", "image_files": "image_files"}
        if column_map:
            for key in ("context", "image_files"):
                if key not in column_map:
                    raise ValueError(f"column_map must map '{key}' to your expected column name if specified")
        self.ufx_type = {"context": "instruction", "text": "pretraining"}

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        # Load and validate images
        images, valid_data = self._load_and_validate_images(sample)

        # Build message list
        raw_context = sample[self._column_map["context"]]
        messages = self._build_messages(raw_context, images)

        # Prepend system prompt if provided
        if self.new_system_prompt:
            system_msg = Message(
                role="system", content=self.new_system_prompt, masked=True, eot=True
            )
            messages.insert(0, system_msg)

        return {"messages": messages, "valid_data": valid_data}

    def _load_and_validate_images(self, sample: Mapping[str, Any]) -> Tuple[List[Any], bool]:
        img_files = sample[self._column_map["image_files"]]
        loaded = []
        valid_data = True
        for path in img_files:
            try:
                img = load_image(path)
                _, H, W = img.shape
                if H * W < (self.min_image_size[0] * self.min_image_size[1]):
                    print(f"Ignoring small image: {path} {img.shape}")
                    valid_data = False
                loaded.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                valid_data = False
        return loaded, valid_data

    def _build_messages(self, context: List[Dict[str, Any]], images: List[Any]) -> List[Message]:
        messages = []
        ufx_type = self.ufx_type[self._column_map["context"]]
        for entry in context:
            role, content_items = self._interpret_entry(entry, ufx_type)
            content, masked = self._assemble_content(content_items, images, role)
            msg = Message(role=role, content=content, masked=masked)
            messages.append(msg)
        return messages

    def _interpret_entry(self, entry: Dict[str, Any], ufx_type: str) -> Tuple[str, Any]:
        if ufx_type == "instruction":
            return entry["role"], entry["content"]
        # pretraining format: treat as assistant message following empty user
        return (
            "assistant",
            entry.get("text", ""),
        )

    def _assemble_content(self, content_item: Any, images: List[Any], role: str) -> Tuple[List[Dict[str, Any]], bool]:
        content_list = []
        masked = role != "assistant"

        if isinstance(content_item, str):  # format type 1
            image_count = content_item.count(self.image_special_token)
            if image_count and len(images) == image_count:
                for _ in range(image_count):
                    content_list.append({"type": "image", "content": images.pop(0)})
                content_item = content_item.replace(self.image_special_token, "")
            content_list.append({"type": "text", "content": content_item})
        elif isinstance(content_item, list):  # format type 2
            expected_imgs = sum(1 for c in content_item if c.get("type") == "image")
            if expected_imgs and len(images) == expected_imgs:
                for part in content_item:
                    if part.get("type") == "image":
                        content_list.append({"type": "image", "content": images.pop(0)})
                    else:
                        content_list.append({"type": "text", "content": part.get("content", "")})
            else:
                raise ValueError("Image count mismatch")
        else:
            raise ValueError(f"Unsupported content_item type: {type(content_item)}")

        return content_list, masked


def ufx_dataset(
    model_transform: Transform,
    *,
    subset: str = "",
    source: str = "",
    column_map: Optional[Dict[str, str]] = None,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    min_image_size: Tuple[int, int] = (14, 14),
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    Support for family of image + text datasets from UFX.
    You can specify one of the datasets using the ``subset`` argument.
    The model transform is expected to be a callable that applies pre-processing steps specific
    to a model. For multimodal datasets, this is expected to be at minimum a tokenizer and
    an image transform. The tokenizer will convert text sequences into token IDs after the dataset
    is converted to a list of :class:`~torchtune.data.Message`. The image transform will load the
    image and process it in accordance to the model's requirements.

    Here is a minimal example for illustrative purposes:

    .. code-block:: python

        from torchtune.models.llama3 import llama3_tokenizer
        from torchtune.models.clip import CLIPImageTransform
        from torchtune.modules.transforms import Transform

        class MyModelTransform(Transform):
            def __init__(
                self,
                model_name_or_pth: str,
                max_seq_len: Optional[int] = None,
            ):
                self.tokenizer = llama3_tokenizer(tokenizer_path)
                self.image_transform = CLIPImageTransform()

            def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
                tokens, mask = self.tokenizer.tokenize_messages(sample["messages"])
                images = self.image_transform(sample["images"])
                return {
                    "tokens": tokens,
                    "mask": mask,
                    "images": images,
                }

    See :class:`~torchtune.datasets.SFTDataset` for more details about model transforms and
    message transforms.

    Args:
        model_transform (Transform): model-specific transform class that takes in a sample dict and applies custom
            transforms on the keys. It should consist of at minimum two components: text tokenization (called
            on the "messages" field) and image transform (called on the "images" field). The keys returned by
            the model transform should be aligned with the expected inputs into the model.
        subset (str): name of the subset of the dataset to load. Default is empty string.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is empty string.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "context"
            and "image_files" column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        SFTDataset: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True, they are not supported for multimodal datasets yet.

    Example:
        >>> ufx_ds = ufx_dataset(model_transform=model_transform, subset="my_subset")
        >>> for batch in Dataloader(ufx_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    if packed:
        raise ValueError("Multimodal datasets don't support packing yet.")

    message_transform = UfxToMessages(
        column_map=column_map,
        new_system_prompt=new_system_prompt,
        min_image_size=min_image_size,
    )

    ds = SFTDataset(
        model_transform=model_transform,
        source=source,
        message_transform=message_transform,
        name=subset,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    return ds


def ufx_transform(
    model_transform: Optional[Transform] = None,
    context_col: str = "context",
    image_files_col: str = "image_files",
    new_system_prompt: Optional[str] = None,
) -> SFTTransform:
    """
    Support for family of image + text datasets similar to UFX format.

    This function instantiates a :class:`~torchtune.datasets.SFTTransform` only (not the dataset).
    See :func:`~torchtune.datasets.ufx_dataset` for more details.

    The model transform is expected to be a callable that applies pre-processing steps specific
    to a model. For multimodal datasets, this is expected to be at minimum a tokenizer and
    an image transform. The tokenizer will convert text sequences into token IDs after the dataset
    is converted to a list of :class:`~torchtune.data.Message`. The image transform will load the
    image and process it in accordance to the model's requirements.

    Args:
        model_transform (Optional[Transform]): model-specific transform class that takes in a sample dict and applies custom
            transforms on the keys. It should consist of at minimum two components: text tokenization (called
            on the "messages" field) and image transform (called on the "images" field). The keys returned by
            the model transform should be aligned with the expected inputs into the model. Default is None.
        context_col (str): name of the column containing the text data. Default is "context".
        image_files_col (str): name of the column containing the image data. Default is "image_files".
        new_system_prompt (Optional[str]): if specified, prepend a system message. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.

    Returns:
        :class:`~torchtune.datasets.SFTTransform` - Callable that transforms samples into UFX format.
    """
    column_map = {"context": context_col, "image_files": image_files_col}
    return SFTTransform(
        message_transform=UfxToMessages(
            column_map=column_map,
            new_system_prompt=new_system_prompt,
        ),
        model_transform=model_transform,
    )
