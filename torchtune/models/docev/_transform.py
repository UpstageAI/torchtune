# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional, Tuple, Literal

from torchtune.data import Message

from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer

from transformers import (
    AddedToken,
    AutoTokenizer,
    LlavaNextImageProcessor,
)
import torch
import random
from torchtune.models.docev._utils import get_encoder_output_feature_size
class DocEVTransform(ModelTokenizer, Transform):
    """
    Transformation class for Document-Enhanced Vision (DocEV) models.

    This class handles the preprocessing of multimodal inputs (text and images) for DocEV models.
    It performs tokenization of text using the model's tokenizer and transforms images into
    appropriate representations for the vision encoder, supporting variable-resolution images
    through tiling and strategic patch selection.

    The transform handles:
    1. Text tokenization with support for chat templates
    2. Image preprocessing and tiling for images
    3. Combined multimodal token sequence generation
    4. Integration of image tokens into the text sequence

    Args:
        model_name_or_path (str):
            Path to the pretrained model or model identifier from huggingface.co/models.
        image_token (str):
            Special token used to represent images in the input text.
        tile_size (int):
            Size of the image tiles in pixels.
        patch_size (int):
            Size of the image patches in pixels.
        max_num_tiles (int):
            Maximum number of tiles allowed for an image.
        min_num_tiles (int, optional):
            Minimum number of tiles required for an image. Defaults to 1.
        num_additional_image_tokens (int, optional):
            Number of additional image tokens (e.g., for CLS token). Defaults to 1.
        vision_feature_select_strategy (Literal["default", "full"], optional):
            Strategy for selecting vision features. Defaults to "default".
        sampling_ratio (List[int], optional):
            List of sampling ratios for downsampling. Defaults to [2, 3].
        apply_random_sampling_ratio (bool, optional):
            Whether to randomly select from sampling_ratio. Defaults to True.
        max_seq_len (Optional[int], optional):
            Maximum sequence length for tokenization. Defaults to None.
        chat_template (Optional[str], optional):
            Template for formatting chat messages. Defaults to None.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        image_token: str,
        stop_tokens: List[str],
        tile_size: int,
        patch_size: int,
        max_num_tiles: int,
        min_num_tiles: int = 1,
        vision_feature_select_strategy: Literal["default", "full"] = "default",
        sampling_ratio: List[int] = [2, 3],
        apply_random_sampling_ratio: bool = True,
        max_seq_len: Optional[int] = None,
        chat_template: Optional[str] = None,
    ):
        self.patch_size = patch_size
        self.tile_size = tile_size

        self.vision_feature_select_strategy = vision_feature_select_strategy
        # Init Image Processor
        tile_range = range(max_num_tiles + 1)
        # A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].
        self.possible_resolutions = [
            [tile_size * i, tile_size * j]
            for i in tile_range
            for j in tile_range
            if min_num_tiles <= i * j and i * j <= max_num_tiles
        ]
        self.image_processor = LlavaNextImageProcessor.from_pretrained(
                    model_name_or_path,
                    crop_size={"height": tile_size, "width": tile_size},
                    image_grid_pinpoints=self.possible_resolutions,
                    size={"shortest_edge": tile_size},
                )

        # Init Tokenizer
        self.max_seq_len = max_seq_len
        self.image_token = image_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._base_vocab_size = self.tokenizer.vocab_size
        # Add image_token. If image_token is already in the tokenizer, it will not be added again.
        self.tokenizer.add_tokens(AddedToken(self.image_token, special=True, normalized=False), special_tokens=True)
        # Check padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.pad_id = self.tokenizer.pad_token_id
        self.stop_tokens = [self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop_tokens]

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if chat_template is None:
            self.chat_template = self.tokenizer.chat_template
        else:
            self.chat_template = chat_template

        # Sampling Ratio args
        self.apply_random_sampling_ratio = apply_random_sampling_ratio
        self.sampling_ratio = sampling_ratio

    @property
    def base_vocab_size(self) -> int:
        """
        Get the base vocabulary size of the tokenizer before adding any special tokens.

        Returns:
            int: The base vocabulary size.
        """
        return self._base_vocab_size

    @property
    def vocab_size(self) -> int:
        """
        Get the current vocabulary size of the tokenizer, including any added special tokens.

        Returns:
            int: The current vocabulary size.
        """
        return self.tokenizer.vocab_size

    def encode(
        self,
        text: str,
        **kwargs: Dict[str, Any]
    ) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            List[int]: The encoded list of token ids.
        """
        return self.tokenizer.encode(text=text, **kwargs)['input_ids']

    def decode(
        self,
        token_ids: List[int],
        **kwargs: Dict[str, Any]
    ) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            str: The decoded text.
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    def message_to_chat_item(
        self,
        message: Message,
        sampling_ratio: int,
    ) -> Dict[str, Any]:
        """
        Convert a Message object to a chat item format suitable for the model.

        This method processes text and image content from the message, transforms images
        using the image processor, and calculates necessary image metadata.

        Args:
            message (Message): The message to convert, containing text and/or image content.
            sampling_ratio (int): The sampling ratio to use for the image.
        Returns:
            Dict[str, Any]: The transformed message with the following fields:
                - chat_template: List of dictionaries with role and content information
                - images: List of transformed image tensors
                - num_image_tokens: List of integers representing the number of tokens needed for each image
        """
        content, images, image_sizes, num_image_tokens = [], [], [], []
        for item in message.content:
            if item["type"] == "image":
                # Process images
                pil_image = item["content"]
                image_tensor = self.image_processor(pil_image, return_tensors="pt")
                image_size = image_tensor["image_sizes"][0]
                images.append(image_tensor["pixel_values"])
                image_sizes.append(image_size)
                encoder_output_feature_size = get_encoder_output_feature_size(image_size, sampling_ratio, self.tile_size, self.patch_size, self.possible_resolutions)
                if self.vision_feature_select_strategy == "default":
                    encoder_output_feature_size -= 1
                num_image_tokens.append(encoder_output_feature_size)
                content.append({"type": "image"})
            elif item["type"] == "text":
                content.append({"type": "text", "text" : item["content"]})
            else:
                raise ValueError(f"Invalid item type: {item['type']}")
        chat_template = [
            {
                "role": message.role,
                "content": content
            }
        ]
        return {"chat_template": chat_template, "images": images, "image_sizes": image_sizes, "num_image_tokens": num_image_tokens}

    def tokenize_messages(
        self,
        messages: List[Message],
        valid_data: bool,
         *,
        inference: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages containing text and/or images into model inputs.

        This method processes each message by:
        1. Converting the message to a chat template format
        2. Applying the tokenizer's chat template to get formatted text
        3. Handling image tokens by replacing them with the appropriate number of tokens
        4. Generating masks based on message properties

        The resulting tokens, masks, position IDs, and encoded images are structured
        for direct input to the DocEV model.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            inference (bool): Whether to run in inference mode. Default is False.

        Returns:
            Dict[str, Any]: The transformed sample with the following fields:
                - tokens: List[int] of tokenized messages
                - mask: List[bool] of masks for the tokenized messages. Ignore the token during loss calculation if masked.
                - encoder_input: Dict[str, Any] of transformed images
                  `images` : List[torch.Tensor] # [num_images, num_tiles, num_channels, tile_size, tile_size]
                  `image_sizes` : List[torch.Tensor]# [num_tiles, 2]
                  `sampling_ratio` : int

        Messages schema:
            [
                Message(
                    role = "user",
                    content = [
                        {"type": "image", "content": <PIL.Image.Image>},
                        {"type": "text", "content": str},
                    ],
                ),
                Message(
                    role = "assistant",
                    content = [
                        {"type": "text", "content": str},
                    ],
                ),
                ...
            ]
        """
        # check if the number of image tokens is correct

        if not valid_data:
            # create dummy tokens, mask, encoder_input
            dummy_num_tile = 2
            dummy_sampling_ratio = 2
            dummy_image_sizes = (self.tile_size, self.tile_size)
            num_dummy_tokens = 10
            encoder_output_feature_size = get_encoder_output_feature_size(dummy_image_sizes, dummy_sampling_ratio, self.tile_size, self.patch_size, self.possible_resolutions)
            encoder_input = {
                "images": [torch.zeros(1, dummy_num_tile, 3, self.tile_size, self.tile_size)],
                "image_sizes": [torch.tensor(dummy_image_sizes)],
                "sampling_ratio": dummy_sampling_ratio
            }
            tokens = [self.image_token_id] * encoder_output_feature_size + [self.tokenizer.pad_token_id] * num_dummy_tokens
            mask = [True] * len(tokens)
            return {
                "tokens": tokens,
                "mask": mask,
                "encoder_input": encoder_input
            }

        total_image_tokens = 0
        if not inference and self.apply_random_sampling_ratio:
            sampling_ratio = random.choice(self.sampling_ratio)
        else:
            sampling_ratio = self.sampling_ratio[0]
        tokens, mask, encoder_input = [], [], {"images": [], "image_sizes": [], "sampling_ratio": sampling_ratio}
        # Process each message
        for message in messages:
            # Convert the message to chat template
            chat_item = self.message_to_chat_item(message, sampling_ratio)
            # Get chat template text
            chat_text = self.tokenizer.apply_chat_template(
                chat_item["chat_template"], tokenize=False, add_generation_prompt=inference, chat_template=self.chat_template
            )
            # Replace image_token with placeholder which is the number of visual tokens
            for num_image_token in chat_item["num_image_tokens"]:
                chat_text = chat_text.replace(self.image_token, "<|placeholder|>" * num_image_token, 1)
                total_image_tokens += num_image_token
            # Replace placeholder with image_token
            chat_text = chat_text.replace("<|placeholder|>", self.image_token)
            # Encode the chat template text
            encoded_token = self.tokenizer(chat_text, add_special_tokens=False)['input_ids']
            # Prepare model inputs
            tokens.extend(encoded_token)
            mask.extend([message.masked] * len(encoded_token)) # If masked, ignore the token during loss calculation
            encoder_input["images"].extend(chat_item["images"])
            encoder_input["image_sizes"].extend(chat_item["image_sizes"])
        if self.max_seq_len is not None and len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            mask = mask[:self.max_seq_len]

        return {
            "tokens": tokens,
            "mask": mask,
            "encoder_input": encoder_input
        }

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply image decoding, transformations and tokenization to messages in the sample.

        This method prepares inputs for early fusion models by converting messages into the
        required format. It is called by torchtune.datasets.SFTTransform in the pipeline:
        raw data → message transform (UfxToMessages) → message → model transform (DocEVTransform) → model input

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field.
            inference (bool): Whether to run in inference mode. Default is False.

        Returns:
            Mapping[str, Any]: The transformed sample with the following fields:
                - tokens: List[int] of tokenized messages
                - mask: List[bool] of masks for the tokenized messages
                - encoder_input: Dict[str, Any] of transformed images
                  `images` : List[torch.Tensor] # [num_images, num_tiles, num_channels, tile_size, tile_size]
                  `image_sizes` : List[torch.Tensor]# [num_tiles, 2]
                  `sampling_ratio` : int
        """
        sample.update(self.tokenize_messages(sample["messages"], valid_data=sample["valid_data"], inference=inference))
        return sample
