import torch
import numpy as np

def select_best_resolution(
    original_size: tuple,
    possible_resolutions: list,
    ) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit

def get_padding(original_height, original_width, current_height, current_width):
    """
    Calculate the padding size for a tensor to match the original size.

    Args:
        original_height (`int`):
            The original height of the image.
        original_width (`int`):
            The original width of the image.
        current_height (`int`):
            The current height of the image.
        current_width (`int`):
            The current width of the image.

    Returns:
        padding (`int`):
            The padding size for the tensor.
        padding_side (`str`):
            The padding side for the tensor.
    """
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        padding_side = "left"
    elif original_aspect_ratio < current_aspect_ratio:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        padding_side = "right"
    else:
        padding = 0
        padding_side = "none"
    return padding, padding_side

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]
    padding, padding_side = get_padding(original_height, original_width, current_height, current_width)

    if padding_side == "left":
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    elif padding_side == "right":
        unpadded_tensor = tensor[:, :, padding : current_width - padding]
    else:  # Handle the case where padding_side is "none"
        unpadded_tensor = tensor

    return unpadded_tensor

def get_anyres_image_grid_shape(image_size, grid_pinpoints, tile_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        tile_size (`int`):
            The size of each image tile.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // tile_size, width // tile_size

def get_num_unpadded_features(
    original_height: int,
    original_width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> tuple[int, int]:
    """
    Calculate the number of unpadded features and newline features based on the original image dimensions.

    This function computes the actual feature size after considering the aspect ratio and
    removing padding to match the original image dimensions.

    Args:
        original_height (int): Original height of the image.
        original_width (int): Original width of the image.
        npatches (int): Number of patches.
        num_patch_height (int): Number of patches in the height dimension.
        num_patch_width (int): Number of patches in the width dimension.

    Returns:
        tuple[int, int]: A tuple containing (unpadded_features, newline_features)
            - unpadded_features: The total number of features without padding
            - newline_features: The number of features in a height row
    """
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width
    padding, padding_side = get_padding(original_height, original_width, current_height, current_width)

    if padding_side == "left":
        current_height = current_height - (2 * padding)
    elif padding_side == "right":
        current_width = current_width - (2 * padding)

    unpadded_features = current_height * current_width
    newline_features = current_height

    return (unpadded_features, newline_features)

def get_encoder_output_feature_size(
    image_size: tuple,
    sampling_ratio: int,
    tile_size: int,
    patch_size: int,
    possible_resolutions: list,
) -> int:
    """
    Calculate the size of the encoder output feature for an image of the given size.

    This method considers downsampling ratio (either randomly chosen or fixed),
    the base feature size, unpadded feature size, newline feature size, and
    additional image tokens to determine the total output feature size.

    Args:
        image_size (tuple): The size of the input image in the format (height, width).
        sampling_ratio (int): The sampling ratio to use for the image.

    Returns:
        int: The total size of the encoder output feature.
    """

    base_feature_size = int(((tile_size // patch_size) // sampling_ratio) ** 2)

    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        image_size=image_size,
        grid_pinpoints=possible_resolutions,
        tile_size=tile_size,
    )

    unpadded_feature_size, newline_feature_size = get_num_unpadded_features(
        original_height=image_size[0],
        original_width=image_size[1],
        npatches=int((tile_size // patch_size) // sampling_ratio),
        num_patch_height=num_patch_height,
        num_patch_width=num_patch_width,
    )
    # tile image + thumbnail image
    return unpadded_feature_size + newline_feature_size + base_feature_size