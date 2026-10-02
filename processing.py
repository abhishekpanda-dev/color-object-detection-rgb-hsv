"""Image-processing utilities for the project."""

from collections.abc import Sequence

import cv2
import numpy as np


def create_rgb_mask(
    rgb_image: np.ndarray,
    min_rgb: Sequence[int],
    max_rgb: Sequence[int],
) -> np.ndarray:
    """Return a binary mask for pixels inside the inclusive RGB thresholds.

    The input must use RGB channel order, as produced by converting a PIL image
    to a NumPy array. OpenCV images are commonly BGR, but no BGR conversion is
    performed here: channel 0 is red, channel 1 is green, and channel 2 is blue.
    """
    image = np.asarray(rgb_image)
    minimum = np.asarray(min_rgb)
    maximum = np.asarray(max_rgb)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("min_rgb and max_rgb must each contain R, G, and B values")
    if np.any(minimum < 0) or np.any(maximum > 255):
        raise ValueError("RGB threshold values must be between 0 and 255")
    if np.any(minimum > maximum):
        raise ValueError("Each RGB minimum must be less than or equal to its maximum")

    # Name the channels explicitly so red and blue cannot be confused with
    # OpenCV's usual BGR ordering.
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    within_threshold = (
        (red_channel >= minimum[0])
        & (red_channel <= maximum[0])
        & (green_channel >= minimum[1])
        & (green_channel <= maximum[1])
        & (blue_channel >= minimum[2])
        & (blue_channel <= maximum[2])
    )

    # Target-color pixels are white (255); all other pixels are black (0).
    return np.where(within_threshold, 255, 0).astype(np.uint8)


def convert_rgb_to_hsv(rgb_image: np.ndarray) -> np.ndarray:
    """Convert a three-channel RGB image to OpenCV's HSV representation."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")

    # The source comes from PIL in RGB order, so COLOR_RGB2HSV is required.
    # OpenCV stores HSV as H: 0-179, S: 0-255, and V: 0-255.
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def create_hsv_mask(
    hsv_image: np.ndarray,
    min_hsv: Sequence[int],
    max_hsv: Sequence[int],
) -> np.ndarray:
    """Return a binary mask for one inclusive OpenCV HSV threshold range."""
    image = _validate_hsv_image(hsv_image)
    minimum, maximum = _validate_hsv_thresholds(min_hsv, max_hsv)

    # inRange returns uint8 values: 255 inside the range and 0 outside it.
    return cv2.inRange(image, minimum, maximum)


def create_red_hsv_mask(
    hsv_image: np.ndarray,
    lower_hue_range: Sequence[int],
    upper_hue_range: Sequence[int],
    saturation_range: Sequence[int],
    value_range: Sequence[int],
) -> np.ndarray:
    """Return a red mask by combining ranges at both ends of the Hue scale."""
    image = _validate_hsv_image(hsv_image)
    lower_hue = _validate_channel_range("Lower red Hue", lower_hue_range, 179)
    upper_hue = _validate_channel_range("Upper red Hue", upper_hue_range, 179)
    saturation = _validate_channel_range("Saturation", saturation_range, 255)
    value = _validate_channel_range("Value", value_range, 255)

    lower_red_mask = cv2.inRange(
        image,
        np.array((lower_hue[0], saturation[0], value[0]), dtype=np.uint8),
        np.array((lower_hue[1], saturation[1], value[1]), dtype=np.uint8),
    )
    upper_red_mask = cv2.inRange(
        image,
        np.array((upper_hue[0], saturation[0], value[0]), dtype=np.uint8),
        np.array((upper_hue[1], saturation[1], value[1]), dtype=np.uint8),
    )

    # Red wraps around Hue 0/179, so both binary masks form the final result.
    return cv2.bitwise_or(lower_red_mask, upper_red_mask)


def clean_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Clean a binary mask using morphological opening followed by closing."""
    binary_mask = np.asarray(mask)

    if binary_mask.ndim != 2:
        raise ValueError("mask must be a 2D grayscale array")
    if binary_mask.dtype != np.uint8:
        raise ValueError("mask must have dtype uint8")
    if not np.all(np.isin(binary_mask, (0, 255))):
        raise ValueError("mask must contain only binary values 0 and 255")
    if kernel_size not in (3, 5, 7):
        raise ValueError("kernel_size must be one of 3, 5, or 7")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Opening is erosion followed by dilation; it removes isolated white noise.
    opened_mask = cv2.morphologyEx(
        binary_mask.copy(),
        cv2.MORPH_OPEN,
        kernel,
    )

    # Closing is dilation followed by erosion; it fills small foreground gaps.
    cleaned_mask = cv2.morphologyEx(
        opened_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )
    return cleaned_mask


def _validate_hsv_image(hsv_image: np.ndarray) -> np.ndarray:
    """Validate the shape and dtype expected by OpenCV HSV operations."""
    image = np.asarray(hsv_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("hsv_image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("hsv_image must have dtype uint8")
    return image


def _validate_hsv_thresholds(
    min_hsv: Sequence[int],
    max_hsv: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate complete H/S/V lower and upper threshold triplets."""
    minimum = np.asarray(min_hsv)
    maximum = np.asarray(max_hsv)
    limits = np.array((179, 255, 255))

    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("min_hsv and max_hsv must each contain H, S, and V values")
    if np.any(minimum < 0) or np.any(maximum > limits):
        raise ValueError("HSV values must use H: 0-179 and S/V: 0-255")
    if np.any(minimum > maximum):
        raise ValueError("Each HSV minimum must be less than or equal to its maximum")

    return minimum.astype(np.uint8), maximum.astype(np.uint8)


def _validate_channel_range(
    name: str,
    values: Sequence[int],
    maximum_allowed: int,
) -> tuple[int, int]:
    """Validate one inclusive minimum/maximum channel range."""
    channel_range = np.asarray(values)
    if channel_range.shape != (2,):
        raise ValueError(f"{name} range must contain a minimum and maximum")
    if (
        channel_range[0] < 0
        or channel_range[1] > maximum_allowed
        or channel_range[0] > channel_range[1]
    ):
        raise ValueError(f"Invalid {name} range")
    return int(channel_range[0]), int(channel_range[1])
