"""Contour-based color object-detection utilities."""

import cv2
import numpy as np


def detect_colored_objects(
    rgb_image: np.ndarray,
    cleaned_mask: np.ndarray,
    minimum_area: float,
    target_color: str,
) -> tuple[np.ndarray, int, list[dict[str, object]], int]:
    """Detect, filter, and annotate colored regions from a cleaned mask.

    Returns the annotated RGB image, accepted object count, accepted detection
    metadata, and the number of external contours found before area filtering.
    """
    image = np.asarray(rgb_image)
    mask = np.asarray(cleaned_mask)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        raise ValueError("rgb_image must have dtype uint8")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError("cleaned_mask must be 2D and match the image dimensions")
    if mask.dtype != np.uint8 or not np.all(np.isin(mask, (0, 255))):
        raise ValueError("cleaned_mask must be uint8 and contain only 0 and 255")
    if minimum_area < 0:
        raise ValueError("minimum_area must be non-negative")
    if target_color not in {"Red", "Green", "Blue", "Yellow"}:
        raise ValueError("target_color must be Red, Green, Blue, or Yellow")

    # A copy protects the cleaned mask from any OpenCV-version-specific changes.
    contour_result = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    detections: list[dict[str, object]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        detections.append(
            {
                "label": f"{target_color.upper()} OBJECT",
                "area": area,
                "bounding_box": (x, y, width, height),
            }
        )

    # Stable top-to-bottom, left-to-right ordering keeps the result table clear.
    detections.sort(key=lambda item: (item["bounding_box"][1], item["bounding_box"][0]))
    annotated_image = image.copy()

    for detection in detections:
        x, y, width, height = detection["bounding_box"]
        label = detection["label"]

        # Black and white remain unambiguous when drawn directly on an RGB image.
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 0, 0),
            4,
        )
        cv2.rectangle(
            annotated_image,
            (x, y),
            (x + width - 1, y + height - 1),
            (255, 255, 255),
            2,
        )
        text_y = max(y - 8, 18)
        cv2.putText(
            annotated_image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated_image, len(detections), detections, len(contours)
