import cv2
import numpy as np


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by the given angle (degrees) around its center (Z-axis).

    Uses white background for the border areas created by rotation.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    return rotated


def generate_rotated_images(image: np.ndarray, angles: list[float]) -> list[tuple[float, np.ndarray]]:
    """Generate rotated versions of an image at the specified angles.

    Returns list of (angle, rotated_image) tuples.
    """
    results = []
    for angle in angles:
        if angle == 0:
            results.append((0.0, image.copy()))
        else:
            results.append((angle, rotate_image(image, angle)))
    return results
