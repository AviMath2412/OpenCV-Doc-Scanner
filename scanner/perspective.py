"""
perspective.py
---------------
Warps the detected quadrilateral into a flat, top-down rectangle
("de-skewing" the page so it reads as if it were scanned flat).
"""

import cv2
import numpy as np


def four_point_transform(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    corners: 4x2 array ordered [top-left, top-right, bottom-right, bottom-left].
    Computes the target rectangle size from the actual side lengths of the
    quadrilateral (so aspect ratio is preserved) and warps the image into it.
    """
    (tl, tr, br, bl) = corners

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    # Trim a thin sliver off each edge: the very border pixels often pick up
    # a faint line from the background just outside the detected corners.
    trim = max(2, int(0.01 * min(max_width, max_height)))
    warped = warped[trim:-trim, trim:-trim]
    return warped


def fine_deskew(gray: np.ndarray) -> np.ndarray:
    """
    Corrects small residual rotation (a couple of degrees) left after the
    perspective warp, using the minAreaRect of all "ink" pixels.
    Operates on and returns a single-channel (grayscale) image.
    """
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return gray  # not enough content to safely estimate an angle

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore near-zero corrections and wild outliers (bad estimate)
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated
