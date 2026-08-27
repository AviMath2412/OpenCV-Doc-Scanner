"""
enhance.py
----------
Cleans up the de-skewed page so it looks like a real scan and OCRs
reliably: removes uneven lighting/shadows, boosts local contrast, and
produces a crisp black-on-white version.
"""

import cv2
import numpy as np


def correct_illumination(gray: np.ndarray) -> np.ndarray:
    """
    Estimates the lighting/shadow pattern with a heavily blurred copy of the
    image, then divides it out. This flattens uneven lighting (a common
    problem with phone-camera scans) without destroying text contrast.
    """
    blurred_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    # avoid divide-by-zero
    blurred_bg = np.where(blurred_bg == 0, 1, blurred_bg)
    normalized = cv2.divide(gray, blurred_bg, scale=255)
    return normalized


def clahe_contrast(gray: np.ndarray) -> np.ndarray:
    """Adaptive histogram equalization for local contrast boost."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def to_scanned_bw(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold -> clean black text on white background,
    the classic 'scanned document' look.
    """
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=15
    )
    # remove speckle noise
    bw = cv2.medianBlur(bw, 3)
    return bw


def sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def enhance_document(warped_bgr: np.ndarray, mode: str = "bw"):
    """
    Full enhancement chain applied to a perspective-corrected page.

    mode:
        "bw"      -> classic black & white scan (best for OCR + small files)
        "gray"    -> clean grayscale (keeps subtle shading, good for photos/diagrams)
        "color"   -> illumination-corrected color (keeps colour, still de-shadowed)

    Returns the enhanced image (single or 3-channel depending on mode).
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = correct_illumination(gray)
    gray = clahe_contrast(gray)
    gray = sharpen(gray)

    if mode == "gray":
        return gray
    if mode == "bw":
        return to_scanned_bw(gray)
    if mode == "color":
        # Re-apply the illumination map to each color channel
        blurred_bg = cv2.GaussianBlur(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY),
                                       (0, 0), sigmaX=25, sigmaY=25)
        blurred_bg = np.where(blurred_bg == 0, 1, blurred_bg).astype(np.float32)
        result = warped_bgr.astype(np.float32)
        for c in range(3):
            result[:, :, c] = cv2.divide(result[:, :, c], blurred_bg, scale=255)
        return np.clip(result, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown mode: {mode}")
