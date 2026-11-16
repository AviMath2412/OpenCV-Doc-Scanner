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


def to_scanned_bw(gray: np.ndarray, block_size: int = 25, c_val: int = 15) -> np.ndarray:
    """
    Adaptive threshold -> clean black text on white background,
    the classic 'scanned document' look.
    """
    # Block size must be odd and >= 3
    b_size = max(3, block_size | 1)
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=b_size, C=c_val
    )
    # remove speckle noise
    bw = cv2.medianBlur(bw, 3)
    return bw


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Applies sharpening to a grayscale or color image."""
    if strength <= 0.0:
        return image
    # Unsharp masking or scaled filter kernel
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.0 + 0.7 * strength, blurred, -0.7 * strength, 0)
    return sharpened


def enhance_document(warped_bgr: np.ndarray, mode: str = "bw",
                     brightness: int = 0, contrast: float = 1.0,
                     sharpness_strength: float = 1.0, remove_shadows: bool = True,
                     adaptive_block_size: int = 25, adaptive_c: int = 15):
    """
    Full enhancement chain applied to a perspective-corrected page.

    mode:
        "bw"       -> classic black & white scan (best for OCR + small files)
        "gray"     -> clean grayscale (keeps subtle shading, good for photos/diagrams)
        "color"    -> illumination-corrected color (keeps colour, still de-shadowed)
        "original" -> warped original without destructive filtering

    Returns the enhanced image (single or 3-channel depending on mode).
    """
    if mode == "original":
        out = warped_bgr.copy()
        if brightness != 0 or contrast != 1.0:
            out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness)
        if sharpness_strength > 0:
            out = sharpen(out, sharpness_strength)
        return out

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    if remove_shadows:
        gray = correct_illumination(gray)
    gray = clahe_contrast(gray)
    if sharpness_strength > 0:
        gray = sharpen(gray, sharpness_strength)

    if brightness != 0 or contrast != 1.0:
        gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

    if mode == "gray":
        return gray
    if mode == "bw":
        return to_scanned_bw(gray, block_size=adaptive_block_size, c_val=adaptive_c)
    if mode == "color":
        result = warped_bgr.astype(np.float32)
        if remove_shadows:
            # Re-apply the illumination map to each color channel
            blurred_bg = cv2.GaussianBlur(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY),
                                           (0, 0), sigmaX=25, sigmaY=25)
            blurred_bg = np.where(blurred_bg == 0, 1, blurred_bg).astype(np.float32)
            for c in range(3):
                result[:, :, c] = cv2.divide(result[:, :, c], blurred_bg, scale=255)
        out = np.clip(result, 0, 255).astype(np.uint8)
        if brightness != 0 or contrast != 1.0:
            out = cv2.convertScaleAbs(out, alpha=contrast, beta=brightness)
        if sharpness_strength > 0:
            out = sharpen(out, sharpness_strength)
        return out

    raise ValueError(f"Unknown mode: {mode}")
