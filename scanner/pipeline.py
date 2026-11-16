"""
pipeline.py
-----------
Glues the whole flow together:
    load -> detect page -> perspective warp -> deskew -> enhance -> OCR -> export
"""

import os
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np

from .edge_detect import find_document_contour, draw_contour_debug
from .perspective import four_point_transform, fine_deskew
from .enhance import enhance_document, correct_illumination, clahe_contrast, sharpen, to_scanned_bw
from .ocr import extract_text_with_confidence, image_to_searchable_pdf


class ScanResult:
    def __init__(self):
        self.original: Optional[np.ndarray] = None
        self.corners: Optional[np.ndarray] = None
        self.debug_contour: Optional[np.ndarray] = None
        self.warped: Optional[np.ndarray] = None
        self.enhanced: Optional[np.ndarray] = None
        self.text: str = ""
        self.confidence: float = 0.0
        self.stages: List[Dict[str, Any]] = []


def get_pipeline_stages(image: np.ndarray, corners: Optional[np.ndarray] = None,
                        mode: str = "bw") -> List[Dict[str, Any]]:
    """
    Generates visual intermediate outputs for each computer vision stage in the
    pipeline. Ideal for academic reports, project reviews, and viva demos.
    """
    stages = []

    # Stage 1: Input Photo
    stages.append({
        "id": 1,
        "title": "1. Input Photo",
        "image": image.copy(),
        "description": "Original image captured from camera or loaded from storage.",
        "concept": "Digital image in 3-channel BGR color space [Height x Width x 3 channels]."
    })

    # Downscale for edge detection matching find_document_contour
    resize_h = 700
    orig_h, orig_w = image.shape[:2]
    ratio = orig_h / float(resize_h)
    resized = cv2.resize(image, (int(orig_w / ratio), resize_h))

    # Stage 2: Grayscale & Gaussian Filtering
    gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred_resized = cv2.GaussianBlur(gray_resized, (5, 5), 0)
    stages.append({
        "id": 2,
        "title": "2. Grayscale & Noise Reduction",
        "image": cv2.cvtColor(blurred_resized, cv2.COLOR_GRAY2BGR),
        "description": "Converts BGR to single-channel luminance (Y = 0.299R + 0.587G + 0.114B) and applies a 5x5 Gaussian blur kernel.",
        "concept": "Suppresses high-frequency sensor noise and paper micro-textures so edge detection focuses on true page boundaries."
    })

    # Stage 3: Canny Edge Detection
    edged = cv2.Canny(blurred_resized, 50, 150)
    stages.append({
        "id": 3,
        "title": "3. Canny Edge Map",
        "image": cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR),
        "description": "Calculates directional gradients using Sobel operators, suppresses non-maximum pixels, and applies hysteresis thresholding (50 to 150).",
        "concept": "Isolates sharp intensity discontinuities representing physical paper borders and text lines."
    })

    # Stage 4: Morphological Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    stages.append({
        "id": 4,
        "title": "4. Morphological Dilation",
        "image": cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR),
        "description": "Expands edge boundaries using a 5x5 rectangular structuring element: A ⊕ B = ⋃ {A + b | b ∈ B}.",
        "concept": "Bridges small gaps in the page boundary caused by lighting variations or uneven background contrast."
    })

    # Stage 5: Contour Detection & Corner Localization
    if corners is None:
        detected_corners = find_document_contour(image)
    else:
        detected_corners = corners
    debug_contour = draw_contour_debug(image, detected_corners)
    stages.append({
        "id": 5,
        "title": "5. Contour Detection & Quad Fit",
        "image": debug_contour,
        "description": "Finds topological contours (cv2.findContours) and approximates polygons using Douglas-Peucker algorithm (cv2.approxPolyDP).",
        "concept": "Identifies the largest quadrilateral contour, extracting 4 corner coordinates ordered: TL, TR, BR, BL."
    })

    # Stage 6: Perspective Transform (Homography)
    warped = four_point_transform(image, detected_corners)
    stages.append({
        "id": 6,
        "title": "6. Perspective Rectification",
        "image": warped,
        "description": "Calculates 3x3 perspective homography matrix M (cv2.getPerspectiveTransform) and warps quad into a top-down orthogonal plane.",
        "concept": "Eliminates camera tilt, skew, and angular perspective distortion: [x', y', 1]^T ~ M · [x, y, 1]^T."
    })

    # Stage 7: Illumination Background Estimation
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred_bg = cv2.GaussianBlur(warped_gray, (0, 0), sigmaX=25, sigmaY=25)
    stages.append({
        "id": 7,
        "title": "7. Illumination Background Model",
        "image": cv2.cvtColor(blurred_bg, cv2.COLOR_GRAY2BGR),
        "description": "Applies low-pass Gaussian filter with large kernel (sigma=25) to isolate slow-varying illumination and shadow gradients.",
        "concept": "Separates the document into low-frequency lighting background B(x, y) and high-frequency ink text details I(x, y)."
    })

    # Stage 8: Shadow-Normalized Image
    normalized = correct_illumination(warped_gray)
    stages.append({
        "id": 8,
        "title": "8. Shadow-Normalized Signal",
        "image": cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR),
        "description": "Divides input luminance by background model: R(x, y) = 255 * (I(x, y) / B(x, y)).",
        "concept": "Flattens phone shadows and ambient light gradients, producing a uniform white paper background."
    })

    # Stage 9: Final Enhancement & Binarization
    enhanced = enhance_document(warped, mode=mode)
    stages.append({
        "id": 9,
        "title": "9. Enhanced Scanned Output",
        "image": cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if enhanced.ndim == 2 else enhanced,
        "description": f"Applies CLAHE local contrast, Laplacian/unsharp sharpening, and {'adaptive Gaussian thresholding' if mode == 'bw' else 'tonal adjustments'}.",
        "concept": "Produces clean, publication-ready scanned pages optimized for human reading and Tesseract OCR text extraction."
    })

    return stages


def process_image(input_path_or_image: Union[str, np.ndarray],
                  output_dir: Optional[str] = None,
                  mode: str = "bw",
                  lang: str = "eng",
                  corners: Optional[np.ndarray] = None,
                  brightness: int = 0,
                  contrast: float = 1.0,
                  sharpness: float = 1.0,
                  fine_deskew_flag: bool = True,
                  remove_shadows: bool = True,
                  adaptive_block_size: int = 25,
                  adaptive_c: int = 15,
                  run_ocr: bool = True,
                  generate_stages: bool = True) -> ScanResult:
    """
    Runs the full scanner pipeline on an image path or in-memory OpenCV image.
    Optionally saves enhanced scan, outline overlay, text, and searchable PDF to output_dir.
    """
    if isinstance(input_path_or_image, str):
        image = cv2.imread(input_path_or_image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path_or_image}")
    else:
        image = input_path_or_image.copy()

    result = ScanResult()
    result.original = image

    # 1. Page boundary detection or user corners
    if corners is None:
        corners = find_document_contour(image)
    result.corners = corners
    result.debug_contour = draw_contour_debug(image, corners)

    # 2. Perspective transform
    warped = four_point_transform(image, corners)
    result.warped = warped

    # 3. Enhancement
    enhanced = enhance_document(
        warped,
        mode=mode,
        brightness=brightness,
        contrast=contrast,
        sharpness_strength=sharpness,
        remove_shadows=remove_shadows,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c
    )

    # 3b. Fine deskew
    if fine_deskew_flag:
        enhanced = fine_deskew(enhanced)

    result.enhanced = enhanced

    # 4. OCR
    if run_ocr:
        text, conf = extract_text_with_confidence(enhanced, lang=lang)
        result.text = text
        result.confidence = conf

    # 5. Pipeline visual stages
    if generate_stages:
        result.stages = get_pipeline_stages(image, corners=corners, mode=mode)

    # 6. Optional file export
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "enhanced.png"), enhanced)
        cv2.imwrite(os.path.join(output_dir, "debug_contour.png"), result.debug_contour)
        if run_ocr:
            with open(os.path.join(output_dir, "text.txt"), "w", encoding="utf-8") as f:
                f.write(result.text)
            image_to_searchable_pdf(enhanced, os.path.join(output_dir, "document.pdf"), lang=lang)

    return result
