"""
pipeline.py
-----------
Glues the whole flow together:
    load -> detect page -> perspective warp -> deskew -> enhance -> OCR -> export
"""

import os
import cv2

from .edge_detect import find_document_contour, draw_contour_debug
from .perspective import four_point_transform, fine_deskew
from .enhance import enhance_document
from .ocr import extract_text_with_confidence, image_to_searchable_pdf


class ScanResult:
    def __init__(self):
        self.original = None
        self.debug_contour = None
        self.warped = None
        self.enhanced = None
        self.text = ""
        self.confidence = 0.0


def process_image(input_path: str, output_dir: str, mode: str = "bw",
                   lang: str = "eng") -> ScanResult:
    """
    Runs the full scanner pipeline on a single image file and writes:
        <output_dir>/enhanced.png   - the cleaned-up scan
        <output_dir>/debug_contour.png - page outline overlay (for reports/demo)
        <output_dir>/text.txt       - extracted text
        <output_dir>/document.pdf   - searchable PDF
    Returns a ScanResult with everything in memory too.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = ScanResult()
    result.original = image

    # 1. Page boundary detection
    corners = find_document_contour(image)
    result.debug_contour = draw_contour_debug(image, corners)

    # 2. Perspective transform (de-skew the big rotation/tilt)
    warped = four_point_transform(image, corners)
    result.warped = warped

    # 3. Enhancement (illumination correction + contrast + optional B/W)
    enhanced = enhance_document(warped, mode=mode)

    # 3b. Fine-grained deskew on near-flat pages (small residual rotation)
    gray_for_skew = enhanced if enhanced.ndim == 2 else cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    if enhanced.ndim == 2:
        enhanced = fine_deskew(enhanced)

    result.enhanced = enhanced

    # 4. OCR
    text, conf = extract_text_with_confidence(enhanced, lang=lang)
    result.text = text
    result.confidence = conf

    # 5. Export everything
    cv2.imwrite(os.path.join(output_dir, "enhanced.png"), enhanced)
    cv2.imwrite(os.path.join(output_dir, "debug_contour.png"), result.debug_contour)
    with open(os.path.join(output_dir, "text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    image_to_searchable_pdf(enhanced, os.path.join(output_dir, "document.pdf"), lang=lang)

    return result
