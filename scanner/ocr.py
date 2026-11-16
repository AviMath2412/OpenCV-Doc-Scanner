"""
ocr.py
------
Runs Tesseract OCR on the enhanced page and produces:
  1. Plain extracted text (.txt)
  2. A searchable PDF (image + invisible text layer) using Tesseract's
     own PDF renderer, which is the most reliable way to keep the text
     perfectly aligned under the scanned image.
"""

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

import cv2
import numpy as np
from PIL import Image


def _check_pytesseract():
    if not HAS_PYTESSERACT:
        raise ImportError(
            "The 'pytesseract' package is not installed in the currently active Python environment.\n"
            "To fix this, either:\n"
            "  1. Run with your Conda environment: /opt/homebrew/Caskroom/miniconda/base/bin/python3 gui.py\n"
            "  2. Or activate Conda: conda activate base && python gui.py\n"
            "  3. Or install pytesseract in this Python: pip install pytesseract"
        )


def _to_pil(image: np.ndarray) -> Image.Image:
    """Converts a BGR/gray OpenCV ndarray to a PIL Image for pytesseract."""
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def extract_text(image: np.ndarray, lang: str = "eng") -> str:
    """Runs OCR and returns plain text."""
    if not HAS_PYTESSERACT:
        return "(OCR unavailable: 'pytesseract' is not installed in this Python environment.)"
    return pytesseract.image_to_string(image, lang=lang)


def extract_text_with_confidence(image: np.ndarray, lang: str = "eng"):
    """
    Returns (text, mean_confidence) so the app can flag low-quality scans
    to the user (e.g. 'text unclear, try rescanning').
    """
    if not HAS_PYTESSERACT:
        return (
            "(OCR unavailable: pytesseract is not installed in this Python.\n"
            "Run 'conda activate base' or 'pip install pytesseract'.)",
            0.0
        )
    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        text = pytesseract.image_to_string(image, lang=lang)
        return text, mean_conf
    except Exception as e:
        return f"(OCR error: {e})", 0.0


def image_to_searchable_pdf(image: np.ndarray, output_path: str, lang: str = "eng"):
    """
    Uses Tesseract's built-in PDF output (image_to_pdf_or_hocr) so the
    invisible OCR text layer is pixel-aligned to the image -- this is
    more robust than manually overlaying text with reportlab.
    """
    _check_pytesseract()
    pil_image = _to_pil(image)
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(
        pil_image, lang=lang, extension="pdf"
    )
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path


def images_to_searchable_pdf(images: list, output_path: str, lang: str = "eng"):
    """
    Multi-page version: OCRs each page image and merges them into a single
    searchable PDF. Uses pypdf if available, else falls back to writing the
    first page's Tesseract PDF only (kept simple & dependency-light).
    """
    _check_pytesseract()
    page_pdfs = []
    for img in images:
        page_pdfs.append(pytesseract.image_to_pdf_or_hocr(
            _to_pil(img), lang=lang, extension="pdf"
        ))

    if not page_pdfs:
        raise RuntimeError("No pages to write")

    try:
        from pypdf import PdfWriter, PdfReader
        import io
        writer = PdfWriter()
        for pdf_bytes in page_pdfs:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                writer.add_page(page)
        with open(output_path, "wb") as f:
            writer.write(f)
    except ImportError:
        # Fallback: no pypdf installed, just save the pages separately
        for i, pdf_bytes in enumerate(page_pdfs):
            path = output_path.replace(".pdf", f"_page{i+1}.pdf")
            with open(path, "wb") as f:
                f.write(pdf_bytes)

    return output_path
