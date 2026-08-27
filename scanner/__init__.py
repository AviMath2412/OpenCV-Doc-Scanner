"""
Document Scanner & OCR package.

Modules:
    edge_detect  - page boundary detection (contours)
    perspective  - four-point perspective transform (de-skew)
    enhance      - illumination correction / cleanup filters
    ocr          - Tesseract OCR + searchable PDF export
    pipeline     - ties everything together
"""

__version__ = "1.0.0"
