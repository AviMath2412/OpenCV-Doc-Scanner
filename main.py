"""
Document Scanner & OCR - CLI
=============================
Usage:
    python main.py --input photo.jpg --output out/
    python main.py --input photo.jpg --output out/ --mode gray
    python main.py --input photo.jpg --output out/ --lang eng+hin

    Batch (multi-page -> one PDF):
    python main.py --input page1.jpg page2.jpg page3.jpg --output out/ --batch
"""

import argparse
import os
import sys
import cv2

from scanner.pipeline import process_image
from scanner.edge_detect import find_document_contour
from scanner.perspective import four_point_transform, fine_deskew
from scanner.enhance import enhance_document
from scanner.ocr import images_to_searchable_pdf, extract_text_with_confidence


def run_single(args):
    result = process_image(args.input[0], args.output, mode=args.mode, lang=args.lang)
    print(f"\nPage boundary detected, perspective corrected, and enhanced ({args.mode} mode).")
    print(f"OCR confidence: {result.confidence:.1f}%")
    if result.confidence < 40:
        print("  Note: low confidence -- try better lighting or a flatter page for cleaner text.")
    print(f"\nFiles written to: {os.path.abspath(args.output)}")
    print("  - enhanced.png       (cleaned scan)")
    print("  - debug_contour.png  (detected page outline, for your report/demo)")
    print("  - text.txt           (extracted text)")
    print("  - document.pdf       (searchable PDF)")
    print("\n--- Extracted text preview ---")
    preview = result.text.strip()
    print(preview[:500] + ("..." if len(preview) > 500 else "") if preview else "(no text detected)")


def run_batch(args):
    os.makedirs(args.output, exist_ok=True)
    enhanced_pages = []
    all_text = []

    for i, path in enumerate(args.input, start=1):
        image = cv2.imread(path)
        if image is None:
            print(f"  Skipping unreadable file: {path}")
            continue
        corners = find_document_contour(image)
        warped = four_point_transform(image, corners)
        enhanced = enhance_document(warped, mode=args.mode)
        if enhanced.ndim == 2:
            enhanced = fine_deskew(enhanced)
        enhanced_pages.append(enhanced)

        text, conf = extract_text_with_confidence(enhanced, lang=args.lang)
        all_text.append(f"----- Page {i} (confidence {conf:.1f}%) -----\n{text}")
        cv2.imwrite(os.path.join(args.output, f"page{i}.png"), enhanced)
        print(f"  Processed page {i}/{len(args.input)}")

    if not enhanced_pages:
        print("No valid pages processed.")
        sys.exit(1)

    pdf_path = os.path.join(args.output, "document.pdf")
    images_to_searchable_pdf(enhanced_pages, pdf_path, lang=args.lang)

    with open(os.path.join(args.output, "text.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print(f"\nMulti-page document written to: {os.path.abspath(args.output)}")
    print("  - page*.png     (each enhanced page)")
    print("  - document.pdf  (combined searchable PDF)")
    print("  - text.txt      (all extracted text)")


def main():
    parser = argparse.ArgumentParser(
        description="Document Scanner & OCR -- turn a photo of a page into a clean, searchable PDF."
    )
    parser.add_argument("--input", "-i", nargs="+", required=True,
                         help="Path(s) to input image(s). Multiple paths + --batch = multi-page PDF.")
    parser.add_argument("--output", "-o", default="output",
                         help="Output directory (default: ./output)")
    parser.add_argument("--mode", "-m", choices=["bw", "gray", "color"], default="bw",
                         help="Enhancement style (default: bw)")
    parser.add_argument("--lang", "-l", default="eng",
                         help="Tesseract language code, e.g. eng, hin, eng+hin (default: eng)")
    parser.add_argument("--batch", action="store_true",
                         help="Treat all --input files as pages of ONE document")
    args = parser.parse_args()

    if args.batch or len(args.input) > 1:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
