"""
edge_detect.py
--------------
Finds the four-sided page boundary in a photo of a document.

Approach:
    1. Downscale for speed, remember the scale factor.
    2. Convert to grayscale, blur to suppress texture/noise.
    3. Canny edge detection.
    4. Dilate edges slightly to close small gaps in the outline.
    5. Find contours, keep the largest ones, and look for a 4-point
       polygon (approxPolyDP) among them -- that is the page.
    6. If nothing looks like a clean quadrilateral (e.g. the page fills
       the whole frame, or the photo is already a crop), fall back to
       the image's own bounding box so the pipeline never crashes.
"""

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sorts 4 (x, y) points into [top-left, top-right, bottom-right, bottom-left]
    order, which is what the perspective transform expects.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest y-x

    return rect


def find_document_contour(image: np.ndarray, resize_height: int = 700):
    """
    Returns a 4x2 float32 array of the page corners in ORIGINAL image
    coordinates, ordered [top-left, top-right, bottom-right, bottom-left].

    Falls back to the full image corners if no good quadrilateral is found,
    so downstream code always receives a valid box.
    """
    orig_h, orig_w = image.shape[:2]
    ratio = orig_h / float(resize_height)
    resized = cv2.resize(image, (int(orig_w / ratio), resize_height))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    page_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * resized.shape[0] * resized.shape[1]:
            page_contour = approx
            break

    if page_contour is None:
        # Fallback: use the whole frame as the "page"
        h, w = resized.shape[:2]
        page_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    pts = page_contour.reshape(4, 2).astype("float32")
    pts *= ratio  # scale back up to original resolution
    return _order_points(pts)


def draw_contour_debug(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Returns a copy of image with the detected page outline drawn on it
    (handy for a report screenshot / viva demo)."""
    debug = image.copy()
    pts = corners.astype(int)
    cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    for (x, y) in pts:
        cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)
    return debug
