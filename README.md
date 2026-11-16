# OpenCV Document Scanner & OCR Engine

A production-quality Python + OpenCV + Tesseract desktop application and CLI that transforms phone photos and live webcam captures of documents into clean, perspective-rectified, shadow-free, and searchable PDFs.

**Academic alignment:**
- **SDG 4:** Quality Education (instant digitization of study notes, handouts, research papers).
- **SDG 9:** Industry, Innovation & Infrastructure (real-world computer vision & OCR pipeline).
- **SDG 12:** Responsible Consumption & Production (paperless digital document workflows).

---

## 🌟 Key Features

### 🖥️ Interactive Desktop GUI (`gui.py`)
- **Interactive Quad Boundary Adjustment (CamScanner-style):**
  - Automatic 4-corner document detection.
  - Interactive canvas with 4 draggable corner handles (TL, TR, BR, BL) and box dragging.
  - **Real-Time Precision Magnifying Loupe:** Floating 2.5x zoom lens while dragging corners for pixel-perfect alignment.
  - 90° Clockwise/Counter-Clockwise rotation, full-frame fallback, and auto-detect reset.
- **9-Stage Computer Vision Pipeline Inspector (Viva / Demo Mode):**
  - Step-by-step visual inspection of every internal OpenCV stage.
  - Academic explanation cards detailing the mathematical concepts (Sobel gradients, Homography $3 \times 3$ matrices, Gaussian background division, Otsu/Adaptive Gaussian binarization, and Tesseract LSTM OCR).
- **Live Webcam / Camera Scanner:**
  - Real-time video stream (`cv2.VideoCapture`).
  - Live document boundary overlay rendered directly on the camera feed at 30 FPS.
  - Single-click document capture (`📸 Capture Page`).
- **Multi-Page & Batch Document Manager:**
  - Sidebar thumbnail list for multi-page documents.
  - Add, remove, and reorder pages (Up / Down).
  - One-click batch scanning of all pages.
  - Combine multiple pages into a single multi-page searchable PDF.
- **Multiple Enhancement Modes & Real-Time Tuning:**
  - Modes: `bw` (clean adaptive binarization), `gray` (smooth de-shadowed grayscale), `color` (shadow-free vivid color), `original` (rectified without filter).
  - Real-time sliders for Contrast ($0.5\times - 2.5\times$), Brightness ($-50$ to $+50$), Sharpness ($0.0 - 3.0$), and fine deskew.
- **Integrated OCR & Searchable Export:**
  - Color-coded OCR confidence meter (Green $>75\%$, Amber $45-75\%$, Red $<45\%$).
  - Live keyword search highlighting within extracted text.
  - One-click "Copy Text", "Save .txt", "Export Searchable PDF", and "Export All Assets Bundle".

### ⚡ CLI Tool (`main.py`)
- Fast single-command processing for single pages and multi-page batches.

---

## 🔬 How the CV Pipeline Works

```
Photo / Webcam Feed
   │
   ▼
1. Page-Boundary Detection   (edge_detect.py)
   Grayscale -> Gaussian Blur (5x5) -> Canny Edges (50, 150) -> Dilate (5x5)
   -> cv2.findContours -> Douglas-Peucker Polygon Approximation (approxPolyDP)
   │
   ▼
2. Perspective Homography    (perspective.py)
   Orders corners [TL, TR, BR, BL], computes 3x3 homography matrix (getPerspectiveTransform),
   and warps quad into a top-down orthogonal plane (warpPerspective)
   │
   ▼
3. Illumination Correction   (enhance.py)
   Low-pass Gaussian filter (sigma=25) models uneven ambient lighting/shadows;
   divides original signal by background model: R(x, y) = 255 * (I / B)
   │
   ▼
4. Contrast & Sharpness      (enhance.py)
   Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and unsharp masking
   │
   ▼
5. Binarization & Deskew     (enhance.py & perspective.py)
   Adaptive Gaussian thresholding + fine orientation correction via minAreaRect
   │
   ▼
6. Tesseract OCR & PDF       (ocr.py)
   Extracts text with confidence scores and embeds an invisible pixel-aligned text layer
   into a standard PDF via Tesseract's image_to_pdf_or_hocr renderer
```

---

## 📁 Project Structure

```
OpenCV-Doc-Scanner/
├── gui.py                  # Full desktop application GUI
├── main.py                 # CLI entry point
├── photo.jpg               # Sample test document
├── angled_sample.jpg       # Sample angled document photo for demos
├── requirements.txt        # Python package dependencies
├── scanner/
│   ├── __init__.py
│   ├── edge_detect.py      # Page contour detection & 4-corner ordering
│   ├── perspective.py      # Homography warp & fine deskew
│   ├── enhance.py          # Shadow division, CLAHE, sharpening, binarization
│   ├── ocr.py              # Tesseract OCR & searchable PDF export
│   └── pipeline.py         # Full orchestrator & 9-stage CV visualizer
└── out/                    # Default output folder for scans
```

---

## 🚀 Setup & Installation

### 1. Install Tesseract OCR Engine
- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Ubuntu / Debian:**
  ```bash
  sudo apt update && sudo apt install -y tesseract-ocr
  ```
- **Windows:**
  Download and install the official installer from [UB-Mannheim Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki). Add `tesseract.exe` to your system PATH.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Launch Desktop GUI (Recommended for Viva / Demos)
```bash
python gui.py
```
*(Or `python3 gui.py`)*

#### Quick Viva Demo Workflow:
1. Click **"📁 Load Sample"** (loads `photo.jpg` or open any photo/use `📷 Live Camera`).
2. Show the **"1. Boundary & Crop"** tab: Drag the corner points to demonstrate manual CamScanner-style corner refinement with the precision magnifying loupe!
3. Click **"⚡ Scan Page"**: Observe the status bar and the instant switch to the **"2. Scanned Result"** tab.
4. Toggle **"Side-by-Side (Original vs Scan)"** to show the before-and-after comparison.
5. Click **"3. CV Pipeline Stages"**: Walk your professor through each of the 9 stages (Grayscale, Canny edges, Dilation, Homography, Background shadow model, Normalized signal, and CLAHE/Binarization) using the on-screen explanation cards.
6. In the right panel, highlight the **OCR Confidence gauge**, search for words in the text box, and click **"📑 Export Searchable PDF"**!

### CLI Usage

**Single page scan:**
```bash
python main.py --input photo.jpg --output out/
```

**Choose enhancement mode (`bw`, `gray`, `color`):**
```bash
python main.py --input photo.jpg --output out/ --mode gray
```

**Multi-page document → Combined searchable PDF:**
```bash
python main.py --input page1.jpg page2.jpg page3.jpg --output out/ --batch
```

**Non-English language (e.g., Hindi or German if tessdata installed):**
```bash
python main.py --input photo.jpg --output out/ --lang eng+hin
```

---

## 🎓 Viva Questions & Answers (Exam / Presentation Prep)

> **Q1: Why do you downscale the image during edge detection?**  
> *A:* Downscaling (to height 700px) significantly speeds up Canny gradient computation and contours processing. It also acts as an implicit low-pass filter, smoothing away paper fiber micro-textures that would otherwise generate spurious edge noise. Once corners are located, they are scaled back to the original resolution by multiplying by `ratio`.

> **Q2: How does the 4-point perspective transform mathematically work?**  
> *A:* Perspective projection is modeled by a $3 \times 3$ Homography matrix $M$ with 8 degrees of freedom:  
> $\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim M \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$  
> Given 4 non-collinear point correspondences $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$, OpenCV solves the linear system using `cv2.getPerspectiveTransform()` and maps the quad into a flat rectangular raster with `cv2.warpPerspective()`.

> **Q3: How does your illumination correction eliminate shadows without a blank reference sheet?**  
> *A:* It uses a Gaussian blur background model with a large spatial variance ($\sigma = 25$). High-frequency text details are blurred away, leaving only the low-frequency ambient lighting and shadow field $B(x, y)$. Dividing the input image $I(x, y)$ by $B(x, y)$ normalizes the paper reflectance across the entire document.

> **Q4: What is the difference between a simple scanned image + text file and a Searchable PDF?**  
> *A:* A searchable PDF embeds an invisible, pixel-aligned text layer directly beneath the document image (`pytesseract.image_to_pdf_or_hocr`). This allows users to search, highlight, and copy text directly in any PDF reader while preserving the authentic visual appearance of the scanned document.

---

## 📄 License
MIT License. Created for Computer Vision & Document Intelligence coursework and research demonstrations.
