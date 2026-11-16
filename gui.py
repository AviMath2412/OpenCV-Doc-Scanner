"""
Document Scanner & OCR - Professional Desktop Application (GUI)
================================================================
A feature-packed, academic-grade OpenCV Document Scanner GUI designed
for live demonstrations, viva presentations, and real-world document digitization.

Core Capabilities:
1. Multi-Page / Batch Document Processing (add/reorder pages, export combined PDF).
2. Interactive Draggable Quad Boundary Detection (CamScanner style corner adjustment with loupe magnifier).
3. 9-Stage Computer Vision Pipeline Inspector (shows exact mathematical stages & OpenCV functions for viva/demos).
4. Live Webcam / Camera Document Scanner with real-time contour overlay.
5. Multiple Enhancement Modes: B&W Scan, Smooth Grayscale, Vivid Color, Original Warped.
6. Real-time Fine-Tuning: Brightness, Contrast, Sharpness, Adaptive Binarization, Shadow Removal, Deskew.
7. Tesseract OCR with live confidence gauge, word/char counts, keyword search, copy to clipboard, and searchable PDF export.
"""

import os
import sys
import io
import threading
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import scanner core pipeline
from scanner.pipeline import process_image, get_pipeline_stages, ScanResult
from scanner.edge_detect import find_document_contour, draw_contour_debug
from scanner.perspective import four_point_transform, fine_deskew
from scanner.enhance import enhance_document
from scanner.ocr import images_to_searchable_pdf, image_to_searchable_pdf


# -----------------------------------------------------------------------------
# Color Palette & Styling Constants
# -----------------------------------------------------------------------------
BG_COLOR = "#0F172A"         # Slate 900
PANEL_BG = "#1E293B"         # Slate 800
CARD_BG = "#334155"          # Slate 700
ACCENT_BLUE = "#38BDF8"      # Sky 400
ACCENT_GREEN = "#34D399"     # Emerald 400
ACCENT_AMBER = "#FBBF24"     # Amber 400
TEXT_WHITE = "#F8FAFC"       # Slate 50
TEXT_MUTED = "#94A3B8"       # Slate 400
LINE_COLOR = "#00E676"       # Bright neon emerald for contour boundary
HANDLE_RADIUS = 10           # Radius for interactive quad handles


# -----------------------------------------------------------------------------
# Data Model for a Document Page
# -----------------------------------------------------------------------------
class PageItem:
    def __init__(self, image: np.ndarray, file_path: Optional[str] = None):
        self.file_path: Optional[str] = file_path
        self.image: np.ndarray = image.copy()   # Current oriented image (BGR)
        self.rotation_angle: int = 0            # 0, 90, 180, 270
        self.corners: np.ndarray = find_document_contour(self.image)
        
        # Enhancement & Filter options
        self.mode: str = "bw"
        self.contrast: float = 1.0
        self.brightness: int = 0
        self.sharpness: float = 1.0
        self.fine_deskew: bool = True
        self.remove_shadows: bool = True
        self.adaptive_block_size: int = 25
        self.adaptive_c: int = 15
        self.lang: str = "eng"
        
        # Output Scan Result
        self.result: Optional[ScanResult] = None
        self.is_processed: bool = False
        self.thumbnail: Optional[ImageTk.PhotoImage] = None


# -----------------------------------------------------------------------------
# Interactive Draggable Crop Canvas with Magnifying Loupe
# -----------------------------------------------------------------------------
class InteractiveCropCanvas(tk.Frame):
    """
    An interactive OpenCV canvas allowing users to inspect detected document boundaries,
    drag 4 corner handles (TL, TR, BR, BL) in real time with a precision magnifying loupe,
    rotate orientation, and reset/auto-detect boundaries.
    """
    def __init__(self, parent, on_change_callback=None):
        super().__init__(parent, bg=BG_COLOR)
        self.on_change_callback = on_change_callback

        self.orig_image: Optional[np.ndarray] = None
        self.corners: Optional[np.ndarray] = None   # 4x2 float32 in orig image coords
        
        self.display_scale: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        
        self.drag_point_idx: Optional[int] = None
        self.hover_point_idx: Optional[int] = None
        self.drag_start_coords: Optional[Tuple[float, float]] = None
        self.corners_start_drag: Optional[np.ndarray] = None
        self.is_dragging_box: bool = False
        self.current_mouse_pos: Tuple[int, int] = (0, 0)
        self.tk_image = None

        # Build Canvas
        self.canvas = tk.Canvas(self, bg="#111827", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Canvas Mouse Bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>", self._on_mouse_move)

    def set_image_and_corners(self, image: np.ndarray, corners: Optional[np.ndarray] = None):
        self.orig_image = image.copy()
        if corners is None:
            self.corners = find_document_contour(self.orig_image)
        else:
            self.corners = corners.copy().astype(np.float32)
        self.redraw()

    def get_corners(self) -> Optional[np.ndarray]:
        return self.corners.copy() if self.corners is not None else None

    def get_image(self) -> Optional[np.ndarray]:
        return self.orig_image.copy() if self.orig_image is not None else None

    def auto_detect(self):
        if self.orig_image is None:
            return
        self.corners = find_document_contour(self.orig_image)
        self.redraw()
        if self.on_change_callback:
            self.on_change_callback()

    def select_full_image(self):
        if self.orig_image is None:
            return
        h, w = self.orig_image.shape[:2]
        self.corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        self.redraw()
        if self.on_change_callback:
            self.on_change_callback()

    def rotate_image(self, direction: int):
        """direction: 1 for 90 CW, -1 for 90 CCW"""
        if self.orig_image is None or self.corners is None:
            return
        orig_h, orig_w = self.orig_image.shape[:2]
        
        if direction == 1:
            self.orig_image = cv2.rotate(self.orig_image, cv2.ROTATE_90_CLOCKWISE)
            # (x, y) -> (orig_h - 1 - y, x)
            new_corners = np.zeros_like(self.corners)
            for i in range(4):
                x, y = self.corners[i]
                new_corners[i] = [orig_h - 1 - y, x]
        else:
            self.orig_image = cv2.rotate(self.orig_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # (x, y) -> (y, orig_w - 1 - x)
            new_corners = np.zeros_like(self.corners)
            for i in range(4):
                x, y = self.corners[i]
                new_corners[i] = [y, orig_w - 1 - x]

        # Re-sort points [TL, TR, BR, BL]
        from scanner.edge_detect import _order_points
        self.corners = _order_points(new_corners)
        self.redraw()
        if self.on_change_callback:
            self.on_change_callback()

    def _on_resize(self, event=None):
        self.redraw()

    def _to_canvas_coords(self, orig_pt: np.ndarray) -> Tuple[float, float]:
        cx = orig_pt[0] * self.display_scale + self.offset_x
        cy = orig_pt[1] * self.display_scale + self.offset_y
        return cx, cy

    def _to_orig_coords(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        ox = (canvas_x - self.offset_x) / self.display_scale
        oy = (canvas_y - self.offset_y) / self.display_scale
        if self.orig_image is not None:
            h, w = self.orig_image.shape[:2]
            ox = max(0.0, min(float(w - 1), ox))
            oy = max(0.0, min(float(h - 1), oy))
        return ox, oy

    def redraw(self):
        self.canvas.delete("all")
        if self.orig_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 300,
                self.canvas.winfo_height() // 2 or 200,
                text="No document loaded.\nClick 'Open Image(s)', 'Live Camera', or 'Load Sample'.",
                fill=TEXT_MUTED, font=("Helvetica", 14), justify="center"
            )
            return

        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        ih, iw = self.orig_image.shape[:2]

        scale_w = (cw - 40) / iw
        scale_h = (ch - 40) / ih
        self.display_scale = min(scale_w, scale_h)

        disp_w = int(iw * self.display_scale)
        disp_h = int(ih * self.display_scale)
        self.offset_x = (cw - disp_w) // 2
        self.offset_y = (ch - disp_h) // 2

        # Resize and render background image
        rgb = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        # Draw Quad Boundary Lines
        if self.corners is not None and len(self.corners) == 4:
            canvas_pts = [self._to_canvas_coords(pt) for pt in self.corners]

            # Translucent boundary polygon
            poly_coords = []
            for pt in canvas_pts:
                poly_coords.extend(pt)
            self.canvas.create_polygon(
                poly_coords,
                outline=LINE_COLOR,
                width=3,
                fill=LINE_COLOR,
                stipple="gray25"  # Semi-transparent overlay effect in Tkinter
            )

            # Draw Handles for each corner
            colors = ["#10B981", "#0EA5E9", "#F59E0B", "#8B5CF6"]  # TL, TR, BR, BL
            labels = ["1: TL", "2: TR", "3: BR", "4: BL"]

            for i, (cx, cy) in enumerate(canvas_pts):
                r = HANDLE_RADIUS + (2 if i == self.hover_point_idx else 0)
                # Outer white halo
                self.canvas.create_oval(cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2,
                                        fill="#FFFFFF", outline="#000000", width=1)
                # Inner colored circle
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                        fill=colors[i], outline="#FFFFFF", width=2)
                # Tag / label badge
                self.canvas.create_text(cx, cy - r - 10, text=labels[i],
                                        fill="#FFFFFF", font=("Helvetica", 9, "bold"))

        # Draw Loupe / Magnifying Lens if dragging a corner
        if self.drag_point_idx is not None and 0 <= self.drag_point_idx < 4:
            self._draw_loupe(self.drag_point_idx)

    def _draw_loupe(self, pt_idx: int):
        """Draws a real-time magnified inspection circle near the dragged point."""
        ox, oy = self.corners[pt_idx]
        cx, cy = self._to_canvas_coords(self.corners[pt_idx])
        
        loupe_size = 110
        zoom = 2.5
        radius = int((loupe_size / 2) / zoom)
        
        ih, iw = self.orig_image.shape[:2]
        x1 = max(0, int(ox - radius))
        y1 = max(0, int(oy - radius))
        x2 = min(iw, int(ox + radius))
        y2 = min(ih, int(oy + radius))
        
        patch = self.orig_image[y1:y2, x1:x2]
        if patch.size == 0:
            return
            
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_zoom = cv2.resize(patch_rgb, (loupe_size, loupe_size), interpolation=cv2.INTER_LINEAR)
        
        # Draw red crosshair in center of zoom
        ch = loupe_size // 2
        cv2.line(patch_zoom, (ch - 8, ch), (ch + 8, ch), (255, 0, 0), 2)
        cv2.line(patch_zoom, (ch, ch - 8), (ch, ch + 8), (255, 0, 0), 2)
        cv2.circle(patch_zoom, (ch, ch), 4, (255, 0, 0), 1)

        pil_loupe = Image.fromarray(patch_zoom)
        self.loupe_tk = ImageTk.PhotoImage(pil_loupe)

        # Position loupe offset from cursor
        lx = cx + 50 if cx < self.canvas.winfo_width() - 150 else cx - loupe_size - 50
        ly = cy - 50 if cy > 150 else cy + 50

        # Background frame for loupe
        self.canvas.create_rectangle(lx - 4, ly - 4, lx + loupe_size + 4, ly + loupe_size + 4,
                                     fill="#0F172A", outline=ACCENT_BLUE, width=2)
        self.canvas.create_image(lx, ly, anchor="nw", image=self.loupe_tk)
        self.canvas.create_text(lx + loupe_size // 2, ly + loupe_size + 12,
                                text=f"({int(ox)}, {int(oy)})", fill=TEXT_WHITE, font=("Helvetica", 8))

    def _on_mouse_down(self, event):
        if self.corners is None or self.orig_image is None:
            return

        canvas_pts = [self._to_canvas_coords(pt) for pt in self.corners]
        self.drag_point_idx = None
        self.is_dragging_box = False

        # Hit test corner handles (within 20 px)
        for i, (cx, cy) in enumerate(canvas_pts):
            dist = np.hypot(event.x - cx, event.y - cy)
            if dist <= 22:
                self.drag_point_idx = i
                self.canvas.config(cursor="crosshair")
                self.redraw()
                return

        # Hit test inside polygon to drag entire selection
        poly_pts = np.array(canvas_pts, dtype=np.int32)
        inside = cv2.pointPolygonTest(poly_pts, (float(event.x), float(event.y)), measureDist=False)
        if inside >= 0:
            self.is_dragging_box = True
            self.drag_start_coords = (event.x, event.y)
            self.corners_start_drag = self.corners.copy()
            self.canvas.config(cursor="fleur")

    def _on_mouse_drag(self, event):
        if self.corners is None or self.orig_image is None:
            return

        if self.drag_point_idx is not None:
            ox, oy = self._to_orig_coords(event.x, event.y)
            self.corners[self.drag_point_idx] = [ox, oy]
            self.redraw()
            if self.on_change_callback:
                self.on_change_callback()
        elif self.is_dragging_box and self.drag_start_coords is not None:
            dx = (event.x - self.drag_start_coords[0]) / self.display_scale
            dy = (event.y - self.drag_start_coords[1]) / self.display_scale
            ih, iw = self.orig_image.shape[:2]

            new_corners = self.corners_start_drag.copy()
            new_corners[:, 0] += dx
            new_corners[:, 1] += dy

            # Check boundaries
            if (np.all(new_corners[:, 0] >= 0) and np.all(new_corners[:, 0] < iw) and
                    np.all(new_corners[:, 1] >= 0) and np.all(new_corners[:, 1] < ih)):
                self.corners = new_corners
                self.redraw()
                if self.on_change_callback:
                    self.on_change_callback()

    def _on_mouse_up(self, event):
        if self.drag_point_idx is not None or self.is_dragging_box:
            self.drag_point_idx = None
            self.is_dragging_box = False
            self.drag_start_coords = None
            self.corners_start_drag = None
            self.canvas.config(cursor="arrow")
            self.redraw()

    def _on_mouse_move(self, event):
        if self.corners is None:
            return
        canvas_pts = [self._to_canvas_coords(pt) for pt in self.corners]
        hover_idx = None
        for i, (cx, cy) in enumerate(canvas_pts):
            if np.hypot(event.x - cx, event.y - cy) <= 20:
                hover_idx = i
                break

        if hover_idx != self.hover_point_idx:
            self.hover_point_idx = hover_idx
            self.canvas.config(cursor="crosshair" if hover_idx is not None else "arrow")
            self.redraw()


# -----------------------------------------------------------------------------
# Live Camera / Webcam Scanner Dialog
# -----------------------------------------------------------------------------
class CameraCaptureDialog(tk.Toplevel):
    """
    Dedicated dialog that streams live webcam frames, displays real-time document
    boundary detection overlay, and allows single-click document capture.
    """
    def __init__(self, parent, on_capture_callback):
        super().__init__(parent)
        self.title("📷 Live Webcam Document Scanner")
        self.geometry("760x620")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)
        self.on_capture_callback = on_capture_callback

        self.cap = None
        self.is_running = False
        self.current_frame = None

        self._build_ui()
        self._start_camera()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        header = tk.Frame(self, bg=PANEL_BG, pady=10, padx=15)
        header.pack(fill="x")
        tk.Label(header, text="Position document under camera. Detected boundary will highlight in green.",
                 fg=TEXT_WHITE, bg=PANEL_BG, font=("Helvetica", 11)).pack(side="left")

        self.canvas = tk.Canvas(self, bg="#000000", width=720, height=500, highlightthickness=0)
        self.canvas.pack(padx=20, pady=10)

        bottom = tk.Frame(self, bg=BG_COLOR, pady=10)
        bottom.pack(fill="x")

        tk.Button(bottom, text="📸 Capture Page", font=("Helvetica", 12, "bold"),
                  bg=ACCENT_GREEN, fg="#000000", activebackground="#10B981", padx=20, pady=6,
                  relief="flat", command=self._capture).pack(side="left", padx=30)

        tk.Button(bottom, text="Cancel", font=("Helvetica", 11),
                  bg=CARD_BG, fg=TEXT_WHITE, padx=15, pady=6,
                  relief="flat", command=self._on_close).pack(side="right", padx=30)

    def _start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Unable to access system camera. Please verify camera permissions.")
            self.is_running = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Camera Error", f"Could not initialize camera:\n{e}", parent=self)
            self._on_close()

    def _capture_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.current_frame = frame.copy()

            # Real-time contour detection on camera stream
            corners = find_document_contour(frame, resize_height=500)
            annotated = draw_contour_debug(frame, corners)

            # Convert to PhotoImage
            disp = cv2.resize(annotated, (720, 500))
            disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(disp_rgb)
            tk_img = ImageTk.PhotoImage(pil_img)

            # Update UI safely
            self.after(0, self._update_canvas, tk_img)

    def _update_canvas(self, tk_img):
        if not self.is_running:
            return
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.canvas.image = tk_img

    def _capture(self):
        if self.current_frame is not None:
            captured = self.current_frame.copy()
            self._on_close()
            self.on_capture_callback(captured)
        else:
            messagebox.showwarning("Warning", "No frame available to capture.", parent=self)

    def _on_close(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.destroy()


# -----------------------------------------------------------------------------
# Main Application Window
# -----------------------------------------------------------------------------
class DocumentScannerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Document Scanner & OCR Engine - OpenCV Project")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 700)
        self.root.configure(bg=BG_COLOR)

        self.pages: List[PageItem] = []
        self.active_page_idx: int = -1
        self.output_dir: str = os.path.join(os.getcwd(), "scanned_output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.available_langs = self._detect_languages()
        self.is_processing = False

        self._setup_styles()
        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()

        # Load initial sample photo if present
        if os.path.exists("photo.jpg"):
            self.load_image_file("photo.jpg")

    def _detect_languages(self) -> List[str]:
        try:
            import pytesseract
            langs = pytesseract.get_languages()
            if langs:
                return sorted(langs)
        except Exception:
            pass
        return ["eng"]

    # -------------------------------------------------------------------------
    # Styles & Theming
    # -------------------------------------------------------------------------
    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", background=BG_COLOR, foreground=TEXT_WHITE)
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("Panel.TFrame", background=PANEL_BG)
        self.style.configure("Card.TFrame", background=CARD_BG)

        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_WHITE, font=("Helvetica", 10))
        self.style.configure("Header.TLabel", font=("Helvetica", 13, "bold"), foreground=ACCENT_BLUE)
        self.style.configure("Sub.TLabel", font=("Helvetica", 9), foreground=TEXT_MUTED)

        self.style.configure("TButton", font=("Helvetica", 10), padding=5, background=CARD_BG, foreground=TEXT_WHITE)
        self.style.map("TButton", background=[("active", ACCENT_BLUE)], foreground=[("active", "#000000")])

        self.style.configure("Primary.TButton", font=("Helvetica", 10, "bold"), background=ACCENT_BLUE, foreground="#000000")
        self.style.map("Primary.TButton", background=[("active", "#0284C7")])

        self.style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=PANEL_BG, foreground=TEXT_WHITE, padding=(12, 6), font=("Helvetica", 10))
        self.style.map("TNotebook.Tab", background=[("selected", CARD_BG)], foreground=[("selected", ACCENT_BLUE)])

    # -------------------------------------------------------------------------
    # Menu & Shortcuts
    # -------------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=PANEL_BG, fg=TEXT_WHITE)
        
        file_menu = tk.Menu(menubar, tearoff=0, bg=PANEL_BG, fg=TEXT_WHITE)
        file_menu.add_command(label="Open Image(s)... (Cmd+O)", command=self.choose_files)
        file_menu.add_command(label="Live Webcam Scanner...", command=self.open_camera)
        file_menu.add_command(label="Load Sample Image (photo.jpg)", command=lambda: self.load_image_file("photo.jpg"))
        file_menu.add_separator()
        file_menu.add_command(label="Export Searchable PDF... (Cmd+S)", command=self.export_current_pdf)
        file_menu.add_command(label="Export Multi-Page PDF...", command=self.export_batch_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        process_menu = tk.Menu(menubar, tearoff=0, bg=PANEL_BG, fg=TEXT_WHITE)
        process_menu.add_command(label="Scan Current Page (Cmd+P)", command=self.run_current_scan)
        process_menu.add_command(label="Scan All Pages (Batch)", command=self.run_batch_scan)
        process_menu.add_separator()
        process_menu.add_command(label="Auto-Detect Corners (Cmd+R)", command=self._on_auto_detect)
        process_menu.add_command(label="Select Full Frame", command=self._on_full_frame)
        menubar.add_cascade(label="Process", menu=process_menu)

        help_menu = tk.Menu(menubar, tearoff=0, bg=PANEL_BG, fg=TEXT_WHITE)
        help_menu.add_command(label="About CV Pipeline & Algorithms...", command=self.show_about_dialog)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _bind_shortcuts(self):
        self.root.bind("<Command-o>", lambda e: self.choose_files())
        self.root.bind("<Control-o>", lambda e: self.choose_files())
        self.root.bind("<Command-p>", lambda e: self.run_current_scan())
        self.root.bind("<Control-p>", lambda e: self.run_current_scan())
        self.root.bind("<Command-s>", lambda e: self.export_current_pdf())
        self.root.bind("<Control-s>", lambda e: self.export_current_pdf())
        self.root.bind("<Command-r>", lambda e: self._on_auto_detect())
        self.root.bind("<Control-r>", lambda e: self._on_auto_detect())

    # -------------------------------------------------------------------------
    # Layout Construction
    # -------------------------------------------------------------------------
    def _build_ui(self):
        # 1. Top Quick Action Toolbar
        top_bar = tk.Frame(self.root, bg=PANEL_BG, pady=8, padx=12)
        top_bar.pack(side="top", fill="x")

        # App Title & Badge
        title_box = tk.Frame(top_bar, bg=PANEL_BG)
        title_box.pack(side="left", padx=(0, 20))
        tk.Label(title_box, text="📄 OpenCV Document Scanner & OCR", font=("Helvetica", 13, "bold"),
                 fg=TEXT_WHITE, bg=PANEL_BG).pack(anchor="w")
        tk.Label(title_box, text="Perspective Rectification • Illumination Flattening • Searchable PDF",
                 font=("Helvetica", 9), fg=TEXT_MUTED, bg=PANEL_BG).pack(anchor="w")

        # Action Buttons
        btn_box = tk.Frame(top_bar, bg=PANEL_BG)
        btn_box.pack(side="left", fill="y")

        tk.Button(btn_box, text="📂 Open Image(s)", bg=CARD_BG, fg=TEXT_WHITE,
                  relief="flat", padx=10, pady=4, command=self.choose_files).pack(side="left", padx=4)
        tk.Button(btn_box, text="📷 Live Camera", bg=CARD_BG, fg=TEXT_WHITE,
                  relief="flat", padx=10, pady=4, command=self.open_camera).pack(side="left", padx=4)
        tk.Button(btn_box, text="📁 Load Sample", bg=CARD_BG, fg=TEXT_WHITE,
                  relief="flat", padx=10, pady=4, command=lambda: self.load_image_file("photo.jpg")).pack(side="left", padx=4)

        self.scan_btn = tk.Button(btn_box, text="⚡ Scan Page", bg=ACCENT_BLUE, fg="#000000",
                                  font=("Helvetica", 10, "bold"), relief="flat", padx=12, pady=4,
                                  command=self.run_current_scan)
        self.scan_btn.pack(side="left", padx=10)

        self.batch_scan_btn = tk.Button(btn_box, text="⚡ Scan All Pages", bg=CARD_BG, fg=ACCENT_GREEN,
                                        font=("Helvetica", 10, "bold"), relief="flat", padx=10, pady=4,
                                        command=self.run_batch_scan)
        self.batch_scan_btn.pack(side="left", padx=4)

        # Right side status
        self.progress_bar = ttk.Progressbar(top_bar, mode="indeterminate", length=140)
        self.progress_bar.pack(side="right", padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready. Open an image or use the live camera to begin.")
        self.status_label = tk.Label(top_bar, textvariable=self.status_var, font=("Helvetica", 9),
                                     fg=ACCENT_AMBER, bg=PANEL_BG)
        self.status_label.pack(side="right", padx=10)

        # 2. Main Workspace (3 Split Panes: Left Thumbnails, Center Viewport, Right Controls)
        main_split = tk.PanedWindow(self.root, orient="horizontal", bg=BG_COLOR, bd=0, sashwidth=4)
        main_split.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: Pages List Sidebar
        self.left_pane = tk.Frame(main_split, bg=PANEL_BG, width=200)
        main_split.add(self.left_pane, minsize=180)
        self._build_left_sidebar()

        # Center: Tabbed Viewport (Interactive Boundary, Enhanced Scan, CV Pipeline Stages)
        self.center_pane = tk.Frame(main_split, bg=BG_COLOR)
        main_split.add(self.center_pane, minsize=520)
        self._build_center_viewport()

        # Right: Enhancement Controls & OCR Inspector
        self.right_pane = tk.Frame(main_split, bg=PANEL_BG, width=320)
        main_split.add(self.right_pane, minsize=300)
        self._build_right_sidebar()

    # -------------------------------------------------------------------------
    # Left Sidebar: Page Thumbnails & Document Manager
    # -------------------------------------------------------------------------
    def _build_left_sidebar(self):
        header = tk.Frame(self.left_pane, bg=PANEL_BG, padx=8, pady=8)
        header.pack(fill="x")
        tk.Label(header, text="📑 Pages List", font=("Helvetica", 11, "bold"),
                 fg=TEXT_WHITE, bg=PANEL_BG).pack(side="left")
        self.page_count_lbl = tk.Label(header, text="(0 pages)", font=("Helvetica", 9),
                                       fg=TEXT_MUTED, bg=PANEL_BG)
        self.page_count_lbl.pack(side="right")

        # Scrollable Page Listbox
        list_frame = tk.Frame(self.left_pane, bg=PANEL_BG)
        list_frame.pack(fill="both", expand=True, padx=6, pady=4)

        self.pages_listbox = tk.Listbox(list_frame, bg="#0F172A", fg=TEXT_WHITE,
                                        selectbackground=ACCENT_BLUE, selectforeground="#000000",
                                        font=("Helvetica", 10), bd=0, highlightthickness=1,
                                        highlightcolor=CARD_BG, activestyle="none")
        self.pages_listbox.pack(side="left", fill="both", expand=True)
        self.pages_listbox.bind("<<ListboxSelect>>", self._on_page_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.pages_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.pages_listbox.config(yscrollcommand=scrollbar.set)

        # Page Manager Toolbar
        tools = tk.Frame(self.left_pane, bg=PANEL_BG, pady=6, padx=6)
        tools.pack(fill="x")

        tk.Button(tools, text="+ Add", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", width=6, command=self.choose_files).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(tools, text="- Remove", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", width=7, command=self._remove_current_page).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(tools, text="▲ Up", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", width=5, command=lambda: self._move_page(-1)).grid(row=0, column=2, padx=2, pady=2)
        tk.Button(tools, text="▼ Down", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", width=5, command=lambda: self._move_page(1)).grid(row=0, column=3, padx=2, pady=2)

        tk.Button(self.left_pane, text="Clear All Pages", bg="#450A0A", fg="#FCA5A5", font=("Helvetica", 9),
                  relief="flat", command=self._clear_all_pages).pack(fill="x", padx=6, pady=(0, 6))

    # -------------------------------------------------------------------------
    # Center Viewport: Notebook Tabs (Boundary, Scan Result, CV Stages)
    # -------------------------------------------------------------------------
    def _build_center_viewport(self):
        self.notebook = ttk.Notebook(self.center_pane)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Interactive Boundary & Crop
        self.tab_boundary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_boundary, text=" 1. Boundary & Crop ")
        self._build_boundary_tab()

        # Tab 2: Enhanced Scan Result
        self.tab_enhanced = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_enhanced, text=" 2. Scanned Result ")
        self._build_enhanced_tab()

        # Tab 3: CV Pipeline Stages (Viva / Demo Explainer)
        self.tab_stages = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_stages, text=" 3. CV Pipeline Stages ")
        self._build_stages_tab()

    def _build_boundary_tab(self):
        # Interactive Canvas
        self.crop_canvas = InteractiveCropCanvas(self.tab_boundary, on_change_callback=self._on_crop_changed)
        self.crop_canvas.pack(fill="both", expand=True)

        # Canvas bottom toolbar
        tb = tk.Frame(self.tab_boundary, bg=PANEL_BG, pady=6, padx=10)
        tb.pack(fill="x")

        tk.Label(tb, text="Tip: Drag colored corners 1-4 to refine document boundary.",
                 fg=TEXT_MUTED, bg=PANEL_BG, font=("Helvetica", 9, "italic")).pack(side="left")

        tk.Button(tb, text="🎯 Auto-Detect", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=self._on_auto_detect).pack(side="right", padx=4)
        tk.Button(tb, text="🔲 Full Frame", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=self._on_full_frame).pack(side="right", padx=4)
        tk.Button(tb, text="↺ Rotate Left", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=lambda: self.crop_canvas.rotate_image(-1)).pack(side="right", padx=4)
        tk.Button(tb, text="↻ Rotate Right", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=lambda: self.crop_canvas.rotate_image(1)).pack(side="right", padx=4)

    def _build_enhanced_tab(self):
        # Enhanced image viewer canvas
        self.enhanced_canvas = tk.Canvas(self.tab_enhanced, bg="#111827", highlightthickness=0)
        self.enhanced_canvas.pack(fill="both", expand=True)
        self.enhanced_canvas.bind("<Configure>", self._redraw_enhanced)
        self.enhanced_tk = None

        # Bottom toolbar
        tb = tk.Frame(self.tab_enhanced, bg=PANEL_BG, pady=6, padx=10)
        tb.pack(fill="x")

        self.compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(tb, text="Side-by-Side (Original vs Scan)", variable=self.compare_var,
                       fg=TEXT_WHITE, bg=PANEL_BG, selectcolor=PANEL_BG, activebackground=PANEL_BG,
                       command=self._redraw_enhanced).pack(side="left")

        tk.Button(tb, text="💾 Save Image (.png)", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=self.export_image_file).pack(side="right", padx=4)
        tk.Button(tb, text="📋 Copy Image", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=self._copy_image_to_clipboard).pack(side="right", padx=4)

    def _build_stages_tab(self):
        """Builds the 9-stage Computer Vision pipeline educational inspector."""
        top_bar = tk.Frame(self.tab_stages, bg=PANEL_BG, pady=6, padx=10)
        top_bar.pack(fill="x")

        tk.Label(top_bar, text="OpenCV Pipeline Stage:", fg=TEXT_WHITE, bg=PANEL_BG,
                 font=("Helvetica", 10, "bold")).pack(side="left", padx=(0, 6))

        self.stage_selector = ttk.Combobox(top_bar, state="readonly", width=36)
        self.stage_selector.pack(side="left")
        self.stage_selector.bind("<<ComboboxSelected>>", self._on_stage_selected)

        # Stage Viewer Canvas
        self.stage_canvas = tk.Canvas(self.tab_stages, bg="#111827", highlightthickness=0)
        self.stage_canvas.pack(fill="both", expand=True, padx=6, pady=4)
        self.stage_canvas.bind("<Configure>", self._redraw_stage)
        self.stage_tk = None

        # Academic Concept Card (Formula & OpenCV Function explainer)
        self.stage_card = tk.Frame(self.tab_stages, bg=CARD_BG, padx=12, pady=10)
        self.stage_card.pack(fill="x", padx=6, pady=(0, 6))

        self.stage_title_lbl = tk.Label(self.stage_card, text="", font=("Helvetica", 11, "bold"),
                                        fg=ACCENT_BLUE, bg=CARD_BG)
        self.stage_title_lbl.pack(anchor="w")

        self.stage_desc_lbl = tk.Label(self.stage_card, text="", font=("Helvetica", 9),
                                       fg=TEXT_WHITE, bg=CARD_BG, wraplength=700, justify="left")
        self.stage_desc_lbl.pack(anchor="w", pady=(2, 4))

        self.stage_concept_lbl = tk.Label(self.stage_card, text="", font=("Helvetica", 9, "italic"),
                                          fg=ACCENT_GREEN, bg=CARD_BG, wraplength=700, justify="left")
        self.stage_concept_lbl.pack(anchor="w")

    # -------------------------------------------------------------------------
    # Right Sidebar: Filter Controls, OCR Results & PDF Export
    # -------------------------------------------------------------------------
    def _build_right_sidebar(self):
        # 1. Enhancement & Filter Card
        filter_card = tk.LabelFrame(self.right_pane, text=" ⚙️ Enhancement & Filters ",
                                    bg=PANEL_BG, fg=ACCENT_BLUE, font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        filter_card.pack(fill="x", padx=8, pady=6)

        # Mode Selector
        mode_box = tk.Frame(filter_card, bg=PANEL_BG)
        mode_box.pack(fill="x", pady=2)
        tk.Label(mode_box, text="Mode:", fg=TEXT_WHITE, bg=PANEL_BG, width=10, anchor="w").pack(side="left")
        self.mode_var = tk.StringVar(value="bw")
        mode_combo = ttk.Combobox(mode_box, textvariable=self.mode_var,
                                  values=["bw", "gray", "color", "original"],
                                  state="readonly", width=12)
        mode_combo.pack(side="left", fill="x", expand=True)
        mode_combo.bind("<<ComboboxSelected>>", self._on_filter_param_changed)

        # Brightness Slider
        b_box = tk.Frame(filter_card, bg=PANEL_BG)
        b_box.pack(fill="x", pady=2)
        tk.Label(b_box, text="Brightness:", fg=TEXT_WHITE, bg=PANEL_BG, width=10, anchor="w").pack(side="left")
        self.brightness_var = tk.IntVar(value=0)
        b_slider = ttk.Scale(b_box, from_=-50, to=50, variable=self.brightness_var,
                             command=lambda v: self._on_filter_param_changed())
        b_slider.pack(side="left", fill="x", expand=True)

        # Contrast Slider
        c_box = tk.Frame(filter_card, bg=PANEL_BG)
        c_box.pack(fill="x", pady=2)
        tk.Label(c_box, text="Contrast:", fg=TEXT_WHITE, bg=PANEL_BG, width=10, anchor="w").pack(side="left")
        self.contrast_var = tk.DoubleVar(value=1.0)
        c_slider = ttk.Scale(c_box, from_=0.5, to=2.5, variable=self.contrast_var,
                             command=lambda v: self._on_filter_param_changed())
        c_slider.pack(side="left", fill="x", expand=True)

        # Toggles: Fine Deskew & Shadow Removal
        toggles = tk.Frame(filter_card, bg=PANEL_BG)
        toggles.pack(fill="x", pady=4)
        self.deskew_var = tk.BooleanVar(value=True)
        tk.Checkbutton(toggles, text="Fine Deskew", variable=self.deskew_var,
                       fg=TEXT_WHITE, bg=PANEL_BG, selectcolor=PANEL_BG,
                       command=self._on_filter_param_changed).pack(side="left", padx=(0, 10))
        self.shadow_var = tk.BooleanVar(value=True)
        tk.Checkbutton(toggles, text="Shadow Removal", variable=self.shadow_var,
                       fg=TEXT_WHITE, bg=PANEL_BG, selectcolor=PANEL_BG,
                       command=self._on_filter_param_changed).pack(side="left")

        # 2. OCR & Text Extraction Card
        ocr_card = tk.LabelFrame(self.right_pane, text=" 🔍 Tesseract OCR & Text ",
                                 bg=PANEL_BG, fg=ACCENT_GREEN, font=("Helvetica", 10, "bold"),
                                 padx=10, pady=8)
        ocr_card.pack(fill="both", expand=True, padx=8, pady=4)

        # OCR Language & Confidence Meter
        top_ocr = tk.Frame(ocr_card, bg=PANEL_BG)
        top_ocr.pack(fill="x", pady=(0, 4))
        tk.Label(top_ocr, text="Lang:", fg=TEXT_WHITE, bg=PANEL_BG).pack(side="left")
        self.lang_var = tk.StringVar(value="eng")
        lang_combo = ttk.Combobox(top_ocr, textvariable=self.lang_var, values=self.available_langs,
                                  width=8)
        lang_combo.pack(side="left", padx=4)

        self.conf_badge = tk.Label(top_ocr, text="OCR Conf: --%", font=("Helvetica", 9, "bold"),
                                   fg=TEXT_MUTED, bg=CARD_BG, padx=6, pady=2)
        self.conf_badge.pack(side="right")

        # Word & Char Count
        self.stats_lbl = tk.Label(ocr_card, text="0 words • 0 characters", font=("Helvetica", 8),
                                  fg=TEXT_MUTED, bg=PANEL_BG)
        self.stats_lbl.pack(anchor="w", pady=(0, 4))

        # Extracted Text Box with Scrollbar
        text_frame = tk.Frame(ocr_card, bg=PANEL_BG)
        text_frame.pack(fill="both", expand=True)

        self.text_box = tk.Text(text_frame, bg="#0F172A", fg=TEXT_WHITE, insertbackground=TEXT_WHITE,
                                font=("Helvetica", 10), wrap="word", bd=0, padx=6, pady=6)
        self.text_box.pack(side="left", fill="both", expand=True)
        self.text_box.tag_configure("highlight", background="#F59E0B", foreground="#000000")

        txt_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_box.yview)
        txt_scroll.pack(side="right", fill="y")
        self.text_box.config(yscrollcommand=txt_scroll.set)

        # Text Search bar & Copy button
        search_box = tk.Frame(ocr_card, bg=PANEL_BG, pady=4)
        search_box.pack(fill="x")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search_text)
        search_entry = ttk.Entry(search_box, textvariable=self.search_var, font=("Helvetica", 9))
        search_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        search_entry.insert(0, "Search...")
        search_entry.bind("<FocusIn>", lambda e: search_entry.delete(0, "end") if self.search_var.get() == "Search..." else None)

        tk.Button(search_box, text="📋 Copy", bg=CARD_BG, fg=TEXT_WHITE, font=("Helvetica", 9),
                  relief="flat", command=self._copy_text_to_clipboard).pack(side="right")

        # 3. Export Hub Card
        export_card = tk.LabelFrame(self.right_pane, text=" 💾 Export Documents ",
                                    bg=PANEL_BG, fg=ACCENT_AMBER, font=("Helvetica", 10, "bold"),
                                    padx=10, pady=8)
        export_card.pack(fill="x", padx=8, pady=(4, 8))

        self.pdf_btn = tk.Button(export_card, text="📑 Export Searchable PDF (Current)", bg=CARD_BG,
                                 fg=TEXT_WHITE, font=("Helvetica", 9), relief="flat",
                                 command=self.export_current_pdf)
        self.pdf_btn.pack(fill="x", pady=2)

        self.batch_pdf_btn = tk.Button(export_card, text="📚 Export Combined PDF (All Pages)", bg=CARD_BG,
                                       fg=TEXT_WHITE, font=("Helvetica", 9), relief="flat",
                                       command=self.export_batch_pdf)
        self.batch_pdf_btn.pack(fill="x", pady=2)

        tk.Button(export_card, text="📦 Export All Assets Bundle", bg=CARD_BG,
                  fg=TEXT_WHITE, font=("Helvetica", 9), relief="flat",
                  command=self.export_all_bundle).pack(fill="x", pady=2)

    # -------------------------------------------------------------------------
    # Document / Page Management
    # -------------------------------------------------------------------------
    def choose_files(self):
        paths = filedialog.askopenfilenames(
            title="Select Document Photo(s)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if not paths:
            return
        for path in paths:
            self.load_image_file(path)

    def load_image_file(self, path: str):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image from:\n{path}")
            return
        page = PageItem(img, file_path=path)
        self.pages.append(page)
        self._update_pages_list()
        self.set_active_page(len(self.pages) - 1)
        self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def open_camera(self):
        CameraCaptureDialog(self.root, on_capture_callback=self._on_camera_captured)

    def _on_camera_captured(self, frame: np.ndarray):
        page = PageItem(frame, file_path="Camera_Capture.png")
        self.pages.append(page)
        self._update_pages_list()
        self.set_active_page(len(self.pages) - 1)
        self.status_var.set("Captured frame from live camera.")
        # Auto-run scan on camera capture
        self.run_current_scan()

    def _update_pages_list(self):
        self.pages_listbox.delete(0, tk.END)
        for i, page in enumerate(self.pages):
            name = os.path.basename(page.file_path) if page.file_path else f"Page_{i+1}"
            status = " [OK]" if page.is_processed else ""
            self.pages_listbox.insert(tk.END, f"Page {i+1}: {name}{status}")
        self.page_count_lbl.config(text=f"({len(self.pages)} pages)")

    def set_active_page(self, idx: int):
        if not (0 <= idx < len(self.pages)):
            return
        self.active_page_idx = idx
        self.pages_listbox.selection_clear(0, tk.END)
        self.pages_listbox.selection_set(idx)
        self.pages_listbox.see(idx)

        page = self.pages[idx]
        self.crop_canvas.set_image_and_corners(page.image, page.corners)

        # Update controls to match page settings
        self.mode_var.set(page.mode)
        self.brightness_var.set(page.brightness)
        self.contrast_var.set(page.contrast)
        self.deskew_var.set(page.fine_deskew)
        self.shadow_var.set(page.remove_shadows)
        self.lang_var.set(page.lang)

        if page.is_processed and page.result:
            self._display_scan_result(page.result)
        else:
            self._clear_scan_displays()

    def _on_page_selected(self, event=None):
        sel = self.pages_listbox.curselection()
        if sel:
            self.set_active_page(sel[0])

    def _remove_current_page(self):
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        del self.pages[self.active_page_idx]
        self._update_pages_list()
        new_idx = min(self.active_page_idx, len(self.pages) - 1)
        if new_idx >= 0:
            self.set_active_page(new_idx)
        else:
            self.active_page_idx = -1
            self.crop_canvas.orig_image = None
            self.crop_canvas.corners = None
            self.crop_canvas.redraw()
            self._clear_scan_displays()

    def _move_page(self, delta: int):
        idx = self.active_page_idx
        new_idx = idx + delta
        if 0 <= new_idx < len(self.pages) and 0 <= idx < len(self.pages):
            self.pages[idx], self.pages[new_idx] = self.pages[new_idx], self.pages[idx]
            self._update_pages_list()
            self.set_active_page(new_idx)

    def _clear_all_pages(self):
        if not self.pages:
            return
        if messagebox.askyesno("Clear All", "Remove all pages from workspace?"):
            self.pages.clear()
            self.active_page_idx = -1
            self._update_pages_list()
            self.crop_canvas.orig_image = None
            self.crop_canvas.corners = None
            self.crop_canvas.redraw()
            self._clear_scan_displays()

    # -------------------------------------------------------------------------
    # User Interactions & Canvas Events
    # -------------------------------------------------------------------------
    def _on_crop_changed(self):
        if 0 <= self.active_page_idx < len(self.pages):
            page = self.pages[self.active_page_idx]
            page.corners = self.crop_canvas.get_corners()
            page.image = self.crop_canvas.get_image()

    def _on_auto_detect(self):
        self.crop_canvas.auto_detect()

    def _on_full_frame(self):
        self.crop_canvas.select_full_image()

    def _on_filter_param_changed(self, event=None):
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        page.mode = self.mode_var.get()
        page.brightness = self.brightness_var.get()
        page.contrast = self.contrast_var.get()
        page.fine_deskew = self.deskew_var.get()
        page.remove_shadows = self.shadow_var.get()
        page.lang = self.lang_var.get()

        # If already scanned, quickly re-render scan without re-running heavy OCR
        if page.is_processed and page.result:
            self._quick_re_enhance_current()

    def _quick_re_enhance_current(self):
        page = self.pages[self.active_page_idx]
        try:
            enhanced = enhance_document(
                page.result.warped,
                mode=page.mode,
                brightness=page.brightness,
                contrast=page.contrast,
                remove_shadows=page.remove_shadows
            )
            if page.fine_deskew:
                enhanced = fine_deskew(enhanced)
            page.result.enhanced = enhanced
            self._redraw_enhanced()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Core Pipeline Processing & Threading
    # -------------------------------------------------------------------------
    def run_current_scan(self):
        if not (0 <= self.active_page_idx < len(self.pages)):
            messagebox.showinfo("Notice", "Please load an image first.")
            return
        if self.is_processing:
            return

        page = self.pages[self.active_page_idx]
        page.corners = self.crop_canvas.get_corners()
        page.image = self.crop_canvas.get_image()
        page.mode = self.mode_var.get()
        page.lang = self.lang_var.get()
        page.brightness = self.brightness_var.get()
        page.contrast = self.contrast_var.get()
        page.fine_deskew = self.deskew_var.get()
        page.remove_shadows = self.shadow_var.get()

        self.is_processing = True
        self.scan_btn.config(state="disabled")
        self.progress_bar.start(10)
        self.status_var.set("Processing: Rectifying perspective -> Removing shadows -> OCR...")

        threading.Thread(target=self._scan_page_worker, args=(page,), daemon=True).start()

    def _scan_page_worker(self, page: PageItem):
        try:
            result = process_image(
                input_path_or_image=page.image,
                corners=page.corners,
                mode=page.mode,
                lang=page.lang,
                brightness=page.brightness,
                contrast=page.contrast,
                fine_deskew_flag=page.fine_deskew,
                remove_shadows=page.remove_shadows,
                run_ocr=True,
                generate_stages=True
            )
            page.result = result
            page.is_processed = True
            self.root.after(0, self._on_scan_success, result)
        except Exception as e:
            self.root.after(0, self._on_scan_error, str(e))

    def _on_scan_success(self, result: ScanResult):
        self.is_processing = False
        self.scan_btn.config(state="normal")
        self.progress_bar.stop()
        self.status_var.set(f"Done! OCR Confidence: {result.confidence:.1f}%")
        self._update_pages_list()
        self._display_scan_result(result)
        # Automatically switch to enhanced tab to show professor the scan
        self.notebook.select(self.tab_enhanced)

    def _on_scan_error(self, err_msg: str):
        self.is_processing = False
        self.scan_btn.config(state="normal")
        self.progress_bar.stop()
        self.status_var.set("Error during processing.")
        messagebox.showerror("Scan Error", f"An error occurred while scanning:\n{err_msg}")

    def run_batch_scan(self):
        if not self.pages:
            messagebox.showinfo("Notice", "No pages to scan.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.batch_scan_btn.config(state="disabled")
        self.progress_bar.start(10)
        self.status_var.set(f"Batch processing 0/{len(self.pages)} pages...")

        threading.Thread(target=self._batch_scan_worker, daemon=True).start()

    def _batch_scan_worker(self):
        try:
            total = len(self.pages)
            for i, page in enumerate(self.pages):
                self.root.after(0, lambda idx=i+1: self.status_var.set(f"Batch processing {idx}/{total} pages..."))
                if page.corners is None:
                    page.corners = find_document_contour(page.image)
                result = process_image(
                    input_path_or_image=page.image,
                    corners=page.corners,
                    mode=page.mode,
                    lang=page.lang,
                    brightness=page.brightness,
                    contrast=page.contrast,
                    fine_deskew_flag=page.fine_deskew,
                    remove_shadows=page.remove_shadows,
                    run_ocr=True,
                    generate_stages=True
                )
                page.result = result
                page.is_processed = True

            self.root.after(0, self._on_batch_scan_done)
        except Exception as e:
            self.root.after(0, self._on_scan_error, str(e))

    def _on_batch_scan_done(self):
        self.is_processing = False
        self.batch_scan_btn.config(state="normal")
        self.progress_bar.stop()
        self.status_var.set(f"Batch complete: all {len(self.pages)} pages scanned successfully.")
        self._update_pages_list()
        if 0 <= self.active_page_idx < len(self.pages):
            self._display_scan_result(self.pages[self.active_page_idx].result)

    # -------------------------------------------------------------------------
    # Result Visualizers & Stage Inspector
    # -------------------------------------------------------------------------
    def _display_scan_result(self, result: ScanResult):
        # 1. Update Enhanced Scan View
        self._redraw_enhanced()

        # 2. Update OCR Text & Stats
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, result.text)

        words = len(result.text.split())
        chars = len(result.text)
        self.stats_lbl.config(text=f"{words} words • {chars} characters")

        # Confidence gauge with color-coding
        conf = result.confidence
        color = ACCENT_GREEN if conf >= 75 else (ACCENT_AMBER if conf >= 45 else "#EF4444")
        self.conf_badge.config(text=f"OCR Conf: {conf:.1f}%", fg=color)

        # 3. Update Pipeline Stages combo
        if result.stages:
            stage_titles = [s["title"] for s in result.stages]
            self.stage_selector["values"] = stage_titles
            self.stage_selector.current(0)
            self._on_stage_selected()

    def _clear_scan_displays(self):
        self.enhanced_canvas.delete("all")
        self.text_box.delete("1.0", tk.END)
        self.stats_lbl.config(text="0 words • 0 characters")
        self.conf_badge.config(text="OCR Conf: --%", fg=TEXT_MUTED)
        self.stage_canvas.delete("all")
        self.stage_title_lbl.config(text="")
        self.stage_desc_lbl.config(text="")
        self.stage_concept_lbl.config(text="")

    def _redraw_enhanced(self, event=None):
        self.enhanced_canvas.delete("all")
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        if not page.result or page.result.enhanced is None:
            self.enhanced_canvas.create_text(
                self.enhanced_canvas.winfo_width() // 2 or 300,
                self.enhanced_canvas.winfo_height() // 2 or 200,
                text="Click 'Scan Page' to produce an enhanced document.",
                fill=TEXT_MUTED, font=("Helvetica", 13)
            )
            return

        cw = max(self.enhanced_canvas.winfo_width(), 100)
        ch = max(self.enhanced_canvas.winfo_height(), 100)

        enhanced = page.result.enhanced
        if enhanced.ndim == 2:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        if self.compare_var.get() and page.result.warped is not None:
            # Side-by-side view (Original Warped vs Enhanced)
            orig_rgb = cv2.cvtColor(page.result.warped, cv2.COLOR_BGR2RGB)
            h = min(orig_rgb.shape[0], enhanced_rgb.shape[0])
            orig_rgb = cv2.resize(orig_rgb, (int(orig_rgb.shape[1] * (h / orig_rgb.shape[0])), h))
            enhanced_rgb = cv2.resize(enhanced_rgb, (int(enhanced_rgb.shape[1] * (h / enhanced_rgb.shape[0])), h))
            combined = np.hstack([orig_rgb, enhanced_rgb])
        else:
            combined = enhanced_rgb

        ih, iw = combined.shape[:2]
        scale = min((cw - 40) / iw, (ch - 40) / ih)
        disp_w = max(1, int(iw * scale))
        disp_h = max(1, int(ih * scale))

        disp_img = cv2.resize(combined, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(disp_img)
        self.enhanced_tk = ImageTk.PhotoImage(pil_img)
        self.enhanced_canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.enhanced_tk)

    def _on_stage_selected(self, event=None):
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        if not page.result or not page.result.stages:
            return

        idx = self.stage_selector.current()
        if not (0 <= idx < len(page.result.stages)):
            return

        stage = page.result.stages[idx]
        self.stage_title_lbl.config(text=stage["title"])
        self.stage_desc_lbl.config(text=stage["description"])
        self.stage_concept_lbl.config(text=f"CV Concept: {stage['concept']}")
        self._redraw_stage()

    def _redraw_stage(self, event=None):
        self.stage_canvas.delete("all")
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        if not page.result or not page.result.stages:
            return

        idx = self.stage_selector.current()
        if not (0 <= idx < len(page.result.stages)):
            return

        img = page.result.stages[idx]["image"]
        if img is None:
            return
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cw = max(self.stage_canvas.winfo_width(), 100)
        ch = max(self.stage_canvas.winfo_height(), 100)
        ih, iw = img_rgb.shape[:2]

        scale = min((cw - 40) / iw, (ch - 40) / ih)
        disp_w = max(1, int(iw * scale))
        disp_h = max(1, int(ih * scale))

        disp_img = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(disp_img)
        self.stage_tk = ImageTk.PhotoImage(pil_img)
        self.stage_canvas.create_image(cw // 2, ch // 2, anchor="center", image=self.stage_tk)

    # -------------------------------------------------------------------------
    # Text Search & Clipboard
    # -------------------------------------------------------------------------
    def _on_search_text(self, *args):
        query = self.search_var.get().strip()
        self.text_box.tag_remove("highlight", "1.0", tk.END)
        if not query or query == "Search...":
            return
        start_pos = "1.0"
        while True:
            start_pos = self.text_box.search(query, start_pos, stopindex=tk.END, nocase=True)
            if not start_pos:
                break
            end_pos = f"{start_pos}+{len(query)}c"
            self.text_box.tag_add("highlight", start_pos, end_pos)
            start_pos = end_pos

    def _copy_text_to_clipboard(self):
        text = self.text_box.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Notice", "No text to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self.status_var.set("Copied extracted text to clipboard!")

    def _copy_image_to_clipboard(self):
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        if not page.result or page.result.enhanced is None:
            return
        try:
            import subprocess
            import tempfile
            enhanced = page.result.enhanced
            pil_img = Image.fromarray(enhanced if enhanced.ndim == 2 else cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

            if sys.platform == "darwin":
                # Native macOS clipboard copy via AppleScript
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pil_img.save(tmp_path, format="PNG")
                script = f'set the clipboard to (read (POSIX file "{tmp_path}") as «class PNGf»)'
                subprocess.run(["osascript", "-e", script], check=True)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            else:
                output = io.BytesIO()
                pil_img.save(output, format="PNG")
            self.status_var.set("Copied image to clipboard!")
        except Exception:
            self.status_var.set("Image ready in memory.")

    # -------------------------------------------------------------------------
    # Export Capabilities
    # -------------------------------------------------------------------------
    def export_current_pdf(self):
        if not (0 <= self.active_page_idx < len(self.pages)):
            messagebox.showinfo("Notice", "Please scan a document first.")
            return
        page = self.pages[self.active_page_idx]
        if not page.result or page.result.enhanced is None:
            messagebox.showinfo("Notice", "Please click 'Scan Page' before exporting.")
            return

        dest = filedialog.asksaveasfilename(
            title="Export Searchable PDF",
            defaultextension=".pdf",
            filetypes=[("Searchable PDF", "*.pdf")]
        )
        if not dest:
            return

        try:
            image_to_searchable_pdf(page.result.enhanced, dest, lang=page.lang)
            messagebox.showinfo("Success", f"Searchable PDF exported successfully:\n{dest}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export PDF:\n{e}")

    def export_batch_pdf(self):
        scanned_pages = [p for p in self.pages if p.is_processed and p.result]
        if not scanned_pages:
            messagebox.showinfo("Notice", "No processed pages found. Please scan pages first.")
            return

        dest = filedialog.asksaveasfilename(
            title="Export Combined Multi-Page Searchable PDF",
            defaultextension=".pdf",
            filetypes=[("Multi-Page Searchable PDF", "*.pdf")]
        )
        if not dest:
            return

        try:
            images = [p.result.enhanced for p in scanned_pages]
            images_to_searchable_pdf(images, dest, lang=self.lang_var.get())
            messagebox.showinfo("Success", f"Combined {len(images)}-page PDF exported successfully:\n{dest}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export multi-page PDF:\n{e}")

    def export_image_file(self):
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        if not page.result or page.result.enhanced is None:
            return
        dest = filedialog.asksaveasfilename(
            title="Save Clean Scan Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")]
        )
        if not dest:
            return
        cv2.imwrite(dest, page.result.enhanced)
        messagebox.showinfo("Success", f"Image saved to:\n{dest}")

    def export_all_bundle(self):
        if not (0 <= self.active_page_idx < len(self.pages)):
            return
        page = self.pages[self.active_page_idx]
        if not page.result or page.result.enhanced is None:
            messagebox.showinfo("Notice", "Please scan page before exporting bundle.")
            return

        folder = filedialog.askdirectory(title="Choose Directory to Save Presentation Bundle")
        if not folder:
            return

        bundle_dir = os.path.join(folder, "scanner_presentation_bundle")
        os.makedirs(bundle_dir, exist_ok=True)

        cv2.imwrite(os.path.join(bundle_dir, "1_original_photo.png"), page.image)
        cv2.imwrite(os.path.join(bundle_dir, "2_detected_boundary.png"), page.result.debug_contour)
        cv2.imwrite(os.path.join(bundle_dir, "3_perspective_warped.png"), page.result.warped)
        cv2.imwrite(os.path.join(bundle_dir, "4_enhanced_scan.png"), page.result.enhanced)
        with open(os.path.join(bundle_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(page.result.text)
        image_to_searchable_pdf(page.result.enhanced, os.path.join(bundle_dir, "searchable_document.pdf"), lang=page.lang)

        messagebox.showinfo("Bundle Exported", f"All presentation assets exported to:\n{bundle_dir}")

    def show_about_dialog(self):
        info = (
            "OpenCV Document Scanner & OCR Engine\n"
            "------------------------------------\n\n"
            "Academic Pipeline Architecture:\n"
            "1. Edge Detection: Grayscale -> Gaussian Filter -> Canny Edge Operator.\n"
            "2. Contour Extraction: cv2.findContours + Douglas-Peucker approxPolyDP.\n"
            "3. Homography & Rectification: cv2.getPerspectiveTransform + warpPerspective.\n"
            "4. Illumination Flattening: Gaussian background subtraction (sigma=25) to remove shadows.\n"
            "5. Contrast & Binarization: CLAHE + Unsharp Mask + Adaptive Gaussian Thresholding.\n"
            "6. Fine Deskew: Ink pixel orientation via cv2.minAreaRect.\n"
            "7. OCR: Tesseract LSTM Engine with pixel-aligned searchable PDF generation.\n\n"
            "Developed with Python, OpenCV, and Tesseract."
        )
        messagebox.showinfo("About CV Pipeline", info)


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = DocumentScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
