"""
Document Scanner & OCR - Desktop GUI
=====================================
A small Tkinter front-end over the same scanner pipeline, meant for a
live demo/viva: pick a photo, see the detected boundary, the cleaned
scan, and the extracted text, then export a searchable PDF.

Run:
    python gui.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from scanner.pipeline import process_image


class ScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Scanner & OCR")
        self.root.geometry("1050x680")

        self.input_path = None
        self.output_dir = os.path.join(os.getcwd(), "scanned_output")
        self.result = None

        self._build_ui()

    # ---------- UI layout ----------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="1. Choose Photo", command=self.choose_file).pack(side="left")
        self.mode_var = tk.StringVar(value="bw")
        ttk.Label(top, text="   Mode:").pack(side="left")
        ttk.Combobox(top, textvariable=self.mode_var, values=["bw", "gray", "color"],
                     width=8, state="readonly").pack(side="left")
        ttk.Label(top, text="   Language:").pack(side="left")
        self.lang_var = tk.StringVar(value="eng")
        ttk.Entry(top, textvariable=self.lang_var, width=8).pack(side="left")

        self.scan_btn = ttk.Button(top, text="2. Scan Document", command=self.run_scan,
                                    state="disabled")
        self.scan_btn.pack(side="left", padx=10)

        self.export_btn = ttk.Button(top, text="3. Export PDF", command=self.export_pdf,
                                      state="disabled")
        self.export_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Choose a photo of a document to begin.")
        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0)).pack(fill="x")

        # Image panels
        panels = ttk.Frame(self.root, padding=10)
        panels.pack(fill="both", expand=True)

        self.original_panel = self._make_panel(panels, "Original")
        self.contour_panel = self._make_panel(panels, "Detected Boundary")
        self.enhanced_panel = self._make_panel(panels, "Enhanced Scan")

        # Text output
        text_frame = ttk.LabelFrame(self.root, text="Extracted Text", padding=5)
        text_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.text_box = tk.Text(text_frame, height=8, wrap="word")
        self.text_box.pack(fill="both", expand=True)

    def _make_panel(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.pack(side="left", fill="both", expand=True, padx=5)
        label = ttk.Label(frame)
        label.pack(fill="both", expand=True)
        return label

    # ---------- Actions ----------
    def choose_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.input_path = path
        self._show_image(self.original_panel, cv2.imread(path))
        self.scan_btn["state"] = "normal"
        self.status_var.set(f"Loaded: {os.path.basename(path)}. Click 'Scan Document'.")

    def run_scan(self):
        self.status_var.set("Processing... (edge detection -> warp -> enhance -> OCR)")
        self.scan_btn["state"] = "disabled"
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        try:
            result = process_image(self.input_path, self.output_dir,
                                    mode=self.mode_var.get(), lang=self.lang_var.get())
            self.result = result
            self.root.after(0, self._on_scan_done)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.scan_btn.__setitem__("state", "normal"))

    def _on_scan_done(self):
        self._show_image(self.contour_panel, self.result.debug_contour)
        enhanced = self.result.enhanced
        if enhanced.ndim == 2:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_bgr = enhanced
        self._show_image(self.enhanced_panel, enhanced_bgr)

        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, self.result.text)

        self.status_var.set(f"Done. OCR confidence: {self.result.confidence:.1f}%  "
                             f"(files saved to {self.output_dir})")
        self.scan_btn["state"] = "normal"
        self.export_btn["state"] = "normal"

    def export_pdf(self):
        dest = filedialog.asksaveasfilename(defaultextension=".pdf",
                                             filetypes=[("PDF", "*.pdf")])
        if not dest:
            return
        src = os.path.join(self.output_dir, "document.pdf")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dest)
            messagebox.showinfo("Exported", f"Searchable PDF saved to:\n{dest}")
        else:
            messagebox.showerror("Error", "No scanned PDF found -- scan a document first.")

    def _show_image(self, label_widget, cv_image, max_size=320):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = max_size / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        pil_img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img  # keep reference


if __name__ == "__main__":
    root = tk.Tk()
    app = ScannerApp(root)
    root.mainloop()
