import os
import sys
import time
import threading
from pathlib import Path
import customtkinter as ctk
from tkinter import filedialog, messagebox

def get_root_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))

import datetime
from PIL import Image
from PIL.ExifTags import TAGS

def get_exif_timestamp(path):
    try:
        with Image.open(path) as img:
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    if TAGS.get(tag) == 'DateTimeOriginal':
                        dt = datetime.datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
                        return dt.timestamp()
    except:
        pass
    return None

# Import the patched processor
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import processor_gui as processor

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class CaroCropApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("MisterMobile - Carousell Image Cropper v4.0")
        self.geometry("850x650")
        
        # Grid layout 1x2 (Left sidebar for settings, Right for operations)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.processing = False
        self.stop_requested = False
        self._init_sidebar()
        self._init_main_view()
        
    def _init_sidebar(self):
        # Sidebar Frame
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # Logo / Title
        title = ctk.CTkLabel(self.sidebar, text="SETTINGS", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(padx=20, pady=30)
        
        # --- Brightness ---
        self.bright_lbl = ctk.CTkLabel(self.sidebar, text=f"Brightness: {processor.GLOBAL_BRIGHTNESS:.2f}")
        self.bright_lbl.pack(padx=20, pady=(10,0), anchor="w")
        
        self.bright_slider = ctk.CTkSlider(self.sidebar, from_=1.00, to=1.50, number_of_steps=50, command=self._update_bright)
        self.bright_slider.set(processor.GLOBAL_BRIGHTNESS)
        self.bright_slider.pack(padx=20, pady=(5, 15), fill="x")
        
        # --- Padding ---
        self.pad_lbl = ctk.CTkLabel(self.sidebar, text="Padding: 4%")
        self.pad_lbl.pack(padx=20, pady=(10,0), anchor="w")
        
        self.pad_slider = ctk.CTkSlider(self.sidebar, from_=0.01, to=0.10, number_of_steps=9, command=self._update_pad)
        self.pad_slider.set(0.04)
        self.pad_slider.pack(padx=20, pady=(5, 15), fill="x")
        
        # --- Toggles ---
        self.adaptive_switch = ctk.CTkSwitch(self.sidebar, text="Adaptive Brightness", command=None)
        if processor.ENABLE_ADAPTIVE_BRIGHTNESS_TARGET: self.adaptive_switch.select()
        self.adaptive_switch.pack(padx=20, pady=10, anchor="w")
        
        self.antigray_switch = ctk.CTkSwitch(self.sidebar, text="Anti-Gray Correction", command=None)
        if processor.ENABLE_ANTI_GRAY_CORRECTION: self.antigray_switch.select()
        self.antigray_switch.pack(padx=20, pady=10, anchor="w")
        
        self.barcode_switch = ctk.CTkSwitch(self.sidebar, text="Scan Barcodes (IMEI)", command=None)
        self.barcode_switch.select()
        self.barcode_switch.pack(padx=20, pady=10, anchor="w")

        # --- Reset Button ---
        reset_btn = ctk.CTkButton(self.sidebar, text="RESET DEFAULTS", font=ctk.CTkFont(size=12), fg_color="gray30", hover_color="gray40", command=self._reset_defaults)
        reset_btn.pack(padx=20, pady=(30, 0), fill="x")
        
        # Info Footer
        footer = ctk.CTkLabel(self.sidebar, text="v4.0 CaroCropper", font=ctk.CTkFont(size=11), text_color="gray")
        footer.pack(side="bottom", pady=20)
        
    def _reset_defaults(self):
        self.bright_slider.set(1.16)
        self.pad_slider.set(0.04)
        self._update_bright(1.16)
        self._update_pad(0.04)
        self.adaptive_switch.select()
        self.antigray_switch.select()
        self.barcode_switch.select()

    def _update_bright(self, val):
        self.bright_lbl.configure(text=f"Brightness: {float(val):.2f}")
        
    def _update_pad(self, val):
        self.pad_lbl.configure(text=f"Padding: {int(float(val)*100)}%")

    def _init_main_view(self):
        # Main View Frame
        self.main_view = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_view.grid_columnconfigure(0, weight=1)
        self.main_view.grid_rowconfigure(2, weight=1)
        
        # Header Header
        header = ctk.CTkLabel(self.main_view, text="Carousell Image Cropper", font=ctk.CTkFont(size=28, weight="bold"))
        header.grid(row=0, column=0, pady=(10, 20), sticky="w")
        
        # Paths Frame
        paths_frame = ctk.CTkFrame(self.main_view)
        paths_frame.grid(row=1, column=0, sticky="ew", pady=10, padx=5)
        paths_frame.grid_columnconfigure(1, weight=1)
        
        # Input Path
        ctk.CTkLabel(paths_frame, text="Input Folder:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.input_entry = ctk.CTkEntry(paths_frame, placeholder_text="Default: ./input")
        self.input_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=10)
        ctk.CTkButton(paths_frame, text="Browse", width=80, command=self._browse_input).grid(row=0, column=2, padx=10, pady=10)
        
        # Output Path
        ctk.CTkLabel(paths_frame, text="Output Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.output_entry = ctk.CTkEntry(paths_frame, placeholder_text="Default: ./output")
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=10)
        ctk.CTkButton(paths_frame, text="Browse", width=80, command=self._browse_output).grid(row=1, column=2, padx=10, pady=10)
        
        # Log Output
        self.log_box = ctk.CTkTextbox(self.main_view, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10, padx=5)
        self.log_box.configure(state="disabled")
        
        # Controls Frame (Bottom)
        controls = ctk.CTkFrame(self.main_view, fg_color="transparent")
        controls.grid(row=3, column=0, sticky="ew", pady=10)
        controls.grid_columnconfigure(0, weight=1)
        
        # Progress Bar
        self.progress = ctk.CTkProgressBar(controls, height=12)
        self.progress.grid(row=0, column=0, padx=5, pady=(10, 0), sticky="ew")
        self.progress.set(0)
        
        self.progress_lbl = ctk.CTkLabel(controls, text="0%", font=ctk.CTkFont(size=11))
        self.progress_lbl.grid(row=1, column=0, padx=5, pady=(2, 5), sticky="w")
        
        # Big Run Button
        self.run_btn = ctk.CTkButton(controls, text="PROCESS BATCH", font=ctk.CTkFont(size=14, weight="bold"), height=40, command=self.start_processing)
        self.run_btn.grid(row=0, column=1, padx=10, pady=10, sticky="e", rowspan=2)

    def _browse_input(self):
        path = filedialog.askdirectory()
        if path:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, path)
            
    def _browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def start_processing(self):
        if self.processing:
            self.stop_requested = True
            self.run_btn.configure(text="STOPPING...", state="disabled")
            self.log("\n[SYSTEM] Stop requested. Waiting for current image to finish...")
            return
            
        self.processing = True
        self.stop_requested = False
        self.run_btn.configure(state="normal", text="STOP PROCESSING", fg_color="#d9534f", hover_color="#c9302c")
        self.progress.set(0)
        self.progress_lbl.configure(text="0%")
        self._toggle_settings_state("disabled")
        
        # Build Config from sliders
        config = {
            'brightness': self.bright_slider.get(),
            'padding': self.pad_slider.get(),
            'adaptive_brightness': self.adaptive_switch.get() == 1,
            'anti_gray': self.antigray_switch.get() == 1,
            'scan_barcodes': self.barcode_switch.get() == 1
        }
        
        inp = self.input_entry.get().strip() or str(Path(get_root_dir()) / "input")
        out = self.output_entry.get().strip() or str(Path(get_root_dir()) / "output")

        # Start loading model if not initialized
        threading.Thread(target=self._run_batch, args=(inp, out, config), daemon=True).start()

    def _run_batch(self, inp_dir, out_dir, config):
        try:
            self.log("Initializing Workspace...")
            # Init model if loading for the first time
            if processor._yolo_model is None:
                weights = os.path.join(get_root_dir(), "runs", "detect", "carocrop_custom_fast4", "weights", "best.pt")
                if not os.path.exists(weights):
                     self.log(f"[ERROR] Weights not found at {weights}")
                     self.after(0, self._done_processing)
                     return
                self.log("Loading AI Model...")
                processor.init_yolo_model(weights)

            inp_path = Path(inp_dir)
            out_path = Path(out_dir)
            
            if not inp_path.exists():
                self.log(f"[ERROR] Input directory not found.")
                self.after(0, self._done_processing)
                return
                
            supported = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}
            if inp_path.is_file():
                files = [inp_path]
            else:
                files = [f for f in inp_path.rglob('*') if f.is_file() and f.suffix.lower() in supported]
                
                # Double sort key with Alphabetical Tiebreaker for burst shots
                def get_sort_key(f):
                    ts = get_exif_timestamp(f)
                    if ts is not None:
                        return (ts, f.name.lower())
                    return (f.stat().st_mtime, f.name.lower())
                    
                files.sort(key=get_sort_key)
                
            if not files:
                self.log("No images found in input folder.")
                self.after(0, self._done_processing)
                return
                
            self.log(f"Found {len(files)} images to process.")
            processing_state = {"current_imei": None, "imei_index": 1}
            
            batch_start = time.time()
            processed_count = 0
            for idx, img in enumerate(files, 1):
                if self.stop_requested:
                    self.log("\n[ABORT] Processing stopped by user.")
                    break
                    
                self.log(f"-> Processing image {idx}/{len(files)}")
                
                rel_path = img.relative_to(inp_path)
                target_out = out_path / rel_path.parent
                target_out.mkdir(parents=True, exist_ok=True)
                
                def _update_progress(p_val):
                    self.progress.set(p_val)
                    new_text = f"{int(p_val*100)}%"
                    if self.progress_lbl.cget("text") != new_text:
                        self.progress_lbl.configure(text=new_text)
                
                res = processor.process_image(
                    img, 
                    target_out, 
                    config=config, 
                    log_callback=lambda m: self.after(0, lambda: self.log(m)),
                    progress_callback=lambda p: self.after(0, lambda: _update_progress((idx - 1 + p) / len(files))),
                    state=processing_state
                )
                self.after(0, lambda: _update_progress(idx / len(files)))
                processed_count += 1

            batch_elapsed = time.time() - batch_start
            self.log(f"\n{'='*50}")
            self.log(f"  BATCH COMPLETE: {processed_count} photos in {batch_elapsed:.1f}s")
            self.log(f"{'='*50}")

            if self.stop_requested:
                self.after(0, lambda: messagebox.showwarning("Stopped", f"Processing stopped.\n{processed_count} photos processed in {batch_elapsed:.1f}s"))
            else:
                self.after(0, lambda: messagebox.showinfo("Done", f"Batch Complete!\n{processed_count} photos processed in {batch_elapsed:.1f}s"))
                
        except Exception as e:
            self.log(f"[EXCEPTION] {e}")
            
        self.after(0, self._done_processing)

    def _done_processing(self):
        self.processing = False
        self.stop_requested = False
        self.run_btn.configure(state="normal", text="PROCESS BATCH", fg_color="#10b981", hover_color="#059669")
        self._toggle_settings_state("normal")

    def _toggle_settings_state(self, state_str="normal"):
        self.bright_slider.configure(state=state_str)
        self.pad_slider.configure(state=state_str)
        self.adaptive_switch.configure(state=state_str)
        self.antigray_switch.configure(state=state_str)
        self.barcode_switch.configure(state=state_str)

if __name__ == "__main__":
    app = CaroCropApp()
    app.mainloop()
