# MisterMobile Carousell Image Cropper — GUI Edition

Desktop application that processes product photos for Carousell marketplace listings.

## Features

- **AI Object Detection** — YOLO-based product detection with automatic artifact removal
- **Smart Cropping** — 1080×1080px square output with intelligent center-crop
- **IMEI Barcode Scanner** — Scans photos for barcodes, renames files by IMEI (last 6 digits)
- **Image Enhancement** — Adaptive brightness, contrast, sharpness, and anti-gray correction
- **Watermark** — Auto-applies MM watermark to outputs
- **Batch Processing** — Drag-and-drop folder processing with progress bar

## IMEI Naming Convention

When "Scan Barcodes (IMEI)" is enabled:

| Photo Type | Filename | Example |
|---|---|---|
| Barcode sticker | `<last6> 00.png` | `271404 00.png` |
| Device photo 1 | `<last6> 1a.png` | `271404 1a.png` |
| Device photo 2 | `<last6> 1b.png` | `271404 1b.png` |
| ... | continues a-z, then 2a, 2b... | `271404 1z.png` → `271404 2a.png` |

The barcode photo (`00`) always sorts first. Sequence resets when a new barcode is scanned.

## Setup (Development)

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements_gui.txt
```

## Run (Development)

```bash
.\venv\Scripts\python.exe gui.py
```

Or double-click `run_gui.bat`.

## Build Standalone EXE

```bash
build_gui_pyinstaller.bat
```

This produces `dist\MisterMobileCropper-GUI\` containing:
- `MisterMobileCropper-GUI.exe`
- `MM Watermark.png`
- `runs\` — AI model weights
- `input\` — place images here
- `output\` — results appear here

## Project Structure

```
gui.py                  — GUI application (CustomTkinter)
processor_gui.py        — Image processing pipeline + barcode scanner
crop_logger.py          — Debug logging for crop operations
MM Watermark.png        — Watermark overlay asset
requirements_gui.txt    — Python dependencies
build_gui_pyinstaller.bat — PyInstaller build script
run_gui.bat             — Quick-launch script
runs/                   — YOLO model weights
```

## Dependencies

- Python 3.10+
- CustomTkinter, Ultralytics (YOLO), Pillow, OpenCV, rembg, zxing-cpp
- See `requirements_gui.txt` for full list
