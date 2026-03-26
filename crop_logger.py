"""
Crop Decision Logger for ML Training Data Collection
=====================================================
Instruments the cropping pipeline to save:
  - Original image
  - AI mask (U2-Net alpha)
  - Proposed & final crop bounds
  - All pipeline decision metadata (scores, thresholds, branches)
  - Human label (added later via interactive labeling tool)

Usage:
  # Labeling collected samples:
  python crop_logger.py label crop_logs/
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from PIL import Image


class CropLogger:
    """Captures crop pipeline decisions for ML training data collection."""

    def __init__(self, log_dir: str = "crop_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_entry: Optional[Path] = None
        self._metadata: Dict[str, Any] = {}

    def start_entry(self, image_name: str) -> Path:
        """Create a new log entry directory for one image."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in image_name)
        entry_name = f"{timestamp}_{safe_name}"
        entry_dir = self.log_dir / entry_name
        entry_dir.mkdir(parents=True, exist_ok=True)
        self._current_entry = entry_dir
        self._metadata = {
            "image_name": image_name,
            "timestamp": datetime.now().isoformat(),
            "pipeline_version": "3.0.0",
        }
        return entry_dir

    def save_original(self, image: Image.Image):
        """Save the original input image."""
        if self._current_entry is None:
            return
        path = self._current_entry / "original.jpg"
        image.convert("RGB").save(path, "JPEG", quality=92)

    def save_ai_mask(self, alpha_array):
        """Save the U2-Net alpha mask as grayscale PNG."""
        if self._current_entry is None:
            return
        try:
            mask_img = Image.fromarray(alpha_array, mode="L")
            path = self._current_entry / "ai_mask.png"
            mask_img.save(path, "PNG")
        except Exception as e:
            print(f"     [CROP-LOG] Failed to save AI mask: {e}")

    def save_cropped_output(self, image: Image.Image):
        """Save the final cropped output for comparison."""
        if self._current_entry is None:
            return
        path = self._current_entry / "cropped_output.png"
        image.convert("RGB").save(path, "PNG")

    def log_ai_detection(
        self,
        components: list,
        seed_index: int,
        cluster_indices: list,
        ai_bounds_xywh: Optional[Tuple[int, int, int, int]],
        fallback_used: bool,
    ):
        """Log AI component detection decisions."""
        self._metadata["ai_detection"] = {
            "num_components": len(components),
            "components": [
                {
                    "x": c[0], "y": c[1], "w": c[2], "h": c[3],
                    "area": c[4], "dist": round(c[5], 1), "score": round(c[6], 1),
                }
                for c in components
            ],
            "seed_index": seed_index,
            "cluster_indices": cluster_indices,
            "ai_bounds_xywh": list(ai_bounds_xywh) if ai_bounds_xywh else None,
            "cv_fallback_used": fallback_used,
        }

    def log_component_filter(
        self,
        kept: list,
        dropped: list,
    ):
        """Log which AI contours were kept vs dropped and why."""
        self._metadata["component_filter"] = {
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "dropped_details": dropped,  # list of dicts with reason, bounds, scores
        }

    def log_platform_detection(
        self,
        platform_bounds: Optional[Tuple[int, int, int, int]],
        edge_trim_left: int,
        edge_trim_right: int,
        left_scores: list,
        right_scores: list,
    ):
        """Log white platform detection results."""
        self._metadata["platform"] = {
            "bounds": list(platform_bounds) if platform_bounds else None,
            "edge_trim_left_px": edge_trim_left,
            "edge_trim_right_px": edge_trim_right,
            "left_contamination_scores": [round(s, 4) for s in left_scores[:10]],
            "right_contamination_scores": [round(s, 4) for s in right_scores[:10]],
        }

    def log_crop_solve(
        self,
        required_bounds: Tuple[int, int, int, int],
        proposed_crop: Tuple[int, int, int, int],
        square_size: int,
        strategy_used: str,
        padding_adjustments: Optional[Dict[str, Any]] = None,
    ):
        """Log the crop solver decision."""
        self._metadata["crop_solve"] = {
            "required_bounds_ltrb": list(required_bounds),
            "proposed_crop_ltrb": list(proposed_crop),
            "square_size": square_size,
            "strategy": strategy_used,
            "padding_adjustments": padding_adjustments,
        }

    def log_border_cleanup(
        self,
        scores: Dict[str, float],
        trim_applied: int,
    ):
        """Log post-crop border cleanup results."""
        self._metadata["border_cleanup"] = {
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "trim_applied_px": trim_applied,
        }

    def log_image_dimensions(self, original_size: Tuple[int, int], cropped_size: Tuple[int, int]):
        """Log image dimensions."""
        self._metadata["dimensions"] = {
            "original_w": original_size[0],
            "original_h": original_size[1],
            "cropped_w": cropped_size[0],
            "cropped_h": cropped_size[1],
        }

    def save_artifacts(self, artifacts: list):
        """Save detected artifacts directly to artifacts.json to pre-populate labeler."""
        if self._current_entry is None or not artifacts:
            return
        art_path = self._current_entry / "artifacts.json"
        try:
            with open(art_path, "w", encoding="utf-8") as f:
                json.dump({"artifacts": artifacts}, f, indent=2)
        except Exception as e:
            print(f"     [CROP-LOG] Failed to save artifacts: {e}")

    def finalize(self):
        """Write metadata JSON and close the current entry."""
        if self._current_entry is None:
            return
        meta_path = self._current_entry / "crop_meta.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2, ensure_ascii=False)
            print(f"     [CROP-LOG] Saved to {self._current_entry.name}/")
        except Exception as e:
            print(f"     [CROP-LOG] Failed to save metadata: {e}")
        self._current_entry = None
        self._metadata = {}


# ---------------------------------------------------------------------------
# Interactive Labeling Tool
# ---------------------------------------------------------------------------

def label_dataset(log_dir: str):
    """Interactive CLI to label crop log entries as GOOD or BAD_*.

    Walks through each entry in the log directory, shows the cropped output,
    and asks for a label. Labels are saved in label.txt.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Error: {log_dir} does not exist.")
        return

    entries = sorted([d for d in log_path.iterdir() if d.is_dir()])
    unlabeled = [e for e in entries if not (e / "label.txt").exists()]

    print(f"\n{'='*50}")
    print(f"  CROP LABELING TOOL")
    print(f"  Total entries: {len(entries)}")
    print(f"  Unlabeled: {len(unlabeled)}")
    print(f"{'='*50}")

    if not unlabeled:
        print("\n  All entries are labeled!")
        return

    print("\n  Labels:")
    print("    G = GOOD (crop is correct)")
    print("    C = BAD_CLIP (product is clipped/cut off)")
    print("    A = BAD_ARTIFACT (unwanted object in crop)")
    print("    O = BAD_OFFSET (product off-center, bad framing)")
    print("    S = Skip (come back later)")
    print("    Q = Quit\n")

    label_map = {
        "g": "GOOD",
        "c": "BAD_CLIP",
        "a": "BAD_ARTIFACT",
        "o": "BAD_OFFSET",
    }

    labeled_count = 0
    for idx, entry in enumerate(unlabeled, 1):
        meta_path = entry / "crop_meta.json"
        cropped_path = entry / "cropped_output.png"

        name = "unknown"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                name = meta.get("image_name", "unknown")
            except Exception:
                pass

        print(f"  [{idx}/{len(unlabeled)}] {name}")
        print(f"    Dir: {entry.name}")

        # Try to show the image (Windows: opens default viewer)
        if cropped_path.exists():
            try:
                img = Image.open(cropped_path)
                img.show()
            except Exception:
                print(f"    (Could not display image, check: {cropped_path})")
        else:
            print(f"    (No cropped output found)")

        while True:
            try:
                choice = input("    Label [G/C/A/O/S/Q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "q"

            if choice == "q":
                print(f"\n  Labeled {labeled_count} entries. Exiting.")
                return
            if choice == "s":
                print("    Skipped.")
                break
            if choice in label_map:
                label = label_map[choice]
                with open(entry / "label.txt", "w") as f:
                    f.write(label + "\n")
                print(f"    -> {label}")
                labeled_count += 1
                break
            print("    Invalid choice. Use G, C, A, O, S, or Q.")

    print(f"\n  Done! Labeled {labeled_count} entries total.")


def print_dataset_stats(log_dir: str):
    """Print summary statistics of the labeled dataset."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Error: {log_dir} does not exist.")
        return

    entries = sorted([d for d in log_path.iterdir() if d.is_dir()])
    labels = {}
    unlabeled = 0

    for entry in entries:
        label_file = entry / "label.txt"
        if label_file.exists():
            label = label_file.read_text().strip()
            labels[label] = labels.get(label, 0) + 1
        else:
            unlabeled += 1

    print(f"\n{'='*40}")
    print(f"  DATASET STATISTICS: {log_dir}")
    print(f"{'='*40}")
    print(f"  Total entries:  {len(entries)}")
    print(f"  Unlabeled:      {unlabeled}")
    for label, count in sorted(labels.items()):
        print(f"  {label:15s}: {count}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop Logger Tools")
    sub = parser.add_subparsers(dest="command")

    label_parser = sub.add_parser("label", help="Interactively label crop log entries")
    label_parser.add_argument("log_dir", type=str, help="Path to crop_logs directory")

    stats_parser = sub.add_parser("stats", help="Print dataset statistics")
    stats_parser.add_argument("log_dir", type=str, help="Path to crop_logs directory")

    args = parser.parse_args()

    if args.command == "label":
        label_dataset(args.log_dir)
    elif args.command == "stats":
        print_dataset_stats(args.log_dir)
    else:
        parser.print_help()
