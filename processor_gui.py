"""
Carousell Image Cropping Pipeline
==================================
Crops images to Singapore Carousell marketplace standard:
- 1:1 square aspect ratio
- 1080x1080px output size
- Center-crop (no background removal)
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import pillow_heif
import numpy as np
import cv2
from crop_logger import CropLogger

# Register HEIF/HEIC opener with PIL
pillow_heif.register_heif_opener()

# Carousell photo specifications
TARGET_SIZE = 1080  # 1080x1080px square
PLATFORM_X_TRIM_PCT = 0.05
PLATFORM_Y_TRIM_PCT = 0.03
DEFAULT_PADDING_PCT = 0.06
YELLOW_MIN_AREA = 350
EDGE_CLEAN_SCAN_PCT = 0.20
EDGE_CLEAN_WINDOW = 24
EDGE_NONWHITE_MAX_RATIO = 0.08
EDGE_CLEAN_STABILITY_WINDOWS = 3
EDGE_CLEAN_PRODUCT_GUARD_PCT = 0.02
ABSOLUTE_CLEAN_MARGIN_PCT = 0.08
AI_MASK_DILATE_PX = 3  # Dilation kernel radius applied to AI mask before contour extraction
PADDING_ARTIFACT_THRESHOLD = 0.50  # Max contamination score before padding is reduced on a side
BACKDROP_TARGET_BRIGHTNESS = 240
BACKDROP_FEATHER_PX = 32
BACKDROP_FEATHER_SIGMA = 5.0
BACKDROP_MAX_LIFT = 45.0
BACKDROP_EDGE_MIN_STRENGTH = 0.72
BACKDROP_SHADOW_NEAR_PX = 180.0
BACKDROP_SHADOW_EXTRA_LIFT = 26.0
BACKDROP_FAR_REF_DIST_PX = 120.0
ENABLE_BACKDROP_NORMALIZATION = False
GLOBAL_BRIGHTNESS = 1.16
ENABLE_ADAPTIVE_BRIGHTNESS_TARGET = True
ADAPTIVE_BRIGHTNESS_TARGET_LUMA = 206.0
ADAPTIVE_BRIGHTNESS_STRENGTH = 0.65
ADAPTIVE_BRIGHTNESS_MIN = 1.08
ADAPTIVE_BRIGHTNESS_MAX = 1.50
ENABLE_ANTI_GRAY_CORRECTION = True
WHITE_POINT_PERCENTILE = 99.0
WHITE_POINT_TARGET_LUMA = 248.0
WHITE_POINT_MAX_GAIN = 1.06
GLOBAL_COLOR_BOOST = 1.07

# Global models
_rembg_session = None
_yolo_model = None

def init_yolo_model(weights_path: str):
    """Load the custom YOLO detection model."""
    global _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is required for YOLO. Run `pip install ultralytics`.")
        sys.exit(1)
        
    print(f"Loading custom YOLO model from {weights_path}...")
    _yolo_model = YOLO(weights_path)
    print("YOLO model loaded successfully.")


def _compute_iou(box_a, box_b):
    """Compute Intersection over Union between two (left, top, right, bottom) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    
    if union <= 0:
        return 0.0
    return inter / union


def get_yolo_detections(image: Image.Image) -> Tuple[Optional[Tuple[int, int, int, int]], list]:
    """Returns ((product_xywh_ltrb), [artifact_ltrb, ...])
    
    Artifacts that overlap with the product bounding box (IoU > 10%) are
    filtered out as false positives (e.g. the yellow 3D logo near the phone).
    """
    if _yolo_model is None:
        return None, []
        
    img = image.convert("RGB")
    try:
        # Low confidence (0.05) to catch subtle edge artifacts like light tent curves
        results = _yolo_model.predict(source=img, conf=0.05, verbose=False)
    except Exception as e:
        print(f"     [YOLO] Prediction failed: {e}")
        return None, []
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        print("     [YOLO] No objects detected above confidence threshold.")
        return None, []
        
    boxes = results[0].boxes
    product_xywh = None
    product_ltrb = None
    raw_artifacts = []
    
    sorted_indices = boxes.conf.argsort(descending=True)
    for idx in sorted_indices:
        box = boxes[idx]
        cls_id = int(box.cls.item())
        xyxy = box.xyxy.squeeze().tolist()
        
        # Handle 1D tensors correctly (if only 1 box is detected, xyxy is 1D array of 4 floats)
        if isinstance(xyxy, list) and len(xyxy) > 0 and isinstance(xyxy[0], list):
            xyxy = xyxy[0]
            
        left, top, right, bottom = [int(x) for x in xyxy]
        
        if cls_id == 0 and product_xywh is None:
            width = right - left
            height = bottom - top
            product_xywh = (left, top, width, height)
            product_ltrb = (left, top, right, bottom)
            print(f"     [YOLO] Detected product bounds {width}x{height} at ({left},{top}) [conf: {box.conf.item():.2f}]")
        elif cls_id == 1 and box.conf.item() >= 0.05:
            raw_artifacts.append((left, top, right, bottom, box.conf.item()))
            
    # Filter artifacts: if an artifact overlaps with the product box, we MUST
    # shrink the product box to exclude it. The user wants the final crop 
    # to tightly exclude the red artifact boxes.
    for (al, at, ar, ab, conf) in raw_artifacts:
        if product_ltrb is not None:
            pl, pt, pr, pb = product_ltrb
            
            # Mathematical intersection
            ix1 = max(pl, al)
            iy1 = max(pt, at)
            ix2 = min(pr, ar)
            iy2 = min(pb, ab)
            
            # If the artifact intersects the product at all...
            if ix1 < ix2 and iy1 < iy2:
                # Figure out which edge of the product box to pull inwards
                # We pull the side that overlaps the LEAST to minimize product loss.
                cut_right = pr - al if (al > pl and al < pr) else float('inf')
                cut_left = ar - pl if (ar > pl and ar < pr) else float('inf')
                cut_bottom = pb - at if (at > pt and at < pb) else float('inf')
                cut_top = ab - pt if (ab > pt and ab < pb) else float('inf')
                
                min_cut = min(cut_right, cut_left, cut_bottom, cut_top)
                
                if min_cut != float('inf'):
                    if min_cut == cut_right: pr = al
                    elif min_cut == cut_left: pl = ar
                    elif min_cut == cut_bottom: pb = at
                    elif min_cut == cut_top: pt = ab
                        
                    print(f"     [YOLO] Artifact at ({al},{at}) intersects product. Shrinking product box by {min_cut}px.")
                    product_ltrb = (pl, pt, pr, pb)
                    product_xywh = (pl, pt, pr - pl, pb - pt)
                else:
                    print(f"     [YOLO] Artifact at ({al},{at}) engulfs product center. Cannot shrink cleanly.")

    return product_xywh, raw_artifacts


def _get_rembg_session():
    """Return a cached rembg session (loads U2-Net model on first call).

    Checks for a bundled u2net.onnx next to the executable first,
    so compiled distributions work offline without downloading.
    """
    global _rembg_session
    if _rembg_session is None:
        import os

        # Point rembg model cache to a local folder so a bundled
        # u2net.onnx is found automatically (no internet needed).
        # Nuitka onefile: sys.executable is temp dir, sys.argv[0] is real exe.
        script_dir = str(Path(__file__).parent)
        if "__compiled__" in globals() or getattr(sys, 'frozen', False):
            exe_dir = str(Path(os.path.abspath(sys.argv[0])).parent)
        else:
            exe_dir = script_dir

        # Stabilize numba cache path to avoid temp-file stalls in onefile runs.
        numba_cache_dir = os.path.join(exe_dir, '.numba_cache')
        try:
            os.makedirs(numba_cache_dir, exist_ok=True)
            os.environ.setdefault('NUMBA_CACHE_DIR', numba_cache_dir)
        except Exception:
            pass
        # Reduce native runtime pressure in compiled onefile mode.
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
        os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
        os.environ.setdefault('NUMBA_DISABLE_CUDA', '1')

        from rembg import new_session
        for candidate in [exe_dir, script_dir]:
            bundled_model = os.path.join(candidate, '.u2net', 'u2net.onnx')
            if os.path.isfile(bundled_model):
                os.environ['U2NET_HOME'] = os.path.join(candidate, '.u2net')
                break

        print("")
        print("  ========================================")
        print("  Loading AI model... (first image only)")
        print("  This may take a moment.")
        print("  ========================================")
        _rembg_session = new_session("u2net")
        print("  [AI] Model loaded and ready!")
        print("")
    return _rembg_session


def load_image(image_path: Path) -> Image.Image:
    """Load an image from various formats including HEIC."""
    img = Image.open(image_path)
    # Handle EXIF orientation
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    return img


def sample_backdrop_color(image: Image.Image) -> tuple:
    """
    Sample the dominant backdrop color from the image edges.
    
    Looks at the outer 15% margin of the image and finds the most common
    light color (brightness > 150) to use as the canvas background.
    
    Returns:
        tuple: RGB color (r, g, b) of the dominant backdrop color
    """
    import numpy as np
    from collections import Counter
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Define edge zones (outer 15%)
    margin_h = int(height * 0.15)
    margin_w = int(width * 0.15)
    
    # Collect pixels from all edge zones
    edge_pixels = []
    
    # Top strip
    edge_pixels.extend(img_array[:margin_h, :].reshape(-1, 3).tolist())
    # Bottom strip
    edge_pixels.extend(img_array[-margin_h:, :].reshape(-1, 3).tolist())
    # Left strip (excluding corners already counted)
    edge_pixels.extend(img_array[margin_h:-margin_h, :margin_w].reshape(-1, 3).tolist())
    # Right strip (excluding corners already counted)
    edge_pixels.extend(img_array[margin_h:-margin_h, -margin_w:].reshape(-1, 3).tolist())
    
    # Filter to only light pixels (potential backdrop, not dark product edges)
    edge_pixels = np.array(edge_pixels)
    brightness = np.mean(edge_pixels, axis=1)
    light_pixels = edge_pixels[brightness > 150]
    
    if len(light_pixels) == 0:
        # Fallback to white if no light pixels found
        print("     No backdrop detected, using white")
        return (255, 255, 255)
    
    # Find median color of light pixels (more robust than mean)
    median_color = np.median(light_pixels, axis=0).astype(int)
    
    print(f"     Sampled backdrop color: RGB({median_color[0]}, {median_color[1]}, {median_color[2]})")
    
    return tuple(median_color)


def brighten_backdrop(
    image: Image.Image,
    target_brightness: int = BACKDROP_TARGET_BRIGHTNESS,
) -> Image.Image:
    """
    Normalize backdrop to a consistent target brightness using the AI (rembg) mask.

    Instead of multiplying by a variable factor, this maps backdrop pixels toward
    a fixed target value so every output has the same backdrop brightness regardless
    of input lighting.

    Args:
        image: Input image (already cropped)
        target_brightness: Desired average backdrop brightness (0-255)

    Returns:
        Image with normalized backdrop and original product
    """
    import numpy as np
    import cv2
    from rembg import remove

    img_array = np.array(image)
    height, width = img_array.shape[:2]

    # Use AI mask (rembg) for accurate product/backdrop separation
    session = _get_rembg_session()
    result_rgba = remove(image, alpha_matting=False, session=session)
    alpha = np.array(result_rgba)[:, :, 3]

    # Product mask: alpha > 128 is definitely product (conservative threshold)
    _, product_core_mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

    # Also include semi-transparent zones (alpha 30-128) as product safety margin
    _, semi_mask = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    product_mask = cv2.bitwise_or(product_core_mask, semi_mask)

    # Dilate product mask generously to protect edges from brightening bleed
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    product_mask = cv2.dilate(product_mask, dilate_kernel, iterations=1)

    backdrop_mask = cv2.bitwise_not(product_mask)
    backdrop_binary = (backdrop_mask > 0).astype(np.uint8)

    # Feathered correction mask from product edge into backdrop with Gaussian softening.
    dist_to_product = cv2.distanceTransform(backdrop_binary, cv2.DIST_L2, 5)
    feather_base = np.clip(dist_to_product / float(BACKDROP_FEATHER_PX), 0.0, 1.0).astype(np.float32)
    feather_base = cv2.GaussianBlur(feather_base, (0, 0), BACKDROP_FEATHER_SIGMA)
    feather_base = np.clip(feather_base, 0.0, 1.0)
    correction_mask = (
        BACKDROP_EDGE_MIN_STRENGTH + (1.0 - BACKDROP_EDGE_MIN_STRENGTH) * feather_base
    ) * backdrop_binary.astype(np.float32)

    # A wider near-product weight helps flatten large shadow rings around objects.
    near_shadow_weight = np.exp(-np.square(dist_to_product / BACKDROP_SHADOW_NEAR_PX)).astype(np.float32)
    near_shadow_weight = near_shadow_weight * backdrop_binary.astype(np.float32)

    # Measure current backdrop brightness (grayscale proxy)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    backdrop_pixels = gray[backdrop_mask > 0]
    product_pixels = gray[product_core_mask > 0]
    if len(backdrop_pixels) > 0:
        avg_backdrop = float(np.mean(backdrop_pixels))

        if avg_backdrop < target_brightness - 2:
            # Lift Lab luminance only where backdrop exists, tapered by feather mask.
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
            l_channel = lab[:, :, 0]

            # Base lift toward target brightness on full backdrop.
            base_lift = np.clip(target_brightness - l_channel, 0.0, BACKDROP_MAX_LIFT) * correction_mask

            # Shadow-ring suppression: use far backdrop as reference and lift nearby darker zones.
            far_region = (backdrop_mask > 0) & (dist_to_product >= BACKDROP_FAR_REF_DIST_PX)
            if np.count_nonzero(far_region) > 500:
                far_ref = float(np.percentile(gray[far_region], 65))
            else:
                far_ref = max(avg_backdrop, float(target_brightness - 4))

            shadow_deficit = np.clip(far_ref - gray, 0.0, BACKDROP_SHADOW_EXTRA_LIFT)
            shadow_lift = shadow_deficit * near_shadow_weight

            total_lift = np.maximum(base_lift, shadow_lift)
            l_channel = l_channel + total_lift
            lab[:, :, 0] = np.clip(l_channel, 0.0, 255.0)
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

            result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
            new_avg = float(np.mean(result_gray[backdrop_mask > 0]))
            if len(product_pixels) > 0:
                product_delta = float(np.mean(result_gray[product_core_mask > 0]) - np.mean(product_pixels))
            else:
                product_delta = 0.0
            print(
                f"     Backdrop normalized: avg={avg_backdrop:.1f} -> {new_avg:.1f} "
                f"(target={target_brightness}, product_delta={product_delta:+.2f})"
            )
        else:
            result = img_array
            print(f"     Backdrop already bright: avg={avg_backdrop:.1f}, target={target_brightness}")
    else:
        result = img_array
        print(f"     No backdrop detected, skipping brightening")

    return Image.fromarray(result)


def enhance_image(image: Image.Image, adaptive: bool = True) -> Image.Image:
    """
    Enhance image with adaptive or fixed brightness, contrast, and sharpening.
    
    Args:
        image: Input image
        adaptive: If True, analyze image and adjust parameters dynamically
    
    Returns:
        Enhanced image with applied adjustments
    """
    import numpy as np
    import cv2

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # FIXED brightness — same for every image (consistency)
    # Keep global brightness neutral; backdrop normalization handles lifting.
    avg_luma = float(np.mean(gray))
    brightness = GLOBAL_BRIGHTNESS

    if adaptive:
        if ENABLE_ADAPTIVE_BRIGHTNESS_TARGET:
            # Nudge each image toward a target luminance, with tight clamp bounds.
            raw_ratio = ADAPTIVE_BRIGHTNESS_TARGET_LUMA / max(avg_luma, 1.0)
            adaptive_factor = 1.0 + ((raw_ratio - 1.0) * ADAPTIVE_BRIGHTNESS_STRENGTH)
            brightness = float(
                np.clip(
                    GLOBAL_BRIGHTNESS * adaptive_factor,
                    ADAPTIVE_BRIGHTNESS_MIN,
                    ADAPTIVE_BRIGHTNESS_MAX,
                )
            )

        # ADAPTIVE contrast
        contrast_level = np.std(gray)
        if contrast_level < 35:
            contrast = 1.25
        elif contrast_level > 65:
            contrast = 1.05
        else:
            contrast = 1.15

        # ADAPTIVE sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        if sharpness_score < 100:
            sharpness = 1.5
        elif sharpness_score > 500:
            sharpness = 1.1
        else:
            sharpness = 1.3

        print(
            f"     Analysis: luma={avg_luma:.1f}, contrast={contrast_level:.1f}, "
            f"sharpness={sharpness_score:.1f}"
        )
    else:
        contrast = 1.05
        sharpness = 1.3

    enhancer = ImageEnhance.Brightness(image)
    enhanced = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(contrast)

    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(sharpness)

    white_gain = 1.0
    if ENABLE_ANTI_GRAY_CORRECTION:
        # Mild white-point lift to reduce gray cast without flattening the image.
        enhanced_np = np.array(enhanced)
        lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        white_ref = float(np.percentile(l_channel, WHITE_POINT_PERCENTILE))
        if white_ref > 1.0:
            white_gain = min(
                WHITE_POINT_MAX_GAIN,
                max(1.0, WHITE_POINT_TARGET_LUMA / white_ref),
            )
            if white_gain > 1.001:
                lab[:, :, 0] = np.clip(l_channel * white_gain, 0.0, 255.0)
                enhanced = Image.fromarray(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))

        # Tiny saturation bump helps counter dull/gray output after lift.
        color_enhancer = ImageEnhance.Color(enhanced)
        enhanced = color_enhancer.enhance(GLOBAL_COLOR_BOOST)

    print(
        f"     Enhanced: brightness={brightness:.2f}, contrast={contrast:.2f}, "
        f"sharpness={sharpness:.2f}, white_gain={white_gain:.3f}, color={GLOBAL_COLOR_BOOST:.2f}"
    )

    return enhanced

def extend_edges(image: Image.Image, extend_percent: float = 0.15) -> Image.Image:
    """
    Extend the image edges outward by tiling the edge pixels.
    
    This creates extra margin around the image by repeating the edge pixels,
    giving more backdrop area for the smart crop to work with.
    
    Args:
        image: Input image
        extend_percent: How much to extend each edge (as percentage of image size)
    
    Returns:
        Extended image with tiled edges
    """
    import numpy as np
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Calculate extension size
    extend_h = int(height * extend_percent)
    extend_w = int(width * extend_percent)
    
    # Create new larger canvas
    new_height = height + 2 * extend_h
    new_width = width + 2 * extend_w
    
    # Start with tiled edges
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Place original image in center
    result[extend_h:extend_h+height, extend_w:extend_w+width] = img_array
    
    # Extend TOP edge (tile the top row)
    top_row = img_array[0:1, :, :]  # First row
    for y in range(extend_h):
        result[y, extend_w:extend_w+width] = top_row
    
    # Extend BOTTOM edge (tile the bottom row)
    bottom_row = img_array[-1:, :, :]  # Last row
    for y in range(extend_h + height, new_height):
        result[y, extend_w:extend_w+width] = bottom_row
    
    # Extend LEFT edge (tile the left column)
    left_col = img_array[:, 0:1, :]  # First column
    for x in range(extend_w):
        result[extend_h:extend_h+height, x] = left_col[:, 0, :]
    
    # Extend RIGHT edge (tile the right column)
    right_col = img_array[:, -1:, :]  # Last column
    for x in range(extend_w + width, new_width):
        result[extend_h:extend_h+height, x] = right_col[:, 0, :]
    
    # Fill corners with corner pixels
    # Top-left corner
    result[:extend_h, :extend_w] = img_array[0, 0]
    # Top-right corner
    result[:extend_h, extend_w+width:] = img_array[0, -1]
    # Bottom-left corner
    result[extend_h+height:, :extend_w] = img_array[-1, 0]
    # Bottom-right corner
    result[extend_h+height:, extend_w+width:] = img_array[-1, -1]
    
    print(f"     Extended edges by {extend_percent*100:.0f}%: {width}x{height} -> {new_width}x{new_height}")
    
    return Image.fromarray(result)


def _clip_ltrb(bounds: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    left = max(0, min(width - 1, int(left)))
    top = max(0, min(height - 1, int(top)))
    right = max(left + 1, min(width, int(right)))
    bottom = max(top + 1, min(height, int(bottom)))
    return left, top, right, bottom


def _union_ltrb(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _xywh_to_ltrb(bounds_xywh: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = bounds_xywh
    return x, y, x + w, y + h


def score_side_contamination(
    img,
    side,
    x1,
    x2,
    row_mask,
    bg_ref,
) -> float:
    """Score how likely a side window is edge clutter instead of clean backdrop."""
    import numpy as np
    import cv2

    height = img.shape[0]
    x1 = max(0, min(img.shape[1] - 1, int(x1)))
    x2 = max(x1 + 1, min(img.shape[1], int(x2)))
    window = img[:, x1:x2, :]
    if window.size == 0:
        return 0.0

    if row_mask is not None and len(row_mask) == height:
        active = bool(np.any(row_mask))
        if active:
            window = window[row_mask, :, :]
    if window.size == 0:
        return 0.0

    gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(window, cv2.COLOR_RGB2LAB)

    luma = float(gray.mean())
    sat = float((window.max(axis=2) - window.min(axis=2)).mean())
    a_mean = float(lab[:, :, 1].mean())
    b_mean = float(lab[:, :, 2].mean())

    luma_delta = abs(luma - bg_ref["luma"]) / 255.0
    sat_delta = max(0.0, sat - bg_ref["sat"]) / 255.0
    chroma_delta = min(
        1.0,
        (((a_mean - bg_ref["a"]) ** 2 + (b_mean - bg_ref["b"]) ** 2) ** 0.5) / 181.0,
    )

    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(edges.mean() / 255.0)
    texture_var = min(1.0, float(gray.var()) / 500.0)

    score = (
        0.30 * luma_delta
        + 0.15 * sat_delta
        + 0.20 * chroma_delta
        + 0.20 * edge_density
        + 0.15 * texture_var
    )
    return float(max(0.0, min(1.0, score)))


def filter_ai_components(
    mask,
    image: Image.Image,
    platform_hint,
    required_center_hint,
):
    """Drop side-touching AI contours that look like backdrop artifacts.
    
    Uses center-biased scoring: score = area * (1 - dist/max_dist)^2
    with 0.3x penalty for edge-touching contours.
    """
    import numpy as np
    import cv2

    img = np.array(image.convert("RGB"))
    height, width = img.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    if required_center_hint is None:
        center_x, center_y = width // 2, height // 2
    else:
        center_x, center_y = required_center_hint

    if platform_hint is None:
        plat_left, plat_top, plat_right, plat_bottom = 0, 0, width, height
    else:
        plat_left, plat_top, plat_right, plat_bottom = _clip_ltrb(platform_hint, width, height)

    cy1 = max(plat_top, int(height * 0.15))
    cy2 = min(plat_bottom, int(height * 0.90))
    cx1 = max(plat_left, int(width * 0.35))
    cx2 = min(plat_right, int(width * 0.65))
    bg = img[cy1:cy2, cx1:cx2, :]
    if bg.size == 0:
        bg = img[int(height * 0.25):int(height * 0.85), int(width * 0.35):int(width * 0.65), :]
    if bg.size == 0:
        bg = img

    bg_gray = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
    bg_lab = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB)
    bg_ref = {
        "luma": float(bg_gray.mean()),
        "sat": float((bg.max(axis=2) - bg.min(axis=2)).mean()),
        "a": float(bg_lab[:, :, 1].mean()),
        "b": float(bg_lab[:, :, 2].mean()),
    }

    min_area = max(200.0, width * height * 0.0005)
    # Border margins: 5% extreme auto-drop, 8% side scoring zone
    extreme_edge_x = int(width * 0.05)
    extreme_edge_y = int(height * 0.05)
    side_margin = int(width * 0.08)
    top_margin = int(height * 0.05)
    bottom_margin = int(height * 0.03)
    max_dist = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5

    components = []
    kept = 0
    dropped = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + (w // 2), y + (h // 2)
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        fill_ratio = area / max(1.0, float(w * h))
        aspect = max(w / max(1.0, h), h / max(1.0, w))

        # Edge touching checks (left/right AND top/bottom)
        touch_left = x <= side_margin
        touch_right = x + w >= width - side_margin
        touch_top = y <= top_margin
        touch_bottom = y + h >= height - bottom_margin
        side_touch = touch_left or touch_right

        row_mask = np.zeros((height,), dtype=bool)
        y1 = max(0, y - int(h * 0.25))
        y2 = min(height, y + h + int(h * 0.25))
        if y2 > y1:
            row_mask[y1:y2] = True

        contam = 0.0
        if side_touch:
            contam = score_side_contamination(img, "left" if touch_left else "right", x, x + w, row_mask, bg_ref)

        # --- CENTER-BIASED COMPOSITE SCORE ---
        # score = area * (1 - dist/max_dist)^2, penalize edge-touching
        proximity = max(0.0, 1.0 - dist / max_dist)
        score = area * (proximity ** 2)
        
        # Penalize edge-touching contours
        any_edge_touch = touch_left or touch_right or touch_top or touch_bottom
        if any_edge_touch:
            score *= 0.3

        # --- DROP LOGIC ---
        # 1. Auto-drop: contours touching extreme outer 5% of any edge
        is_extreme_edge = (
            (x <= extreme_edge_x) or
            (x + w >= width - extreme_edge_x) or
            (y <= extreme_edge_y) or
            (y + h >= height - extreme_edge_y)
        )

        shape_bad = (aspect > 5.5 and fill_ratio < 0.25) or (fill_ratio < 0.12)
        smallish = area < (width * height * 0.03)
        far_from_center = dist > (min(width, height) * 0.32)
        very_small = area < (width * height * 0.008)
        peripheral = dist > (min(width, height) * 0.25)
        # New tier: small-to-medium objects that are far away (logos, stickers, props)
        small_medium = area < (width * height * 0.025)
        distant = dist > (min(width, height) * 0.38)
        
        should_drop = (
            is_extreme_edge or
            (side_touch and smallish and contam > 0.26 and (shape_bad or far_from_center)) or
            # Drop very small + peripheral (paper labels, case corners)
            (very_small and peripheral and (shape_bad or fill_ratio < 0.35)) or
            # Drop small-to-medium objects that are very far from center
            # (logos, stickers, props at table edges)
            (small_medium and distant)
        )

        if should_drop:
            dropped += 1
            print(
                f"     [AI-COMPONENT] dropped artifact at ({x},{y}) "
                f"{w}x{h}, score={score:.0f}, contam={contam:.3f}, fill={fill_ratio:.3f}"
            )
            continue

        kept += 1
        components.append((x, y, w, h, area, dist, score))

    print(f"     [AI-COMPONENT] kept={kept} dropped={dropped}")
    return components


def get_ai_crop_bounds(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """
    AI-first detection using rembg/U2-Net.
    Returns bounds as (x, y, w, h).
    """
    try:
        from rembg import remove
        import numpy as np
        import cv2

        session = _get_rembg_session()
        result = remove(image, alpha_matting=False, session=session)
        alpha = np.array(result)[:, :, 3]
        _, mask = cv2.threshold(alpha, 20, 255, cv2.THRESH_BINARY)

        # Dilate mask to fuse fragile edges before contour extraction.
        # Prevents thin device boundaries from fragmenting into small
        # contours that get dropped by filter_ai_components().
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (AI_MASK_DILATE_PX * 2 + 1, AI_MASK_DILATE_PX * 2 + 1),
        )
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print('     [AI-FIRST] No contours from AI alpha mask')
            return None

        img_h, img_w = alpha.shape
        center_x, center_y = img_w // 2, img_h // 2
        components = filter_ai_components(
            mask=mask,
            image=image,
            platform_hint=None,
            required_center_hint=(center_x, center_y),
        )

        # Pick seed by highest composite score (center-biased, area-weighted).
        seed = max(components, key=lambda c: c[6])
        seed_cx = seed[0] + seed[2] // 2
        seed_cy = seed[1] + seed[3] // 2
        max_cluster_dist = min(img_w, img_h) * 0.35
        seed_area = seed[4]
        total_img_area = img_w * img_h

        cluster = []
        for c in components:
            cx = c[0] + c[2] // 2
            cy = c[1] + c[3] // 2
            dist = ((cx - seed_cx) ** 2 + (cy - seed_cy) ** 2) ** 0.5
            c_area = c[4]
            
            # MULTI-PART COMPONENT FILTER
            # Keep if close to seed. For distant components, require them to be
            # large (>20% of seed) and reasonably close to be part of same product.
            # This prevents logos, stickers, and other peripheral objects from
            # pulling the crop bounds outward.
            if dist <= max_cluster_dist:
                cluster.append(c)
            elif c_area >= seed_area * 0.20 and dist <= max_cluster_dist * 1.4:
                # Large component relatively near — likely part of product (e.g. stylus)
                cluster.append(c)
                print(f"     [AI-FIRST] Including distant component area={c_area:.0f} dist={dist:.0f}")

        if not cluster:
            cluster = [seed]

        left = min(c[0] for c in cluster)
        top = min(c[1] for c in cluster)
        right = max(c[0] + c[2] for c in cluster)
        bottom = max(c[1] + c[3] for c in cluster)
        print(f'     [AI-FIRST] U2-Net bounds {right-left}x{bottom-top} at ({left},{top})')
        return left, top, right - left, bottom - top

    except Exception as e:
        print(f'     [AI-FIRST] AI detection failed: {e}')
        return None


def get_cv_crop_bounds(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Fallback CV detector using Lab-based foreground masking.
    
    Converts to Lab, thresholds on L channel using Otsu,
    applies morphology cleanup, then scores contours by
    center proximity and area.
    """
    import numpy as np
    import cv2

    img = np.array(image.convert('RGB'))
    height, width = img.shape[:2]

    # Lab-based foreground mask (more stable than Canny on light backgrounds)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    blurred_l = cv2.GaussianBlur(l_channel, (5, 5), 0)
    
    # Otsu threshold on L channel (separates dark product from bright backdrop)
    _, mask = cv2.threshold(blurred_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology: close (fill gaps in product), then open (remove noise specks)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate mask for consistency with AI path — fuse fragile edges
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (AI_MASK_DILATE_PX * 2 + 1, AI_MASK_DILATE_PX * 2 + 1),
    )
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print('     [AI-FIRST] CV fallback found no contours')
        return None

    center_x = width // 2
    center_y = height // 2
    # Wider edge margins matching the light-box strategy
    edge_margin_x = int(width * 0.08)
    edge_margin_y = int(height * 0.05)
    min_area = (width * height) * 0.001
    max_dist = ((width / 2) ** 2 + (height / 2) ** 2) ** 0.5

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < min_area or w < 30 or h < 30:
            continue

        # Auto-drop contours touching outer edge margins
        touching_edge = (
            x <= edge_margin_x
            or y <= edge_margin_y
            or x + w >= width - edge_margin_x
            or y + h >= height - edge_margin_y
        )
        if touching_edge:
            continue

        cx = x + w // 2
        cy = y + h // 2
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        
        # Center-biased composite score
        proximity = max(0.0, 1.0 - dist / max_dist)
        score = area * (proximity ** 2)
        valid_contours.append((x, y, w, h, area, dist, score))

    if not valid_contours:
        print('     [AI-FIRST] CV fallback had only edge/noise contours')
        return None

    # Sort by composite score (highest = most central + largest)
    valid_contours.sort(key=lambda c: c[6], reverse=True)
    main = valid_contours[0]
    cluster_center_x = main[0] + main[2] // 2
    cluster_center_y = main[1] + main[3] // 2
    max_cluster_dist = min(width, height) * 0.35

    cluster = [main]
    for c in valid_contours[1:]:
        cx = c[0] + c[2] // 2
        cy = c[1] + c[3] // 2
        dist = ((cx - cluster_center_x) ** 2 + (cy - cluster_center_y) ** 2) ** 0.5
        if dist <= max_cluster_dist:
            cluster.append(c)
            cluster_center_x = (cluster_center_x + cx) // 2
            cluster_center_y = (cluster_center_y + cy) // 2

    left = min(c[0] for c in cluster)
    top = min(c[1] for c in cluster)
    right = max(c[0] + c[2] for c in cluster)
    bottom = max(c[1] + c[3] for c in cluster)
    print(f'     [AI-FIRST] CV fallback bounds {right-left}x{bottom-top} at ({left},{top})')
    return left, top, right - left, bottom - top



def detect_yellow_logo_bounds(
    image: Image.Image,
    product_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect yellow mascot/logo and return union bounds (left, top, right, bottom).
    Inclusion is compulsory when detected.
    """
    import numpy as np
    import cv2

    img = np.array(image.convert('RGB'))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([15, 90, 90], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = img.shape[:2]
    max_area = width * height * 0.02
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < YELLOW_MIN_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1, h)
        if not (0.3 <= aspect <= 3.0):
            continue

        if product_bounds is not None:
            p_left, p_top, p_right, p_bottom = product_bounds
            cx = x + (w // 2)
            cy = y + (h // 2)
            if p_left <= cx <= p_right and p_top <= cy <= p_bottom:
                # Ignore yellow regions that are clearly on the main product body/screen.
                continue

        valid.append((x, y, x + w, y + h))

    if not valid:
        print('     [LOGO] No yellow mascot/logo detected')
        return None

    left = min(b[0] for b in valid)
    top = min(b[1] for b in valid)
    right = max(b[2] for b in valid)
    bottom = max(b[3] for b in valid)
    print(f'     [LOGO] Included yellow bounds {right-left}x{bottom-top} at ({left},{top})')
    return left, top, right, bottom


def solve_square_crop(
    required_bounds: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    padding_percent: float = 0.06,
    artifact_boxes: list = None,
) -> Tuple[int, int, int, int, dict]:
    """
    Solve for a square crop that contains required_bounds plus padding.

    Unlike pure center-locking, this solver can shift the square within image bounds
    to preserve the requested padding whenever possible.

    Returns:
        (left, top, right, bottom, debug_info)
    """
    width, height = image_size
    req_left, req_top, req_right, req_bottom = required_bounds

    req_left, req_top, req_right, req_bottom = _clip_ltrb(required_bounds, width, height)

    # Calculate dimensions
    req_w = req_right - req_left
    req_h = req_bottom - req_top

    # Calculate padding based on product size
    pad_left = int(req_w * padding_percent)
    pad_right = int(req_w * padding_percent)
    pad_top = int(req_h * padding_percent)
    pad_bottom = int(req_h * padding_percent)

    # Requested square from padded required bounds.
    target_square = max(req_w + pad_left + pad_right, req_h + pad_top + pad_bottom)
    min_cover_square = max(req_w, req_h)
    max_image_square = min(width, height)

    # Fit request inside image limits while still covering required bounds.
    square_size = int(max(min_cover_square, min(target_square, max_image_square)))

    req_cx = (req_left + req_right) / 2.0
    req_cy = (req_top + req_bottom) / 2.0

    def _feasible_interval(axis_len: int, req_lo: int, req_hi: int, size: int):
        lo = max(0, req_hi - size)
        hi = min(axis_len - size, req_lo)
        return lo, hi

    x_lo, x_hi = _feasible_interval(width, req_left, req_right, square_size)
    y_lo, y_hi = _feasible_interval(height, req_top, req_bottom, square_size)

    # If needed, shrink slightly until both axes have a feasible placement.
    while (x_lo > x_hi or y_lo > y_hi) and square_size > min_cover_square:
        square_size -= 1
        x_lo, x_hi = _feasible_interval(width, req_left, req_right, square_size)
        y_lo, y_hi = _feasible_interval(height, req_top, req_bottom, square_size)

    # Final safety fallback.
    if x_lo > x_hi or y_lo > y_hi:
        square_size = max(1, min(width, height))
        x_lo, x_hi = 0, max(0, width - square_size)
        y_lo, y_hi = 0, max(0, height - square_size)

    # Prefer centered placement, then clamp into feasible interval.
    preferred_left = int(round(req_cx - square_size / 2.0))
    preferred_top = int(round(req_cy - square_size / 2.0))
    base_left = max(x_lo, min(x_hi, preferred_left))
    base_top = max(y_lo, min(y_hi, preferred_top))

    def _artifact_penalty(left: int, top: int, size: int) -> float:
        if not artifact_boxes:
            return 0.0
        right = left + size
        bottom = top + size
        penalty = 0.0
        for box in artifact_boxes:
            al, at, ar, ab = box[:4]
            # Ignore artifacts fully inside required object box.
            if al >= req_left and ar <= req_right and at >= req_top and ab <= req_bottom:
                continue
            ix1 = max(left, al)
            iy1 = max(top, at)
            ix2 = min(right, ar)
            iy2 = min(bottom, ab)
            if ix1 < ix2 and iy1 < iy2:
                penalty += float((ix2 - ix1) * (iy2 - iy1))
        return penalty

    # Evaluate a small set of feasible candidates and pick lowest artifact overlap,
    # with tie-breaker favoring closer-to-center placement.
    x_candidates = sorted(set([x_lo, base_left, x_hi]))
    y_candidates = sorted(set([y_lo, base_top, y_hi]))

    best = None
    for left in x_candidates:
        for top in y_candidates:
            penalty = _artifact_penalty(left, top, square_size)
            center_dist = abs((left + square_size / 2.0) - req_cx) + abs((top + square_size / 2.0) - req_cy)
            rank = (penalty, center_dist)
            if best is None or rank < best[0]:
                best = (rank, left, top, penalty)

    _, crop_left, crop_top, penalty = best
    crop_right = crop_left + square_size
    crop_bottom = crop_top + square_size

    debug_info = {
        "padding_percent": float(padding_percent),
        "target_square": int(target_square),
        "final_square": int(square_size),
        "artifact_overlap_px": float(penalty),
    }
    return int(crop_left), int(crop_top), int(crop_right), int(crop_bottom), debug_info


def border_contamination(strip, bg_ref=(255, 255, 255), dark_thresh=40):
    """Score a border strip for non-white contamination (0.0 = clean, 1.0 = dirty)."""
    import numpy as np
    arr = np.asarray(strip, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    diff = np.abs(arr - np.array(bg_ref, dtype=np.float32))
    max_diff = np.max(diff, axis=2)
    dirty = max_diff > dark_thresh
    return float(np.mean(dirty))


def cleanup_crop_borders(cropped: Image.Image, threshold: float = 0.15) -> Image.Image:
    """Post-crop border cleanup: scan 4 border strips + 4 corners for contamination and trim."""
    import numpy as np
    w, h = cropped.size
    if w < 100 or h < 100:
        return cropped

    img = np.array(cropped.convert('RGB'))
    strip_w = max(4, int(w * 0.08))
    strip_h = max(4, int(h * 0.08))

    left_strip = img[:, :strip_w, :]
    right_strip = img[:, w-strip_w:, :]
    top_strip = img[:strip_h, :, :]
    bottom_strip = img[h-strip_h:, :, :]

    l_score = border_contamination(left_strip)
    r_score = border_contamination(right_strip)
    t_score = border_contamination(top_strip)
    b_score = border_contamination(bottom_strip)

    trim_left = strip_w if l_score > threshold else 0
    trim_right = strip_w if r_score > threshold else 0
    trim_top = strip_h if t_score > threshold else 0
    trim_bottom = strip_h if b_score > threshold else 0

    # Corner scan: 4 quadrants scored independently
    corner_w = max(4, int(w * 0.12))
    corner_h = max(4, int(h * 0.12))
    tl_corner = img[:corner_h, :corner_w, :]
    tr_corner = img[:corner_h, w-corner_w:, :]
    bl_corner = img[h-corner_h:, :corner_w, :]
    br_corner = img[h-corner_h:, w-corner_w:, :]

    tl_score = border_contamination(tl_corner)
    tr_score = border_contamination(tr_corner)
    bl_score = border_contamination(bl_corner)
    br_score = border_contamination(br_corner)

    corner_trim_left = corner_w if (tl_score > threshold or bl_score > threshold) else 0
    corner_trim_right = corner_w if (tr_score > threshold or br_score > threshold) else 0
    corner_trim_top = corner_h if (tl_score > threshold or tr_score > threshold) else 0
    corner_trim_bottom = corner_h if (bl_score > threshold or br_score > threshold) else 0

    trim_left = max(trim_left, corner_trim_left)
    trim_right = max(trim_right, corner_trim_right)
    trim_top = max(trim_top, corner_trim_top)
    trim_bottom = max(trim_bottom, corner_trim_bottom)

    if trim_left == 0 and trim_right == 0 and trim_top == 0 and trim_bottom == 0:
        return cropped

    total_trim = max(trim_left, trim_right, trim_top, trim_bottom)
    # Cap trim to preserve the intended padding zone.
    # Allow at most 2% trim — must not eat the 6% padding.
    max_trim = max(1, int(min(w, h) * 0.02))
    total_trim = min(total_trim, max_trim)

    new_left = total_trim
    new_top = total_trim
    new_right = w - total_trim
    new_bottom = h - total_trim

    if new_right - new_left < int(w * 0.80) or new_bottom - new_top < int(h * 0.80):
        print(f"     [BORDER-CLEANUP] Skipped (trim too aggressive: {total_trim}px)")
        return cropped

    print(
        f"     [BORDER-CLEANUP] Trimming {total_trim}px symmetrically "
        f"(scores L={l_score:.2f} R={r_score:.2f} T={t_score:.2f} B={b_score:.2f}, "
        f"corners TL={tl_score:.2f} TR={tr_score:.2f} BL={bl_score:.2f} BR={br_score:.2f})"
    )
    return cropped.crop((new_left, new_top, new_right, new_bottom))

def tight_crop_to_object(image: Image.Image, padding_percent: float = DEFAULT_PADDING_PCT, logger: Optional['CropLogger'] = None) -> Image.Image:
    """
    AI-first crop with white-platform constraints and compulsory logo inclusion.
    Returns cropped (not yet resized) image.
    """
    width, height = image.size
    if image.mode != 'RGB':
        image = image.convert('RGB')

    product_ltrb: Optional[Tuple[int, int, int, int]] = None
    yolo_artifacts = []

    if _yolo_model is not None:
        yolo_bounds, yolo_artifacts = get_yolo_detections(image)
        if yolo_bounds is not None:
            product_ltrb = _xywh_to_ltrb(yolo_bounds)
        else:
            print("     [YOLO] Detection failed. Falling back to U2-Net/CV.")

    if product_ltrb is None:
        ai_bounds = get_ai_crop_bounds(image)
        if ai_bounds is not None:
            product_ltrb = _xywh_to_ltrb(ai_bounds)
        else:
            cv_bounds = get_cv_crop_bounds(image)
            if cv_bounds is not None:
                product_ltrb = _xywh_to_ltrb(cv_bounds)

    if product_ltrb is None:
        # Safety fallback: centered middle box.
        product_ltrb = (
            int(width * 0.35),
            int(height * 0.35),
            int(width * 0.65),
            int(height * 0.65),
        )
        print('     [AI-FIRST] Detector fallback to center box')

    product_ltrb = _clip_ltrb(product_ltrb, width, height)

    # Detect yellow logo — only include if it's close to the product.
    # Far-away logos (e.g. at table corners) are excluded.
    logo_ltrb = detect_yellow_logo_bounds(image, product_bounds=product_ltrb)
    if logo_ltrb is not None:
        p_left, p_top, p_right, p_bottom = product_ltrb
        l_left, l_top, l_right, l_bottom = logo_ltrb
        logo_cx = (l_left + l_right) / 2
        logo_cy = (l_top + l_bottom) / 2
        prod_cx = (p_left + p_right) / 2
        prod_cy = (p_top + p_bottom) / 2
        prod_diag = ((p_right - p_left) ** 2 + (p_bottom - p_top) ** 2) ** 0.5
        logo_dist = ((logo_cx - prod_cx) ** 2 + (logo_cy - prod_cy) ** 2) ** 0.5
        # Include logo only if within 60% of the product diagonal
        if logo_dist <= prod_diag * 0.6:
            required_ltrb = _union_ltrb(product_ltrb, logo_ltrb)
            print(f'     [LOGO] Included (dist={logo_dist:.0f}, threshold={prod_diag*0.6:.0f})')
        else:
            print(f'     [LOGO] Excluded — too far from product (dist={logo_dist:.0f}, threshold={prod_diag*0.6:.0f})')
            required_ltrb = product_ltrb
    else:
        required_ltrb = product_ltrb

    crop_left, crop_top, crop_right, crop_bottom, solver_debug = solve_square_crop(
        required_bounds=required_ltrb,
        image_size=(width, height),
        padding_percent=padding_percent,
        artifact_boxes=yolo_artifacts
    )

    square_size = crop_right - crop_left
    print(
        f'     [CROP-SOLVER] Final crop ({crop_left},{crop_top}) to '
        f'({crop_right},{crop_bottom}), size={square_size}x{square_size}'
    )
    
    if logger:
        logger.log_crop_solve(
            required_bounds=required_ltrb,
            proposed_crop=(crop_left, crop_top, crop_right, crop_bottom),
            square_size=square_size,
            strategy_used="tight_crop_shifted",
            padding_adjustments=solver_debug,
        )
        logger.log_image_dimensions(image.size, (square_size, square_size))
        if yolo_artifacts:
            logger.save_artifacts(yolo_artifacts)
        
    cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return cleanup_crop_borders(cropped)


def resize_to_target(image: Image.Image, target_size: int = TARGET_SIZE) -> Image.Image:
    """Resize image to target dimensions (1080x1080)."""
    if image.size == (target_size, target_size):
        return image
    return image.resize((target_size, target_size), Image.Resampling.LANCZOS)


def apply_white_background(image: Image.Image) -> Image.Image:
    """Apply white background to image, replacing any transparency or other backgrounds."""
    # Create white background
    white_bg = Image.new('RGB', image.size, (255, 255, 255))
    
    if image.mode == 'RGBA':
        # Paste image onto white background using alpha as mask
        white_bg.paste(image, mask=image.split()[3])
    elif image.mode == 'RGB':
        white_bg = image
    else:
        # Convert other modes to RGB
        white_bg = image.convert('RGB')
    
    return white_bg


def apply_watermark(image: Image.Image, watermark_path: Path) -> Image.Image:
    """Overlay the MM watermark at the same position as the reference photo.

    The watermark PNG content is NOT rescaled — native pixels are composited
    directly.  Only the position is matched to the reference output.
    """
    if not watermark_path.exists():
        print(f"     [WATERMARK] File not found: {watermark_path}")
        return image

    import numpy as np

    wm = Image.open(watermark_path).convert('RGBA')

    # Find the bounding box of non-transparent pixels (the actual logo)
    alpha = np.array(wm)[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    # Crop to just the logo content (no resize)
    logo = wm.crop((x_min, y_min, x_max + 1, y_max + 1))

    # Target position: top-left corner matching reference (~2.5% margin)
    img_w, img_h = image.size
    pos_x = int(img_w * 0.025)
    pos_y = int(img_h * 0.025)

    # Composite onto the image
    base = image.convert('RGBA')
    base.paste(logo, (pos_x, pos_y), logo)
    result = base.convert('RGB')

    print(f"     Watermark applied: {logo.width}x{logo.height} at ({pos_x},{pos_y}) [native res]")
    return result

# Max time (seconds) to spend scanning one image for barcodes
MAX_BARCODE_SCAN_TIME = 1.0

def read_barcode(image) -> Optional[str]:
    """Scan image for barcodes using zxing-cpp with optimized preprocessing.

    Optimisation strategy:
      1. Single grayscale conversion, then numpy slicing for crops.
      2. Restrict to 1-D barcode symbologies (IMEI stickers).
      3. 3 ultra-fast native passes (~0.12 s total).
      4. 1 lightweight fallback (adaptive threshold on a *downscaled* top-half).
      5. Hard time-budget of MAX_BARCODE_SCAN_TIME seconds.
    """
    try:
        import zxingcpp
        import numpy as np
        import re
        import cv2

        scan_start = time.time()

        w, h = image.size

        # --- Single grayscale conversion, then slice -------------------------
        gray_full = np.array(image.convert('L'))
        gh, gw = gray_full.shape

        # Top-half slice (rows 0 → 50%)
        top_end = int(gh * 0.5)
        gray_top = gray_full[:top_end, :]

        # Center 90 % slice
        cy1, cy2 = int(gh * 0.05), int(gh * 0.95)
        cx1, cx2 = int(gw * 0.05), int(gw * 0.95)
        gray_center = gray_full[cy1:cy2, cx1:cx2]

        # --- Restrict to 1-D barcode formats (IMEI stickers) -----------------
        linear_formats = (
            zxingcpp.BarcodeFormat.Codabar
            | zxingcpp.BarcodeFormat.Code128
            | zxingcpp.BarcodeFormat.Code39
            | zxingcpp.BarcodeFormat.Code93
            | zxingcpp.BarcodeFormat.EAN8
            | zxingcpp.BarcodeFormat.EAN13
            | zxingcpp.BarcodeFormat.ITF
            | zxingcpp.BarcodeFormat.UPCA
            | zxingcpp.BarcodeFormat.UPCE
        )

        def extract(barcodes):
            if not barcodes:
                return None
            best_code = None
            max_len = 0
            for b in barcodes:
                text = b.text.strip()
                clean = re.sub(r'[^A-Za-z0-9_-]', '', text)
                if clean:
                    if clean.isdigit() and len(clean) == 15:
                        return clean  # IMEI — exact match, instant return
                    if len(clean) > max_len:
                        best_code = clean
                        max_len = len(clean)
            return best_code if best_code else None

        # --- FAST PASSES (native pixels, ~0.04 s each) ----------------------
        for arr in (gray_full, gray_center, gray_top):
            res = extract(zxingcpp.read_barcodes(arr, formats=linear_formats))
            if res:
                return res

        # --- FALLBACK STAGES (each guarded by time budget) --------------------

        # Ensure contiguous arrays for OpenCV (slices may be non-contiguous)
        gray_top_c = np.ascontiguousarray(gray_top)

        # Stage 1: Adaptive threshold on native (or downscaled) top-half
        if (time.time() - scan_start) < MAX_BARCODE_SCAN_TIME:
            max_w = 1500
            if gray_top_c.shape[1] > max_w:
                scale = max_w / gray_top_c.shape[1]
                work_top = cv2.resize(
                    gray_top_c,
                    (max_w, int(gray_top_c.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                work_top = gray_top_c

            athresh = cv2.adaptiveThreshold(
                work_top, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                71, 10,
            )
            res = extract(zxingcpp.read_barcodes(athresh, formats=linear_formats))
            if res:
                return res

        # Stage 2: 2x upscale on top-half (only for low-res images ≤ 1500 px)
        # WhatsApp-compressed images (960px) have bars too thin for native decode
        if gw <= 1500 and (time.time() - scan_start) < MAX_BARCODE_SCAN_TIME:
            up_top = cv2.resize(
                gray_top_c,
                (gray_top_c.shape[1] * 2, gray_top_c.shape[0] * 2),
                interpolation=cv2.INTER_CUBIC,
            )
            res = extract(zxingcpp.read_barcodes(up_top, formats=linear_formats))
            if res:
                return res

            # Stage 2b: Adaptive threshold on upscaled top
            if (time.time() - scan_start) < MAX_BARCODE_SCAN_TIME:
                athresh_up = cv2.adaptiveThreshold(
                    up_top, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                    71, 10,
                )
                res = extract(zxingcpp.read_barcodes(athresh_up, formats=linear_formats))
                if res:
                    return res

            # Stage 2c: CLAHE on upscaled top
            if (time.time() - scan_start) < MAX_BARCODE_SCAN_TIME:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl_up = clahe.apply(up_top)
                res = extract(zxingcpp.read_barcodes(cl_up, formats=linear_formats))
                if res:
                    return res

        # Stage 3: 3x upscale for very low-res images (≤ 1000 px wide)
        # Some WhatsApp-compressed images need extra resolution to resolve bar widths
        if gw <= 1000 and (time.time() - scan_start) < MAX_BARCODE_SCAN_TIME:
            up3_top = cv2.resize(
                gray_top_c,
                (gray_top_c.shape[1] * 3, gray_top_c.shape[0] * 3),
                interpolation=cv2.INTER_CUBIC,
            )
            res = extract(zxingcpp.read_barcodes(up3_top, formats=linear_formats))
            if res:
                return res

        return None

    except Exception as e:
        print(f"     [BARCODE-ERROR] zxing-cpp decode failed: {e}")
    return None

def _get_image_timestr(input_path: Path, image) -> str:
    """Extract 'DateTimeOriginal' EXIF or fallback to file modification time."""
    import datetime
    import os
    try:
        exif = image.getexif()
        if exif is not None:
            # 36867 is DateTimeOriginal, 306 is DateTime
            dt_str = exif.get(36867) or exif.get(306)
            if dt_str:
                # Format is usually: "YYYY:MM:DD HH:MM:SS" -> "YYYYMMDD-HHMMSS"
                dt_obj = datetime.datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                return dt_obj.strftime("%Y%m%d-%H%M%S")
    except Exception:
        pass

    try:
        mtime = os.path.getmtime(input_path)
        dt_obj = datetime.datetime.fromtimestamp(mtime)
        return dt_obj.strftime("%Y%m%d-%H%M%S")
    except Exception:
        pass

    return "UnknownTime"

def process_image(input_path: Path, output_dir: Path, config: dict = None, log_callback = None, progress_callback = None, state: dict = None) -> tuple:
    """Process a single image through the cropping pipeline."""
    def dlog(msg):
        if log_callback: log_callback(msg)
        else: print(msg)

    if config:
        global GLOBAL_BRIGHTNESS, ENABLE_ADAPTIVE_BRIGHTNESS_TARGET, DEFAULT_PADDING_PCT, ENABLE_ANTI_GRAY_CORRECTION
        if 'brightness' in config: GLOBAL_BRIGHTNESS = float(config['brightness'])
        if 'adaptive_brightness' in config: ENABLE_ADAPTIVE_BRIGHTNESS_TARGET = bool(config['adaptive_brightness'])
        if 'padding' in config: DEFAULT_PADDING_PCT = float(config['padding'])
        if 'anti_gray' in config: ENABLE_ANTI_GRAY_CORRECTION = bool(config['anti_gray'])

    dlog(f"\n{'='*50}")
    dlog(f"Processing: {input_path.name}")
    dlog('='*50)
    
    start_time = time.time()
    
    logger = CropLogger("crop_logs")
    logger.start_entry(input_path.name)
    
    try:
        # Step 1: Load
        dlog("  -> Loading image...")
        original = load_image(input_path)
        dlog(f"     Original: {original.width}x{original.height}")
        logger.save_original(original)
        
        # --- BARCODE PRE-PROCESS ---
        # Naming convention:
        #   Barcode photo  -> "<last6> 00.png"          (sorts first)
        #   Device photo 1 -> "<last6> 1a.png"          (a-z sequence)
        #   Device photo 2 -> "<last6> 1b.png"
        #   ... after 1z   -> "<last6> 2a.png", etc.
        output_filename = f"{input_path.stem}.png"
        skip_pipeline = False
        if config and config.get('scan_barcodes', False):
            dlog("  -> Scanning for IMEI barcode...")
            detected_barcode = read_barcode(original)
            if detected_barcode:
                dlog(f"     [BARCODE] Found IMEI: {detected_barcode}")
                last6 = detected_barcode[-6:]
                if state is not None:
                    state["current_imei"] = detected_barcode
                    state["imei_last6"] = last6
                    state["imei_seq"] = 0  # reset sequence counter
                output_filename = f"{last6} 00.png"
                skip_pipeline = True
            else:
                if state is not None and state.get("current_imei"):
                    last6 = state["imei_last6"]
                    seq = state.get("imei_seq", 0)
                    # Convert sequence to <digit><letter>: 0->1a, 1->1b, ... 25->1z, 26->2a
                    digit = (seq // 26) + 1
                    letter = chr(ord('a') + (seq % 26))
                    output_filename = f"{last6} {digit}{letter}.png"
                    state["imei_seq"] = seq + 1
                    dlog(f"     [BARCODE] No barcode. Grouping under IMEI: {output_filename}")
                else:
                    dlog(f"     [BARCODE] No barcode found, using original name: {output_filename}")
        # -----------------------------

        if skip_pipeline:
            dlog("  -> Barcode detected. Skipping cropping and enhancement pipeline.")
            final = original
        else:
            # Step 2: Tight crop to object
            dlog("  -> Tight cropping to object...")
            active_padding = DEFAULT_PADDING_PCT
            if config and 'padding' in config:
                try:
                    active_padding = float(config['padding'])
                except Exception:
                    active_padding = DEFAULT_PADDING_PCT
            # Guard against invalid values coming from external callers.
            active_padding = max(0.0, min(0.25, active_padding))
            dlog(f"     Crop padding: {active_padding*100:.1f}%")
            cropped = tight_crop_to_object(original, padding_percent=active_padding, logger=logger)
            dlog(f"     Output: {cropped.width}x{cropped.height}")
            
            # Step 3: Optional localized backdrop normalization
            if ENABLE_BACKDROP_NORMALIZATION:
                dlog("  -> Normalizing backdrop...")
                pre_enhance = brighten_backdrop(cropped, target_brightness=BACKDROP_TARGET_BRIGHTNESS)
            else:
                dlog("  -> Skipping localized backdrop normalization (global enhance mode)...")
                pre_enhance = cropped
            
            # Step 4: Enhance image (adaptive sharpening and contrast)
            dlog("  -> Enhancing image...")
            enhanced = enhance_image(pre_enhance)
            
            # Step 5: Resize to target output
            dlog("  -> Resizing to 1080x1080...")
            final = resize_to_target(enhanced, target_size=TARGET_SIZE)

            # Step 6: Apply MM Watermark
            watermark_path = _resolve_asset_path("MM Watermark.png")
            if watermark_path is not None and watermark_path.exists():
                dlog("  -> Applying MM watermark...")
                final = apply_watermark(final, watermark_path)

        # Step 7: Save
        # Use output_filename from barcode preprocess
        output_path = output_dir / output_filename
        
        
        dlog(f"  -> Saving: {output_path.name}")
        final.save(output_path, 'PNG', quality=95)
        
        logger.save_cropped_output(final)
        logger.finalize()
        
        elapsed = time.time() - start_time
        dlog(f"  [OK] Complete! ({elapsed:.1f}s)")
        return (input_path, output_path, True)
        
    except Exception as e:
        dlog(f"  [ERROR] {str(e)}")
        return (input_path, None, False)


def generate_comparison(input_path: Path, output_path: Path, comparison_dir: Path) -> Optional[Path]:
    """Generate a side-by-side comparison image (input left, output right)."""
    try:
        original = load_image(input_path)
        processed = Image.open(output_path).convert('RGB')

        target_h = 1080
        orig_scale = target_h / original.height
        orig_w = int(original.width * orig_scale)
        original_resized = original.resize((orig_w, target_h), Image.LANCZOS)
        proc_resized = processed.resize((target_h, target_h), Image.LANCZOS)

        gap = 20
        label_h = 40
        canvas_w = orig_w + gap + target_h
        canvas_h = target_h + label_h
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))

        canvas.paste(original_resized, (0, label_h))
        canvas.paste(proc_resized, (orig_w + gap, label_h))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()

        draw.text((orig_w // 2 - 30, 8), "INPUT", fill=(100, 100, 100), font=font)
        draw.text((orig_w + gap + target_h // 2 - 40, 8), "OUTPUT", fill=(0, 128, 0), font=font)
        draw.line([(orig_w + gap // 2, label_h), (orig_w + gap // 2, canvas_h)], fill=(200, 200, 200), width=2)

        comp_name = f"{input_path.stem}_comparison.png"
        comp_path = comparison_dir / comp_name
        canvas.save(comp_path, 'PNG')
        return comp_path
    except Exception as e:
        print(f"  [COMPARE-ERROR] {input_path.name}: {e}")
        return None


def generate_all_comparisons(results: list, comparison_dir: Path):
    """Generate side-by-side comparisons for all successful results."""
    comparison_dir.mkdir(parents=True, exist_ok=True)
    successful = [(inp, out) for inp, out, ok in results if ok and out is not None]

    if not successful:
        print("\n[COMPARE] No successful results to compare.")
        return

    print(f"\n{'='*60}")
    print(f"  GENERATING COMPARISONS ({len(successful)} images)")
    print(f"{'='*60}")

    for idx, (inp, out) in enumerate(successful, 1):
        comp = generate_comparison(inp, out, comparison_dir)
        if comp:
            print(f"  [{idx}/{len(successful)}] {comp.name}")

    print(f"\n  [DIR] Comparisons: {comparison_dir}")


def _resolve_exe_dir() -> Path:
    """Return the directory where the executable (or script) lives.

    Works for both normal Python execution and Nuitka-compiled binaries.
    In Nuitka onefile mode, sys.executable points to a temp extraction
    directory — sys.argv[0] always points to the real .exe on disk.
    """
    if "__compiled__" in globals() or getattr(sys, 'frozen', False):
        return Path(os.path.abspath(sys.argv[0])).parent
    return Path(__file__).parent


def _resolve_asset_path(filename: str) -> Optional[Path]:
    """Find asset next to exe first, then bundled temp path for onefile builds."""
    exe_candidate = _resolve_exe_dir() / filename
    if exe_candidate.exists():
        return exe_candidate

    script_candidate = Path(__file__).parent / filename
    if script_candidate.exists():
        return script_candidate

    return None


def _pause_exit(prompt: str = "  Press Enter to exit...") -> None:
    """Pause if interactive, but never crash on missing stdin."""
    try:
        input(prompt)
    except Exception:
        pass


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Carousell Image Cropper')
    parser.add_argument('input', nargs='?', type=Path, default=None,
                        help='Input folder or image file (default: ./input)')
    parser.add_argument('--output', type=Path, help='Output directory (default: ./output)')
    parser.add_argument('--no-yolo', action='store_true', help='Use U2-Net instead of custom YOLO model')
    parser.add_argument("--yolo-weights", type=str, 
                      default="runs/detect/carocrop_custom_fast4/weights/best.pt",
                      help="Path to YOLO weights (.pt file). Default: runs/detect/carocrop_custom_fast4/weights/best.pt")
    args = parser.parse_args()
    
    # We invert the logic so YOLO is used by default unless --no-yolo is passed
    args.use_yolo = not args.no_yolo

    print("\n" + "="*60)
    print("  MISTER MOBILE - CAROUSELL IMAGE CROPPER  v3.0.0")
    print("  1080x1080 Square  |  AI Object Detection")
    print("="*60)

    exe_dir = _resolve_exe_dir()
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}

    # ------------------------------------------------------------------
    # Resolve INPUT — accepts a folder, a single image, or defaults to
    # an interactive prompt so users can drag-and-drop paths.
    # ------------------------------------------------------------------
    input_path = args.input
    if input_path is None:
        # Interactive: ask user (works nicely in compiled .exe console)
        print("\n  Drag & drop a folder or image here, then press Enter.")
        print("  Or just press Enter to use the default ./input folder.")
        raw = input("\n  Path: ").strip().strip('"').strip("'")
        input_path = Path(raw) if raw else exe_dir / "input"

    input_path = Path(input_path)

    # Auto-create default input/output folders next to the exe
    if not input_path.exists():
        # Only auto-create if it's the default ./input folder
        default_input = exe_dir / "input"
        if input_path == default_input:
            input_path.mkdir(parents=True, exist_ok=True)
            print(f"\n  [INFO] Created input folder: {input_path}")
            print("         Place your images there and run again.")
            _pause_exit("\n  Press Enter to exit...")
            sys.exit(0)
        print(f"\n  [ERROR] Path not found: {input_path}")
        _pause_exit("\n  Press Enter to exit...")
        sys.exit(1)

    # Determine if input is a single file or a directory
    if input_path.is_file():
        if input_path.suffix.lower() not in supported_formats:
            print(f"\n  [ERROR] Unsupported format: {input_path.suffix}")
            _pause_exit("\n  Press Enter to exit...")
            sys.exit(1)
        image_files = [input_path]
        input_dir = input_path.parent
    else:
        input_dir = input_path
        image_files = [f for f in input_dir.rglob('*')
                       if f.is_file() and f.suffix.lower() in supported_formats]

    # Output directory
    output_dir = args.output if args.output else exe_dir / "output"
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"\n  [ERROR] Cannot access/create folders: {e}")
        print("         Try a writable location (not protected by OneDrive/Windows Security).")
        _pause_exit("\n  Press Enter to exit...")
        sys.exit(1)

    print(f"\n  [DIR] Input:  {input_dir}")
    print(f"  [DIR] Output: {output_dir}")
    print(f"  Supported formats: JPG, PNG, WebP, HEIC")
    
    if not image_files:
        print("\n  [!] No images found.")
        print("      Place images or subfolders in the input path.")
        _pause_exit("\n  Press Enter to exit...")
        sys.exit(1)
    
    print(f"\n  [SCAN] Found {len(image_files)} image(s)")

    # Pre-load AI model once before the batch loop
    print("")
    if args.use_yolo:
        if not os.path.exists(args.yolo_weights):
            dlog(f"  [ERROR] YOLO weights file not found at {args.yolo_weights}")
            sys.exit(1)
        init_yolo_model(args.yolo_weights)
    else:
        _get_rembg_session()
    
    # Process with progress
    total_start = time.time()
    total = len(image_files)
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        percent = (idx / total) * 100
        bar_len = 25
        filled = int(bar_len * idx / total)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\n[{bar}] {percent:.0f}% ({idx}/{total})")
        
        try:
            rel_path = image_path.relative_to(input_dir)
            target_out = output_dir / rel_path.parent
            target_out.mkdir(parents=True, exist_ok=True)
            result = process_image(image_path, target_out)
            results.append(result)
        except Exception as e:
            dlog(f"  [ERROR] {e}")
            results.append((image_path, None, False))
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    total_time = time.time() - total_start
    success = sum(1 for r in results if r[2])
    failed = sum(1 for r in results if not r[2])
    
    print(f"\n  [+] Processed: {success}")
    print(f"  [-] Failed:    {failed}")
    print(f"  [TIME] Total:  {total_time:.1f}s")
    print(f"\n  [DIR] Output:  {output_dir}")
    print("\n" + "="*60)
    print("  Done! Images ready for Carousell upload.")
    print("="*60 + "\n")

    _pause_exit("  Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n[FATAL] Unhandled error:")
        traceback.print_exc()
        _pause_exit("\n  Press Enter to exit...")
        sys.exit(1)
