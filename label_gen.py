"""
MECHMAXX / TrackFlow shipping-label image compositor.

Base photo assumptions:
- SHIP FROM is static in base.png
- Barcode is static in base.png
- SHIP TO area and tracking-number text are the only dynamic fields

This version:
- patches only SHIP TO content area
- patches only tracking number text area
- preserves original barcode by restoring it from base.png
- adds a subtle softness / blur so the label looks more natural
"""

import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base.png")

# Latest approved MECHMAXX base photo size:
# 1447 x 1087
#
# Label corners in the base image:
# top-left, top-right, bottom-right, bottom-left
LABEL_CORNERS = np.array([
    [691, 289],
    [991, 326],
    [971, 490],
    [671, 452],
], dtype=np.float32)

# Internal flat label canvas
LABEL_W = 1000
LABEL_H = 560

# Only patch the SHIP TO content area (not the "SHIP TO:" header)
SHIP_TO_PATCH_RECT = (510, 145, 845, 300)
SHIP_TO_TEXT_START = (528, 152)

# Patch only the tracking number text area under barcode
TRACKING_PATCH_RECT = (305, 505, 700, 548)
TRACKING_TEXT_RECT = (305, 505, 700, 548)

# Barcode area remains static from base image
BARCODE_LOCK_RECT = (55, 382, 903, 486)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 25)
APPLY_REALISM_PASS = True


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_font(path: str, size: int):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)

    fallback_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]

    for fp in fallback_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)

    return ImageFont.load_default()


def _label_to_image_matrix() -> np.ndarray:
    src = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src, LABEL_CORNERS)


def _image_to_label_matrix() -> np.ndarray:
    dst = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(LABEL_CORNERS, dst)


def normalize_tracking(tracking_number: str | None) -> str:
    raw = (tracking_number or "").strip().upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)

    if raw.startswith("TF-") and len(raw) > 3:
        return raw

    core = re.sub(r"[^A-Z0-9]", "", raw)

    if not core:
        core = "QZ6HTDVPRH"

    return f"TF-{core}"


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill,
    max_width: int,
    line_h: int,
) -> int:
    x, y = xy
    words = text.split()
    line = ""

    for word in words:
        test_line = (line + " " + word).strip()

        if draw.textlength(test_line, font=font) <= max_width:
            line = test_line
        else:
            if line:
                draw.text((x, y), line, font=font, fill=fill)
                y += line_h
            line = word

    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h

    return y


def _ship_to_lines(recipient: dict) -> list[str]:
    name = (recipient.get("name") or "").strip()
    line1 = (recipient.get("line1") or "").strip()
    line2 = (recipient.get("line2") or "").strip()
    line3 = (recipient.get("line3") or "").strip()
    line4 = (recipient.get("line4") or "").strip()
    phone = (recipient.get("phone") or "").strip()

    lines = []

    if name:
        lines.append(name)

    if line1:
        lines.append(line1)

    # Keep US / country immediately after address block
    if line2 and line3:
        lines.append(f"{line2} {line3}")
    elif line2:
        lines.append(line2)
    elif line3:
        lines.append(line3)

    if line4:
        lines.append(line4)

    if phone:
        lines.append(phone)

    return lines


def _sample_paper_color(base_bgr: np.ndarray) -> tuple[int, int, int]:
    """
    Sample clean paper color from the label after flattening perspective.
    """
    M = _image_to_label_matrix()
    flat = cv2.warpPerspective(base_bgr, M, (LABEL_W, LABEL_H), flags=cv2.INTER_CUBIC)

    sample_points = [
        (100, 60),
        (930, 60),
        (950, 180),
        (920, 520),
        (120, 520),
        (480, 520),
        (970, 260),
    ]

    samples = []

    for x, y in sample_points:
        if 0 <= x < LABEL_W and 0 <= y < LABEL_H:
            b, g, r = flat[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (232, 232, 232)

    arr = np.array(samples, dtype=np.float32)
    median = np.median(arr, axis=0)

    # Slightly pull toward neutral gray for a clean paper tone
    gray = median.mean()
    final = median * 0.88 + gray * 0.12

    return tuple(int(v) for v in final)


def _paper_patch(size: tuple[int, int], paper_rgb: tuple[int, int, int], seed: int = 7) -> Image.Image:
    """
    Build a slightly textured paper patch with feathered edges.
    """
    w, h = size
    rng = np.random.default_rng(seed)

    base = np.zeros((h, w, 4), dtype=np.uint8)
    base[:, :, 0] = paper_rgb[0]
    base[:, :, 1] = paper_rgb[1]
    base[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 2.2, (h, w, 1))
    rgb = np.clip(base[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    base[:, :, :3] = rgb

    alpha = np.ones((h, w), dtype=np.float32) * 245.0

    feather = 8
    for i in range(feather):
        a = 245.0 * ((i + 1) / feather)
        alpha[i, :] = np.minimum(alpha[i, :], a)
        alpha[h - 1 - i, :] = np.minimum(alpha[h - 1 - i, :], a)
        alpha[:, i] = np.minimum(alpha[:, i], a)
        alpha[:, w - 1 - i] = np.minimum(alpha[:, w - 1 - i], a)

    base[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(base, "RGBA")


def _paste_paper_rect(
    overlay: Image.Image,
    rect: tuple[int, int, int, int],
    paper_rgb: tuple[int, int, int],
    seed: int
) -> None:
    x0, y0, x1, y1 = rect
    patch = _paper_patch((x1 - x0, y1 - y0), paper_rgb, seed=seed)
    overlay.alpha_composite(patch, (x0, y0))


# ---------------------------------------------------------------------------
# OVERLAY CREATION
# ---------------------------------------------------------------------------

def build_label_overlay(
    recipient: dict,
    tracking_number: str,
    paper_rgb: tuple[int, int, int],
) -> Image.Image:
    """
    Transparent overlay that:
    - covers old broken SHIP TO text with paper patch
    - covers old tracking text with paper patch
    - redraws them cleanly
    """

    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Paper patches first
    _paste_paper_rect(overlay, SHIP_TO_PATCH_RECT, paper_rgb, seed=11)
    _paste_paper_rect(overlay, TRACKING_PATCH_RECT, paper_rgb, seed=12)

    # Fonts
    f_text = _load_font(FONT_REG, 27)
    f_text_bold = _load_font(FONT_BOLD, 28)
    f_tracking = _load_font(FONT_BOLD, 24)

    # SHIP TO text
    lines = _ship_to_lines(recipient)

    total_chars = sum(len(x) for x in lines)

    # Auto shrink a little if needed
    if total_chars > 70 or len(lines) > 4:
        f_text = _load_font(FONT_REG, 25)
        f_text_bold = _load_font(FONT_BOLD, 26)
        line_h = 31
    else:
        line_h = 33

    x, y = SHIP_TO_TEXT_START
    max_w = SHIP_TO_PATCH_RECT[2] - SHIP_TO_PATCH_RECT[0] - 18

    for i, line in enumerate(lines):
        if not line:
            continue

        font = f_text_bold if i == 0 else f_text
        y = _draw_wrapped(
            draw,
            (x, y),
            line,
            font,
            TEXT_COLOR + (242,),
            max_w,
            line_h,
        )

    # Tracking number under barcode
    trk = normalize_tracking(tracking_number)

    x0, y0, x1, y1 = TRACKING_TEXT_RECT
    text_w = draw.textlength(trk, font=f_tracking)

    draw.text(
        (x0 + ((x1 - x0) - text_w) / 2, y0 + 1),
        trk,
        font=f_tracking,
        fill=TEXT_COLOR + (242,),
    )

    return overlay


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def warp_overlay_to_image(
    overlay_rgba: Image.Image,
    base_shape: tuple[int, int, int],
) -> np.ndarray:
    h, w = base_shape[:2]
    overlay_np = np.array(overlay_rgba)

    M = _label_to_image_matrix()

    warped = cv2.warpPerspective(
        overlay_np,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Slight blur on the overlay itself so new text blends naturally
    warped = cv2.GaussianBlur(warped, (3, 3), 0.45)

    return warped


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    overlay_bgr = cv2.cvtColor(
        rgb.astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    ).astype(np.float32)

    out = base_bgr.astype(np.float32) * (1 - alpha) + overlay_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_locked_barcode_last(
    result_bgr: np.ndarray,
    base_bgr: np.ndarray,
) -> np.ndarray:
    """
    Restore the original barcode from base.png.
    """
    M = _label_to_image_matrix()
    x0, y0, x1, y1 = BARCODE_LOCK_RECT

    flat_pts = np.array([
        [[x0, y0]],
        [[x1, y0]],
        [[x1, y1]],
        [[x0, y1]],
    ], dtype=np.float32)

    img_pts = cv2.perspectiveTransform(flat_pts, M).reshape(-1, 2).astype(np.int32)

    mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, img_pts, 255)

    out = result_bgr.copy()
    out[mask == 255] = base_bgr[mask == 255]

    return out


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Add a tiny amount of softness / blur to hide small imperfections
    while keeping the content readable.
    """
    if not APPLY_REALISM_PASS:
        return img_bgr

    # Tiny softness
    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.20)

    # Very light JPEG recompression look
    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_label(
    recipient: dict,
    tracking_number: str | None = None,
    out_path: str | None = None,
):
    base_bgr = cv2.imread(BASE_IMAGE_PATH)

    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_rgb = _sample_paper_color(base_bgr)

    overlay = build_label_overlay(
        recipient=recipient,
        tracking_number=tracking_number or "",
        paper_rgb=paper_rgb,
    )

    warped_overlay = warp_overlay_to_image(
        overlay,
        base_bgr.shape,
    )

    result_bgr = alpha_composite_bgr(
        base_bgr,
        warped_overlay,
    )

    # Add slight softness
    result_bgr = apply_photo_realism(result_bgr)

    # Restore barcode last so it stays static and clean
    result_bgr = paste_locked_barcode_last(
        result_bgr,
        base_bgr,
    )

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_img = Image.fromarray(result_rgb)

    if out_path:
        ext = os.path.splitext(out_path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            result_img.save(out_path, quality=95, subsampling=0)
        else:
            result_img.save(out_path)

    return result_img


# ---------------------------------------------------------------------------
# LOCAL TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    img = generate_label(
        recipient={
            "name": "Steve Cross",
            "line1": "9200 Delashmit Road",
            "line2": "Millington, TN",
            "line3": "38053",
            "line4": "US",
            "phone": "",
        },
        tracking_number="TF-QZ6HTDVPRH",
        out_path="test_mechmaxx_output.png",
    )

    print("done", img.size)
