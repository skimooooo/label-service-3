"""
MECHMAXX / TrackFlow shipping-label image compositor.

Made for the latest clean MECHMAXX base photo:
- SHIP FROM is static in base.png
- SHIP TO area is blank in base.png
- Barcode is static in base.png
- Tracking number text under barcode is blank in base.png

Dynamic:
- SHIP TO address
- tracking number text under barcode
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

# Latest approved base photo size: 1447 x 1087
# Label corners on the image:
# top-left, top-right, bottom-right, bottom-left
LABEL_CORNERS = np.array([
    [688, 301],
    [982, 347],
    [974, 504],
    [626, 452],
], dtype=np.float32)

# Flat label canvas used internally
LABEL_W = 1000
LABEL_H = 560

# Blank SHIP TO writing area inside the label
# This does NOT cover the SHIP TO title.
SHIP_TO_TEXT_AREA = (535, 118, 955, 310)

# Blank tracking number text area under the barcode
TRACKING_TEXT_AREA = (90, 485, 920, 535)

# Barcode area copied back from base image, so barcode never changes
BARCODE_LOCK_RECT = (90, 360, 920, 480)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 25)


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

    # Example output:
    # Millington, TN 38053
    if line2 and line3:
        lines.append(f"{line2} {line3}")
    elif line2:
        lines.append(line2)
    elif line3:
        lines.append(line3)

    # Country comes directly after address, no big gap
    if line4:
        lines.append(line4)

    if phone:
        lines.append(phone)

    return lines


# ---------------------------------------------------------------------------
# OVERLAY CREATION
# ---------------------------------------------------------------------------

def build_label_overlay(
    recipient: dict,
    tracking_number: str,
) -> Image.Image:
    """
    Transparent overlay:
    - only contains SHIP TO dynamic text
    - only contains tracking number text
    - does not redraw the whole label
    """

    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    f_text = _load_font(FONT_REG, 26)
    f_text_bold = _load_font(FONT_BOLD, 26)
    f_tracking = _load_font(FONT_BOLD, 24)

    # -----------------------------------------------------------------------
    # SHIP TO dynamic text
    # -----------------------------------------------------------------------

    x0, y0, x1, y1 = SHIP_TO_TEXT_AREA

    y = y0
    max_w = x1 - x0 - 10
    line_h = 34

    lines = _ship_to_lines(recipient)

    total_chars = sum(len(x) for x in lines)

    # Auto-shrink if address is long
    if total_chars > 90 or len(lines) > 4:
        f_text = _load_font(FONT_REG, 23)
        f_text_bold = _load_font(FONT_BOLD, 23)
        line_h = 30

    for i, line in enumerate(lines):
        if not line:
            continue

        font = f_text_bold if i == 0 else f_text

        y = _draw_wrapped(
            draw,
            (x0, y),
            line,
            font,
            TEXT_COLOR + (245,),
            max_w,
            line_h,
        )

    # -----------------------------------------------------------------------
    # Tracking number under barcode
    # -----------------------------------------------------------------------

    trk = normalize_tracking(tracking_number)

    x0, y0, x1, y1 = TRACKING_TEXT_AREA

    text_w = draw.textlength(trk, font=f_tracking)

    draw.text(
        (x0 + ((x1 - x0) - text_w) / 2, y0),
        trk,
        font=f_tracking,
        fill=TEXT_COLOR + (245,),
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

    # Tiny blur so inserted text does not look digitally pasted
    warped = cv2.GaussianBlur(warped, (3, 3), 0.25)

    return warped


def alpha_composite_bgr(
    base_bgr: np.ndarray,
    overlay_rgba: np.ndarray,
) -> np.ndarray:
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
    Restore barcode from the original base image.
    This means the barcode is always static and never regenerated.
    """

    M = _label_to_image_matrix()

    x0, y0, x1, y1 = BARCODE_LOCK_RECT

    flat_pts = np.array([
        [[x0, y0]],
        [[x1, y0]],
        [[x1, y1]],
        [[x0, y1]],
    ], dtype=np.float32)

    img_pts = cv2.perspectiveTransform(
        flat_pts,
        M,
    ).reshape(-1, 2).astype(np.int32)

    mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, img_pts, 255)

    out = result_bgr.copy()
    out[mask == 255] = base_bgr[mask == 255]

    return out


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """
    Very light compression only.
    Do not destroy readability.
    """

    ok, encoded = cv2.imencode(
        ".jpg",
        img_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 93],
    )

    if ok:
        img_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img_bgr


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

    overlay = build_label_overlay(
        recipient=recipient,
        tracking_number=tracking_number or "",
    )

    warped_overlay = warp_overlay_to_image(
        overlay,
        base_bgr.shape,
    )

    result_bgr = alpha_composite_bgr(
        base_bgr,
        warped_overlay,
    )

    # Apply light realism first
    result_bgr = apply_photo_realism(result_bgr)

    # Restore original barcode last so it stays static
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
