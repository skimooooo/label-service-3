"""
MECHMAXX / TrackFlow shipping-label image compositor.

Static from base.png:
- SHIP FROM
- barcode
- label background / paper texture

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

LABEL_CORNERS = np.array([
    [691, 289],
    [991, 326],
    [971, 490],
    [671, 452],
], dtype=np.float32)

LABEL_W = 1000
LABEL_H = 560

SHIP_TO_PATCH_RECT = (500, 125, 900, 305)
SHIP_TO_TEXT_START = (525, 145)

TRACKING_PATCH_RECT = (285, 505, 720, 552)
TRACKING_TEXT_RECT = (285, 505, 720, 552)

BARCODE_LOCK_RECT = (55, 382, 903, 486)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (28, 28, 28)

OVERLAY_BLUR = 0.85
FINAL_BLUR = 0.45
JPEG_QUALITY = 84


# ---------------------------------------------------------------------------
# FONT HELPERS
# ---------------------------------------------------------------------------

def load_font(path, size):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)

    fallback_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]

    for fallback in fallback_paths:
        if os.path.exists(fallback):
            return ImageFont.truetype(fallback, size)

    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# PERSPECTIVE HELPERS
# ---------------------------------------------------------------------------

def label_to_image_matrix():
    src = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src, LABEL_CORNERS)


def image_to_label_matrix():
    dst = np.array([
        [0, 0],
        [LABEL_W, 0],
        [LABEL_W, LABEL_H],
        [0, LABEL_H],
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(LABEL_CORNERS, dst)


# ---------------------------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------------------------

def normalize_tracking(tracking_number):
    raw = (tracking_number or "").strip().upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)

    if raw.startswith("TF-") and len(raw) > 3:
        return raw

    core = re.sub(r"[^A-Z0-9]", "", raw)

    if not core:
        core = "QZ6HTDVPRH"

    return f"TF-{core}"


def ship_to_lines(recipient):
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


def draw_wrapped(draw, xy, text, font, fill, max_width, line_h):
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


# ---------------------------------------------------------------------------
# PAPER PATCH
# ---------------------------------------------------------------------------

def sample_paper_color(base_bgr):
    matrix = image_to_label_matrix()

    flat = cv2.warpPerspective(
        base_bgr,
        matrix,
        (LABEL_W, LABEL_H),
        flags=cv2.INTER_CUBIC,
    )

    sample_points = [
        (90, 60),
        (930, 70),
        (940, 180),
        (930, 300),
        (120, 520),
        (500, 520),
        (880, 520),
    ]

    samples = []

    for x, y in sample_points:
        if 0 <= x < LABEL_W and 0 <= y < LABEL_H:
            b, g, r = flat[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (233, 233, 233)

    arr = np.array(samples, dtype=np.float32)
    median = np.median(arr, axis=0)

    gray = median.mean()
    final = median * 0.86 + gray * 0.14

    return tuple(int(v) for v in final)


def paper_patch(size, paper_rgb, seed=7):
    w, h = size
    rng = np.random.default_rng(seed)

    base = np.zeros((h, w, 4), dtype=np.uint8)
    base[:, :, 0] = paper_rgb[0]
    base[:, :, 1] = paper_rgb[1]
    base[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 2.8, (h, w, 1))

    rgb = np.clip(
        base[:, :, :3].astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    base[:, :, :3] = rgb

    alpha = np.ones((h, w), dtype=np.float32) * 253.0
    feather = 12

    for i in range(feather):
        a = 253.0 * ((i + 1) / feather)

        alpha[i, :] = np.minimum(alpha[i, :], a)
        alpha[h - 1 - i, :] = np.minimum(alpha[h - 1 - i, :], a)
        alpha[:, i] = np.minimum(alpha[:, i], a)
        alpha[:, w - 1 - i] = np.minimum(alpha[:, w - 1 - i], a)

    base[:, :, 3] = alpha.astype(np.uint8)

    return Image.fromarray(base, "RGBA")


def paste_paper_rect(overlay, rect, paper_rgb, seed):
    x0, y0, x1, y1 = rect

    patch = paper_patch(
        (x1 - x0, y1 - y0),
        paper_rgb,
        seed=seed,
    )

    overlay.alpha_composite(patch, (x0, y0))


# ---------------------------------------------------------------------------
# OVERLAY CREATION
# ---------------------------------------------------------------------------

def build_label_overlay(recipient, tracking_number, paper_rgb):
    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    paste_paper_rect(overlay, SHIP_TO_PATCH_RECT, paper_rgb, seed=11)
    paste_paper_rect(overlay, TRACKING_PATCH_RECT, paper_rgb, seed=12)

    lines = ship_to_lines(recipient)

    f_text = load_font(FONT_REG, 30)
    f_text_bold = load_font(FONT_BOLD, 31)
    f_tracking = load_font(FONT_BOLD, 25)

    total_chars = sum(len(x) for x in lines)

    if total_chars > 78 or len(lines) > 4:
        f_text = load_font(FONT_REG, 27)
        f_text_bold = load_font(FONT_BOLD, 28)
        line_h = 33
    else:
        line_h = 37

    x, y = SHIP_TO_TEXT_START
    max_w = SHIP_TO_PATCH_RECT[2] - SHIP_TO_PATCH_RECT[0] - 30

    for i, line in enumerate(lines):
        if not line:
            continue

        font = f_text_bold if i == 0 else f_text

        y = draw_wrapped(
            draw,
            (x, y),
            line,
            font,
            TEXT_COLOR + (238,),
            max_w,
            line_h,
        )

    tracking = normalize_tracking(tracking_number)

    x0, y0, x1, y1 = TRACKING_TEXT_RECT
    text_w = draw.textlength(tracking, font=f_tracking)

    draw.text(
        (x0 + ((x1 - x0) - text_w) / 2, y0 + 4),
        tracking,
        font=f_tracking,
        fill=TEXT_COLOR + (238,),
    )

    return overlay


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def warp_overlay_to_image(overlay_rgba, base_shape):
    h, w = base_shape[:2]

    overlay_np = np.array(overlay_rgba)
    matrix = label_to_image_matrix()

    warped = cv2.warpPerspective(
        overlay_np,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )

    warped = cv2.GaussianBlur(warped, (3, 3), OVERLAY_BLUR)

    return warped


def alpha_composite_bgr(base_bgr, overlay_rgba):
    rgb = overlay_rgba[:, :, :3].astype(np.float32)

    alpha = (
        overlay_rgba[:, :, 3].astype(np.float32) / 255.0
    )[:, :, None]

    overlay_bgr = cv2.cvtColor(
        rgb.astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    ).astype(np.float32)

    out = base_bgr.astype(np.float32) * (1 - alpha) + overlay_bgr * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def paste_locked_barcode_last(result_bgr, base_bgr):
    matrix = label_to_image_matrix()

    x0, y0, x1, y1 = BARCODE_LOCK_RECT

    flat_points = np.array([
        [[x0, y0]],
        [[x1, y0]],
        [[x1, y1]],
        [[x0, y1]],
    ], dtype=np.float32)

    image_points = cv2.perspectiveTransform(
        flat_points,
        matrix,
    ).reshape(-1, 2).astype(np.int32)

    mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)

    cv2.fillConvexPoly(mask, image_points, 255)

    out = result_bgr.copy()
    out[mask == 255] = base_bgr[mask == 255]

    return out


def apply_photo_realism(img_bgr):
    img = cv2.GaussianBlur(img_bgr, (3, 3), FINAL_BLUR)

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )

    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_label(recipient, tracking_number=None, out_path=None):
    base_bgr = cv2.imread(BASE_IMAGE_PATH)

    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_rgb = sample_paper_color(base_bgr)

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

    result_bgr = apply_photo_realism(result_bgr)

    result_bgr = paste_locked_barcode_last(
        result_bgr,
        base_bgr,
    )

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    result_img = Image.fromarray(result_rgb)

    if out_path:
        ext = os.path.splitext(out_path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            result_img.save(out_path, quality=90, subsampling=0)
        else:
            result_img.save(out_path)

    return result_img


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
