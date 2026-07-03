"""
MECHMAXX / TrackFlow shipping-label image compositor.

This version uses a safer method:
1. Flatten the existing label from the base photo.
2. Edit only:
   - SHIP TO block
   - tracking number text
3. Keep SHIP FROM static from the base photo.
4. Keep the barcode static from the base photo.
5. Warp the edited label back onto the package.
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

# Label corners in the MECHMAXX base photo:
# top-left, top-right, bottom-right, bottom-left
LABEL_CORNERS = np.array([
    [388, 437],
    [666, 362],
    [913, 528],
    [641, 676],
], dtype=np.float32)

LABEL_W = 1000
LABEL_H = 650

# Flat-label edit zones
SHIP_TO_RECT = (450, 165, 970, 440)
TRACKING_TEXT_RECT = (105, 595, 700, 645)

# Barcode area copied back from the original base photo at the end
BARCODE_LOCK_RECT = (35, 510, 690, 595)

FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 30)

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
    line_h: int
) -> int:
    x, y = xy
    words = text.split(" ")
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


def _sample_paper_color(flat_label_bgr: np.ndarray) -> tuple[int, int, int]:
    sample_points = [
        (910, 95),
        (890, 455),
        (760, 430),
        (930, 615),
        (830, 120),
        (700, 580),
    ]

    samples = []

    for x, y in sample_points:
        if 0 <= x < LABEL_W and 0 <= y < LABEL_H:
            b, g, r = flat_label_bgr[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (226, 228, 230)

    arr = np.array(samples, dtype=np.float32)
    median = np.median(arr, axis=0)

    gray = median.mean()
    final = median * 0.85 + gray * 0.15

    return tuple(int(v) for v in final)


def _make_paper_patch(
    size: tuple[int, int],
    paper_rgb: tuple[int, int, int],
    seed: int
) -> Image.Image:
    w, h = size
    rng = np.random.default_rng(seed)

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = paper_rgb[0]
    arr[:, :, 1] = paper_rgb[1]
    arr[:, :, 2] = paper_rgb[2]

    noise = rng.normal(0, 1.1, (h, w, 1))
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGB")


def _paste_patch(
    pil_img: Image.Image,
    rect: tuple[int, int, int, int],
    paper_rgb: tuple[int, int, int],
    seed: int
):
    x0, y0, x1, y1 = rect
    patch = _make_paper_patch((x1 - x0, y1 - y0), paper_rgb, seed)
    pil_img.paste(patch, (x0, y0))


# ---------------------------------------------------------------------------
# FLAT LABEL EDITING
# ---------------------------------------------------------------------------

def flatten_label(base_bgr: np.ndarray) -> np.ndarray:
    M = _image_to_label_matrix()

    flat = cv2.warpPerspective(
        base_bgr,
        M,
        (LABEL_W, LABEL_H),
        flags=cv2.INTER_CUBIC
    )

    return flat


def edit_flat_label(
    flat_label_bgr: np.ndarray,
    recipient: dict,
    tracking_number: str
) -> np.ndarray:
    paper_rgb = _sample_paper_color(flat_label_bgr)

    flat_rgb = cv2.cvtColor(flat_label_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(flat_rgb)
    draw = ImageDraw.Draw(pil)

    # Fonts tuned to match the original generated label
    f_head = _load_font(FONT_BOLD, 32)
    f_text = _load_font(FONT_BOLD, 30)
    f_text_small = _load_font(FONT_BOLD, 26)
    f_tracking = _load_font(FONT_BOLD, 25)

    # -----------------------------------------------------------------------
    # SHIP TO block
    # -----------------------------------------------------------------------

    _paste_patch(pil, SHIP_TO_RECT, paper_rgb, seed=11)

    tx = 490
    ty = 220
    max_w = 420

    draw.text(
        (tx, ty),
        "SHIP TO:",
        font=f_head,
        fill=TEXT_COLOR
    )

    ty += 48

    name = recipient.get("name", "").strip()
    line1 = recipient.get("line1", "").strip()
    line2 = recipient.get("line2", "").strip()
    line3 = recipient.get("line3", "").strip()
    line4 = recipient.get("line4", "").strip()
    phone = recipient.get("phone", "").strip()

    fields = []

    if name:
        fields.append(name)

    if line1:
        fields.append(line1)

    # Important: city/state + postal stays on ONE line
    # Example: Millington, TN 38053
    if line2 and line3:
        fields.append(f"{line2} {line3}")
    elif line2:
        fields.append(line2)
    elif line3:
        fields.append(line3)

    # Important: country goes DIRECTLY under address, no empty space
    if line4:
        fields.append(line4)

    if phone:
        fields.append(phone)

    total_chars = sum(len(x) for x in fields)

    if total_chars > 95 or len(fields) > 5:
        font_use = f_text_small
        first_font = f_text_small
        line_h = 34
    else:
        font_use = f_text
        first_font = f_text
        line_h = 40

    for i, line in enumerate(fields):
        font = first_font if i == 0 else font_use
        ty = _draw_wrapped(
            draw,
            (tx, ty),
            line,
            font,
            TEXT_COLOR,
            max_w,
            line_h
        )

    # -----------------------------------------------------------------------
    # Tracking number text only
    # -----------------------------------------------------------------------

    _paste_patch(pil, TRACKING_TEXT_RECT, paper_rgb, seed=12)

    trk = normalize_tracking(tracking_number)

    x0, y0, x1, y1 = TRACKING_TEXT_RECT
    text_w = draw.textlength(trk, font=f_tracking)

    draw.text(
        (x0 + ((x1 - x0) - text_w) / 2, y0 + 8),
        trk,
        font=f_tracking,
        fill=TEXT_COLOR
    )

    edited_rgb = np.array(pil)
    edited_bgr = cv2.cvtColor(edited_rgb, cv2.COLOR_RGB2BGR)

    return edited_bgr


# ---------------------------------------------------------------------------
# COMPOSITING
# ---------------------------------------------------------------------------

def warp_flat_label_back(
    edited_flat_bgr: np.ndarray,
    base_bgr: np.ndarray
) -> np.ndarray:
    h, w = base_bgr.shape[:2]

    M = _label_to_image_matrix()

    warped_label = cv2.warpPerspective(
        edited_flat_bgr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )

    flat_mask = np.ones((LABEL_H, LABEL_W), dtype=np.uint8) * 255

    warped_mask = cv2.warpPerspective(
        flat_mask,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    warped_mask = cv2.GaussianBlur(warped_mask, (5, 5), 1.0)
    alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]

    out = base_bgr.astype(np.float32) * (1 - alpha) + warped_label.astype(np.float32) * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    if not APPLY_REALISM_PASS:
        return img_bgr

    h, w = img_bgr.shape[:2]

    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.16)

    img_f = img.astype(np.float32)
    img_f = img_f * 0.985 + 128 * 0.015
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    small = cv2.resize(
        img,
        (int(w * 0.985), int(h * 0.985)),
        interpolation=cv2.INTER_AREA
    )

    img = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 88]
    )

    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def paste_locked_barcode_last(
    result_bgr: np.ndarray,
    base_bgr: np.ndarray
) -> np.ndarray:
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
        M
    ).reshape(-1, 2).astype(np.int32)

    mask = np.zeros(result_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, img_pts, 255)

    out = result_bgr.copy()
    out[mask == 255] = base_bgr[mask == 255]

    return out


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_label(
    recipient: dict,
    tracking_number: str | None = None,
    out_path: str | None = None
):
    base_bgr = cv2.imread(BASE_IMAGE_PATH)

    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    flat_label = flatten_label(base_bgr)

    edited_flat_label = edit_flat_label(
        flat_label,
        recipient,
        tracking_number or ""
    )

    result_bgr = warp_flat_label_back(
        edited_flat_label,
        base_bgr
    )

    # Apply realism before restoring barcode
    result_bgr = apply_photo_realism(result_bgr)

    # Restore barcode from original base photo as final operation
    result_bgr = paste_locked_barcode_last(
        result_bgr,
        base_bgr
    )

    result_rgb = cv2.cvtColor(
        result_bgr,
        cv2.COLOR_BGR2RGB
    )

    result_img = Image.fromarray(result_rgb)

    if out_path:
        ext = os.path.splitext(out_path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            result_img.save(out_path, quality=95, subsampling=0)
        else:
            result_img.save(out_path)

    return result_img


if __name__ == "__main__":
    img = generate_label(
        {
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
