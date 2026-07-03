"""
MECHMAXX / TrackFlow shipping-label image compositor.

This version is adapted for the MECHMAXX warehouse base photo.
It does NOT redraw the full label. It patches only:
- SHIP TO address block
- tracking number text under the barcode

The SHIP FROM block remains fixed in the base photo.
The barcode remains static by copying the barcode pixels from the base photo
as the final operation.
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

# Label corners in the MECHMAXX base photo, in image pixel coordinates:
# top-left, top-right, bottom-right, bottom-left
LABEL_CORNERS = np.array([
    [388, 437],
    [666, 362],
    [913, 528],
    [641, 676],
], dtype=np.float32)

# Flat label coordinate system used internally
LABEL_W = 1000
LABEL_H = 650

# Editable regions in flat-label coordinates
# Only the dynamic parts are patched
SHIP_TO_RECT = (455, 160, 960, 395)          # x0, y0, x1, y1
TRACKING_TEXT_RECT = (160, 590, 660, 645)    # tracking text below barcode only

# Locked barcode area in flat-label coordinates
# Copied from base photo as the FINAL step
BARCODE_LOCK_RECT = (25, 515, 660, 590)      # barcode bars only, not tracking text

# Font paths
FONT_DIR = "/usr/share/fonts/truetype/liberation"
FONT_BOLD = f"{FONT_DIR}/LiberationSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/LiberationSans-Regular.ttf"

TEXT_COLOR = (25, 25, 30)
APPLY_REALISM_PASS = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_font(path: str, size: int):
    """Load a TTF font with fallbacks."""
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


def _sample_paper_color(base_bgr: np.ndarray) -> tuple[int, int, int]:
    """Sample the existing label paper color from clean areas of the base image."""
    M = _image_to_label_matrix()
    flat = cv2.warpPerspective(base_bgr, M, (LABEL_W, LABEL_H), flags=cv2.INTER_CUBIC)

    sample_points = [
        (920, 90), (880, 430), (760, 395), (700, 120),
        (930, 610), (850, 500), (370, 110),
    ]

    samples = []
    for x, y in sample_points:
        if 0 <= x < LABEL_W and 0 <= y < LABEL_H:
            b, g, r = flat[y, x]
            samples.append((int(r), int(g), int(b)))

    if not samples:
        return (226, 228, 230)

    arr = np.array(samples, dtype=np.float32)
    median = np.median(arr, axis=0)

    gray = median.mean()
    final = median * 0.82 + gray * 0.18
    return tuple(int(v) for v in final)


def _paper_patch(size: tuple[int, int], paper_rgb: tuple[int, int, int], seed: int = 9) -> Image.Image:
    """Create a subtle photographed-paper patch with tiny texture and soft edges."""
    w, h = size
    rng = np.random.default_rng(seed)

    base = np.zeros((h, w, 4), dtype=np.uint8)
    base[:, :, 0] = paper_rgb[0]
    base[:, :, 1] = paper_rgb[1]
    base[:, :, 2] = paper_rgb[2]

    # subtle photographed paper noise
    noise = rng.normal(0, 1.2, (h, w, 1))
    rgb = np.clip(base[:, :, :3].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    base[:, :, :3] = rgb

    # soft alpha edges so the patch blends better
    alpha = np.ones((h, w), dtype=np.float32) * 255.0
    feather = 7

    for i in range(feather):
        a = 255.0 * ((i + 1) / feather)
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


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill,
    max_width: int,
    line_h: int
) -> int:
    """Draw text with simple word wrapping. Returns final y."""
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


def normalize_tracking(tracking_number: str | None) -> str:
    """
    Normalize to TrackFlow style:
    - "Q8ZH7PDVRH" becomes "TF-Q8ZH7PDVRH"
    - "TF-Q8ZH7PDVRH" stays as-is
    """
    raw = (tracking_number or "").strip().upper()
    raw = re.sub(r"[^A-Z0-9-]", "", raw)

    if raw.startswith("TF-") and len(raw) > 3:
        return raw

    core = re.sub(r"[^A-Z0-9]", "", raw)
    if not core:
        core = "QZ6HTDVPRH"

    return f"TF-{core}"


# ---------------------------------------------------------------------------
# Overlay creation
# ---------------------------------------------------------------------------

def build_label_overlay(
    recipient: dict,
    tracking_number: str,
    paper_rgb: tuple[int, int, int]
) -> Image.Image:
    """Build a transparent flat-label overlay containing only patched fields."""
    overlay = Image.new("RGBA", (LABEL_W, LABEL_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    f_head = _load_font(FONT_BOLD, 24)
    f_text = _load_font(FONT_REG, 22)
    f_text_bold = _load_font(FONT_BOLD, 22)
    f_tracking = _load_font(FONT_BOLD, 22)

    # ------------------ Patch and redraw SHIP TO ------------------
    _paste_paper_rect(overlay, SHIP_TO_RECT, paper_rgb, seed=11)

    tx = 500
    ty = 185
    max_w = 410

    draw.text((tx, ty), "SHIP TO:", font=f_head, fill=TEXT_COLOR + (255,))
    ty += 34

    fields = [
        recipient.get("name", ""),
        recipient.get("line1", ""),
    ]

    postal_code = recipient.get("line3", "")
    city = recipient.get("line2", "")

    if postal_code and city:
        fields.append(f"{city}, {postal_code}" if "," not in city else f"{city} {postal_code}")
    elif postal_code or city:
        fields.append(postal_code or city)

    country = recipient.get("line4", "")
    if country:
        fields.append(country)

    if recipient.get("phone"):
        fields.append(recipient.get("phone", ""))

    total_chars = sum(len(x) for x in fields)

    if len(fields) > 5 or total_chars > 105:
        f_text_use = _load_font(FONT_REG, 19)
        f_first_use = _load_font(FONT_BOLD, 20)
        line_h = 24
    else:
        f_text_use = f_text
        f_first_use = f_text_bold
        line_h = 27

    for i, line in enumerate(fields):
        if not line:
            continue

        font = f_first_use if i == 0 else f_text_use
        ty = _draw_wrapped(
            draw,
            (tx, ty),
            line,
            font,
            TEXT_COLOR + (255,),
            max_w,
            line_h
        )

    # ------------------ Patch and redraw tracking number text ------------------
    _paste_paper_rect(overlay, TRACKING_TEXT_RECT, paper_rgb, seed=12)

    trk = normalize_tracking(tracking_number)
    x0, y0, x1, y1 = TRACKING_TEXT_RECT
    text_w = draw.textlength(trk, font=f_tracking)

    draw.text(
        (x0 + ((x1 - x0) - text_w) / 2, y0 + 5),
        trk,
        font=f_tracking,
        fill=TEXT_COLOR + (255,)
    )

    return overlay


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def warp_overlay_to_image(
    overlay_rgba: Image.Image,
    base_shape: tuple[int, int, int]
) -> np.ndarray:
    """Warp flat-label RGBA overlay to base-image perspective."""
    h, w = base_shape[:2]
    overlay_np = np.array(overlay_rgba)

    M = _label_to_image_matrix()

    warped = cv2.warpPerspective(
        overlay_np,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT
    )

    # tiny blur so patched text is not too digitally crisp
    warped = cv2.GaussianBlur(warped, (3, 3), 0.30)

    return warped


def alpha_composite_bgr(base_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Composite RGBA overlay onto BGR image."""
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]

    overlay_bgr = cv2.cvtColor(
        rgb.astype(np.uint8),
        cv2.COLOR_RGB2BGR
    ).astype(np.float32)

    out = base_bgr.astype(np.float32) * (1 - alpha) + overlay_bgr * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_photo_realism(img_bgr: np.ndarray) -> np.ndarray:
    """Make final output look like a slightly low-quality warehouse phone photo."""
    if not APPLY_REALISM_PASS:
        return img_bgr

    h, w = img_bgr.shape[:2]

    # mild blur, still readable
    img = cv2.GaussianBlur(img_bgr, (3, 3), 0.35)

    # slight flattening of contrast
    img_f = img.astype(np.float32)
    img_f = img_f * 0.985 + 128 * 0.015
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # small downscale/upscale to soften details naturally
    small = cv2.resize(
        img,
        (int(w * 0.96), int(h * 0.96)),
        interpolation=cv2.INTER_AREA
    )
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # JPEG compression makes it more realistic when zoomed in
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if ok:
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return img


def paste_locked_barcode_last(
    result_bgr: np.ndarray,
    base_bgr: np.ndarray
) -> np.ndarray:
    """Copy the original barcode pixels from the base photo as the final operation."""
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_label(
    recipient: dict,
    tracking_number: str | None = None,
    out_path: str | None = None
):
    """
    recipient keys:
        name
        line1 = street
        line2 = city/state
        line3 = postal code
        line4 = country/country code
        phone optional

    tracking_number examples:
        "QZ6HTDVPRH" -> "TF-QZ6HTDVPRH"
        "TF-QZ6HTDVPRH" -> "TF-QZ6HTDVPRH"
    """
    base_bgr = cv2.imread(BASE_IMAGE_PATH)

    if base_bgr is None:
        raise FileNotFoundError(BASE_IMAGE_PATH)

    paper_rgb = _sample_paper_color(base_bgr)

    overlay = build_label_overlay(
        recipient,
        tracking_number or "",
        paper_rgb
    )

    warped_overlay = warp_overlay_to_image(overlay, base_bgr.shape)

    result_bgr = alpha_composite_bgr(base_bgr, warped_overlay)

    # Global realism BEFORE locking barcode
    result_bgr = apply_photo_realism(result_bgr)

    # Final step: locked barcode from base image
    result_bgr = paste_locked_barcode_last(result_bgr, base_bgr)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
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
        },
        tracking_number="QZ6HTDVPRH",
        out_path="test_mechmaxx_output.png",
    )

    print("done", img.size)
