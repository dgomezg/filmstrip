import os
import sys
import argparse
from typing import List, Dict, Tuple

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin, ExifTags

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_STRIP_WIDTH = 1500          # Base strip width (used in vertical mode)
DEFAULT_STRIP_HEIGHT = 1500         # Reserved if you want special horizontal sizing logic
DEFAULT_FRAME_GAP = 100             # Gap between frames
DEFAULT_HOLE_WIDTH = 70             # Perforation width
DEFAULT_HOLE_HEIGHT = 90            # Perforation height
DEFAULT_PERF_SPACING = 140          # Distance between perforations
DEFAULT_TEXT_COLOR = (255, 215, 0, 255)  # "Kodak Gold" yellow
DEFAULT_FONT_SIZE = 48
DEFAULT_TEXT_BAND_RATIO = 0.4       # % of the band width used for text
DEFAULT_FRAMES_PER_STRIP = 4        # Used in contact-sheet mode
DEFAULT_FONT_PATH = ""              # Optional TTF font path
DEFAULT_EXIF_FONT_SIZE = 24
DEFAULT_EXIF_TEXT_COLOR = (255, 215, 0, 255)
DEFAULT_EXIF_MAX_LINES = 2
DEFAULT_EXIF_PADDING = 8


# ============================================================
# UTILITIES
# ============================================================

def get_resample_filter():
    """Return a LANCZOS resampling filter compatible across Pillow versions."""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TTF font or fall back to Pillow's default font."""
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def extract_frame_suffix(filename: str) -> str:
    """Extract the trailing suffix from the filename.

    Example: 'DVG-250501-0087.jpg' -> '0087'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("-")[-1]


def parse_meta(meta_list: List[str]) -> Dict[str, str]:
    """Convert a list of 'key=value' entries into a dictionary.

    Example: ["Author=David", "Film=Kodak"] -> {"Author": "David", "Film": "Kodak"}
    """
    result: Dict[str, str] = {}
    for item in meta_list:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()
    return result


# ============================================================
# EXIF HELPERS
# ============================================================

def _exif_get(exif, key: str):
    """Safely get an EXIF value by tag name."""
    if not exif:
        return None
    try:
        return exif.get(key)
    except Exception:
        return None


def _format_exif_value(value) -> str:
    if value is None:
        return ""
    # Convert rationals like (num, den) to float
    try:
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            num, den = value
            if den:
                return str(num / den)
    except Exception:
        pass
    return str(value)


def _format_exposure_time(value) -> str:
    """Format exposure time like 1/200 when possible."""
    if value is None:
        return ""
    try:
        # Pillow may provide a Rational or fraction-like
        if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
            num = value.numerator
            den = value.denominator
            if den:
                if num == 1:
                    return f"1/{den}"
                return f"{num}/{den}"
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            if den:
                if num == 1:
                    return f"1/{den}"
                return f"{num}/{den}"
        # float seconds
        v = float(value)
        if v >= 1:
            # 1.3s
            return f"{v:.1f}s"
        # 0.005 -> 1/200
        inv = round(1 / v)
        return f"1/{inv}"
    except Exception:
        return str(value)


def format_exif_line(image_path: str) -> str:
    """Build a compact EXIF summary line for display under the frame."""
    try:
        im = Image.open(image_path)
        exif = im.getexif()
        if not exif:
            return ""

        # Map numeric EXIF tags to readable names
        named = {}
        for k, v in exif.items():
            name = ExifTags.TAGS.get(k, k)
            named[name] = v

        make = _format_exif_value(named.get("Make")).strip()
        model = _format_exif_value(named.get("Model")).strip()

        focal = named.get("FocalLength")
        focal_s = ""
        try:
            if hasattr(focal, 'numerator') and hasattr(focal, 'denominator') and focal.denominator:
                focal_s = f"{round(focal.numerator / focal.denominator)}mm"
            elif isinstance(focal, tuple) and len(focal) == 2 and focal[1]:
                focal_s = f"{round(focal[0] / focal[1])}mm"
            elif focal is not None:
                focal_s = f"{round(float(focal))}mm"
        except Exception:
            focal_s = ""

        fnumber = named.get("FNumber")
        f_s = ""
        try:
            if hasattr(fnumber, 'numerator') and hasattr(fnumber, 'denominator') and fnumber.denominator:
                f_s = f"f/{(fnumber.numerator / fnumber.denominator):.1f}"
            elif isinstance(fnumber, tuple) and len(fnumber) == 2 and fnumber[1]:
                f_s = f"f/{(fnumber[0] / fnumber[1]):.1f}"
            elif fnumber is not None:
                f_s = f"f/{float(fnumber):.1f}"
        except Exception:
            f_s = ""

        exposure = _format_exposure_time(named.get("ExposureTime"))

        iso = named.get("ISOSpeedRatings")
        iso_s = ""
        if iso is not None:
            try:
                if isinstance(iso, (list, tuple)) and iso:
                    iso_s = f"ISO {iso[0]}"
                else:
                    iso_s = f"ISO {int(iso)}"
            except Exception:
                iso_s = f"ISO {iso}"

        # Compose camera label
        camera = " ".join([x for x in [make, model] if x]).strip()

        parts = [p for p in [camera, focal_s, f_s, exposure, iso_s] if p]
        return " | ".join(parts)
    except Exception:
        return ""


def wrap_text_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int):
    """Wrap text into up to max_lines so each line fits max_width."""
    if not text:
        return []

    words = text.split()
    lines: List[str] = []
    current = ""

    def text_width(s: str) -> int:
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    for w in words:
        test = (current + " " + w).strip()
        if not current:
            current = w
            continue
        if text_width(test) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
            if len(lines) >= max_lines - 1:
                break

    if len(lines) < max_lines and current:
        lines.append(current)

    # If still too wide, hard-trim last line
    if lines:
        while text_width(lines[-1]) > max_width and len(lines[-1]) > 3:
            lines[-1] = lines[-1][:-4].rstrip() + "…"

    return lines[:max_lines]


def make_vertical_text_scaled(
    text: str,
    font: ImageFont.FreeTypeFont,
    target_width: int,
    text_color: Tuple[int, int, int, int] = DEFAULT_TEXT_COLOR,
) -> Image.Image:
    """Create a vertical text image by drawing horizontally, rotating 90°, and scaling."""
    # Large temporary canvas
    temp = Image.new("RGBA", (2000, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    draw.text((20, 20), text, fill=text_color, font=font)

    bbox = temp.getbbox()
    if bbox:
        temp = temp.crop(bbox)

    rotated = temp.rotate(90, expand=True)
    w, h = rotated.size
    if w == 0:
        return rotated

    scale = target_width / w
    new_h = int(h * scale)
    resample = get_resample_filter()
    return rotated.resize((target_width, new_h), resample)


# ============================================================
# VERTICAL STRIP GENERATOR
# ============================================================

def build_vertical_strip(
    image_files: List[str],
    roll_code: str,
    strip_width: int = DEFAULT_STRIP_WIDTH,
    frame_gap: int = DEFAULT_FRAME_GAP,
    hole_w: int = DEFAULT_HOLE_WIDTH,
    hole_h: int = DEFAULT_HOLE_HEIGHT,
    perf_spacing: int = DEFAULT_PERF_SPACING,
    band_ratio: float = DEFAULT_TEXT_BAND_RATIO,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = DEFAULT_FONT_SIZE,
    emulsion: str = "GOLD",
    iso: int | None = None,
    exif_under_frame: bool = False,
) -> Image.Image:
    # Sort images
    image_files = sorted(image_files)

    # Load images + suffixes + exif lines
    images: List[Image.Image] = []
    suffixes: List[str] = []
    exif_lines: List[str] = []
    for path in image_files:
        im = Image.open(path).convert("RGB")
        images.append(im)
        suffixes.append(extract_frame_suffix(path))
        exif_lines.append(format_exif_line(path))

    # Horizontal geometry
    left_margin = 40
    right_margin = 40
    strip_w = strip_width
    band_w = hole_w  # Text band width equals perforation width

    left_hole_x0 = left_margin
    left_hole_x1 = left_hole_x0 + hole_w

    left_band_x0 = left_hole_x1
    left_band_x1 = left_band_x0 + band_w

    right_hole_x1 = strip_w - right_margin
    right_hole_x0 = right_hole_x1 - hole_w

    right_band_x1 = right_hole_x0
    right_band_x0 = right_band_x1 - band_w

    # Inner width for photos
    inner_w = strip_w - (left_margin + hole_w + band_w + band_w + hole_w + right_margin)
    photo_x = left_band_x1

    # Resize photos to the same width
    resized: List[Image.Image] = []
    heights: List[int] = []
    resample = get_resample_filter()
    for im in images:
        w, h = im.size
        new_h = int(h * (inner_w / w))
        resized.append(im.resize((inner_w, new_h), resample))
        heights.append(new_h)

    # EXIF text height reservation (optional)
    exif_font = load_font(font_path, DEFAULT_EXIF_FONT_SIZE)
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
    line_bbox = dummy_draw.textbbox((0, 0), "Ag", font=exif_font)
    line_h = (line_bbox[3] - line_bbox[1])

    exif_extra_heights: List[int] = []
    if exif_under_frame:
        for exif_text in exif_lines:
            if exif_text:
                # Up to DEFAULT_EXIF_MAX_LINES lines
                lines = wrap_text_lines(dummy_draw, exif_text, exif_font, inner_w, DEFAULT_EXIF_MAX_LINES)
                exif_extra_heights.append(len(lines) * line_h + DEFAULT_EXIF_PADDING)
            else:
                exif_extra_heights.append(0)
    else:
        exif_extra_heights = [0] * len(resized)

    total_h = sum(h + eh for h, eh in zip(heights, exif_extra_heights)) + frame_gap * (len(resized) + 1)

    # Base canvas
    strip = Image.new("RGBA", (strip_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(strip)

    # Side perforations
    num_holes = max(1, total_h // perf_spacing)
    for i in range(num_holes):
        cy = int(i * perf_spacing + perf_spacing / 2)
        y0 = cy - hole_h // 2
        y1 = cy + hole_h // 2

        draw.rounded_rectangle([left_hole_x0, y0, left_hole_x1, y1], radius=15, fill=(0, 0, 0, 0))
        draw.rounded_rectangle([right_hole_x0, y0, right_hole_x1, y1], radius=15, fill=(0, 0, 0, 0))

    # Text
    font = load_font(font_path, font_size)
    target_width = int(band_w * band_ratio)

    y = frame_gap
    for index, im in enumerate(resized):
        h = im.size[1]
        extra_h = exif_extra_heights[index] if exif_under_frame else 0
        suffix = suffixes[index]

        # Black frame window (includes optional EXIF area below the photo)
        draw.rectangle([photo_x - 10, y - 10, photo_x + inner_w + 10, y + h + extra_h + 10], fill=(0, 0, 0, 255))

        strip.paste(im, (photo_x, y))
        cy = y + h // 2

        # LEFT — "KODAK GOLD"
        left_label = f"KODAK {emulsion}".strip()
        if iso is not None:
            left_label = f"{left_label} {iso}".strip()
        left_text = make_vertical_text_scaled(left_label, font, target_width)
        lx = (left_band_x0 + left_band_x1) // 2 - left_text.size[0] // 2
        ly = cy - left_text.size[1] // 2
        strip.paste(left_text, (lx, ly), left_text)

        # RIGHT — suffix
        right_text = make_vertical_text_scaled(suffix, font, target_width)
        rx = (right_band_x0 + right_band_x1) // 2 - right_text.size[0] // 2
        ry = cy - right_text.size[1] // 2
        strip.paste(right_text, (rx, ry), right_text)

        # BETWEEN FRAMES — roll code
        if index < len(resized) - 1:
            gap_center = y + h + extra_h + frame_gap // 2
            gap_text = make_vertical_text_scaled(roll_code, font, target_width)
            gx = (right_band_x0 + right_band_x1) // 2 - gap_text.size[0] // 2
            gy = gap_center - gap_text.size[1] // 2
            strip.paste(gap_text, (gx, gy), gap_text)

        # EXIF info under the frame
        if exif_under_frame and exif_lines[index]:
            lines = wrap_text_lines(draw, exif_lines[index], exif_font, inner_w, DEFAULT_EXIF_MAX_LINES)
            text_y = y + h + (DEFAULT_EXIF_PADDING // 2)
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=exif_font)
                line_w = bbox[2] - bbox[0]
                line_h_local = bbox[3] - bbox[1]
                text_x = photo_x + (inner_w // 2) - (line_w // 2)
                draw.text((text_x, text_y), line, font=exif_font, fill=DEFAULT_EXIF_TEXT_COLOR)
                text_y += line_h_local

        y += h + extra_h + frame_gap

    return strip


def build_horizontal_strip_upright(
    image_files: List[str],
    roll_code: str,
    strip_width: int = DEFAULT_STRIP_WIDTH,
    frame_gap: int = DEFAULT_FRAME_GAP,
    hole_w: int = DEFAULT_HOLE_WIDTH,
    hole_h: int = DEFAULT_HOLE_HEIGHT,
    perf_spacing: int = DEFAULT_PERF_SPACING,
    band_ratio: float = DEFAULT_TEXT_BAND_RATIO,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = DEFAULT_FONT_SIZE,
    emulsion: str = "GOLD",
    iso: int | None = None,
    exif_under_frame: bool = False,
) -> Image.Image:
    """Horizontal orientation:

    - Photos remain upright for the viewer.
    - Trick: rotate photos 90º CCW before layout, then rotate the full strip 90º CW at the end.
      Net photo rotation: -90º + 90º = 0º (upright).
    - Perforations and text bands end up horizontal.

    Ordering:
    - We iterate images in reverse so that, after the final strip rotation, frames read left-to-right.
    """

    image_files = sorted(image_files)

    # Load images rotated 90º CCW (positive angle in Pillow) + exif lines
    images: List[Image.Image] = []
    suffixes: List[str] = []
    exif_lines: List[str] = []
    for path in reversed(image_files):
        im = Image.open(path).convert("RGB")
        im = im.rotate(90, expand=True)  # CCW
        images.append(im)
        suffixes.append(extract_frame_suffix(path))
        exif_lines.append(format_exif_line(path))

    # Geometry (same as vertical during layout)
    left_margin = 40
    right_margin = 40
    strip_w = strip_width
    band_w = hole_w

    left_hole_x0 = left_margin
    left_hole_x1 = left_hole_x0 + hole_w

    left_band_x0 = left_hole_x1
    left_band_x1 = left_band_x0 + band_w

    right_hole_x1 = strip_w - right_margin
    right_hole_x0 = right_hole_x1 - hole_w

    right_band_x1 = right_hole_x0
    right_band_x0 = right_band_x1 - band_w

    inner_w = strip_w - (left_margin + hole_w + band_w + band_w + hole_w + right_margin)
    photo_x = left_band_x1

    # Resize photos to the same width
    resized: List[Image.Image] = []
    heights: List[int] = []
    resample = get_resample_filter()
    for im in images:
        w, h = im.size
        new_h = int(h * (inner_w / w))
        resized.append(im.resize((inner_w, new_h), resample))
        heights.append(new_h)

    # EXIF text height reservation (optional)
    exif_font = load_font(font_path, DEFAULT_EXIF_FONT_SIZE)
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
    line_bbox = dummy_draw.textbbox((0, 0), "Ag", font=exif_font)
    line_h = (line_bbox[3] - line_bbox[1])

    exif_extra_heights: List[int] = []
    if exif_under_frame:
        for exif_text in exif_lines:
            if exif_text:
                # Up to DEFAULT_EXIF_MAX_LINES lines
                lines = wrap_text_lines(dummy_draw, exif_text, exif_font, inner_w, DEFAULT_EXIF_MAX_LINES)
                exif_extra_heights.append(len(lines) * line_h + DEFAULT_EXIF_PADDING)
            else:
                exif_extra_heights.append(0)
    else:
        exif_extra_heights = [0] * len(resized)

    total_h = sum(h + eh for h, eh in zip(heights, exif_extra_heights)) + frame_gap * (len(resized) + 1)

    # Base canvas
    strip = Image.new("RGBA", (strip_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(strip)

    # Perforations (side at this stage; will end up top/bottom after rotation)
    num_holes = max(1, total_h // perf_spacing)
    for i in range(num_holes):
        cy = int(i * perf_spacing + perf_spacing / 2)
        y0 = cy - hole_h // 2
        y1 = cy + hole_h // 2

        draw.rounded_rectangle([left_hole_x0, y0, left_hole_x1, y1], radius=15, fill=(0, 0, 0, 0))
        draw.rounded_rectangle([right_hole_x0, y0, right_hole_x1, y1], radius=15, fill=(0, 0, 0, 0))

    font = load_font(font_path, font_size)
    target_width = int(band_w * band_ratio)

    y = frame_gap
    for index, im in enumerate(resized):
        h = im.size[1]
        extra_h = exif_extra_heights[index] if exif_under_frame else 0
        suffix = suffixes[index]

        # Black frame window (includes optional EXIF area below the photo)
        draw.rectangle([photo_x - 10, y - 10, photo_x + inner_w + 10, y + h + extra_h + 10], fill=(0, 0, 0, 255))

        strip.paste(im, (photo_x, y))
        cy = y + h // 2

        # LEFT — "KODAK GOLD"
        left_label = f"KODAK {emulsion}".strip()
        if iso is not None:
            left_label = f"{left_label} {iso}".strip()
        left_text = make_vertical_text_scaled(left_label, font, target_width)
        lx = (left_band_x0 + left_band_x1) // 2 - left_text.size[0] // 2
        ly = cy - left_text.size[1] // 2
        strip.paste(left_text, (lx, ly), left_text)

        # RIGHT — suffix
        right_text = make_vertical_text_scaled(suffix, font, target_width)
        rx = (right_band_x0 + right_band_x1) // 2 - right_text.size[0] // 2
        ry = cy - right_text.size[1] // 2
        strip.paste(right_text, (rx, ry), right_text)

        # BETWEEN FRAMES — roll code
        if index < len(resized) - 1:
            gap_center = y + h + extra_h + frame_gap // 2
            gap_text = make_vertical_text_scaled(roll_code, font, target_width)
            gx = (right_band_x0 + right_band_x1) // 2 - gap_text.size[0] // 2
            gy = gap_center - gap_text.size[1] // 2
            strip.paste(gap_text, (gx, gy), gap_text)

        # EXIF info under the frame
        if exif_under_frame and exif_lines[index]:
            lines = wrap_text_lines(draw, exif_lines[index], exif_font, inner_w, DEFAULT_EXIF_MAX_LINES)
            text_y = y + h + (DEFAULT_EXIF_PADDING // 2)
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=exif_font)
                line_w = bbox[2] - bbox[0]
                line_h_local = bbox[3] - bbox[1]
                text_x = photo_x + (inner_w // 2) - (line_w // 2)
                draw.text((text_x, text_y), line, font=exif_font, fill=DEFAULT_EXIF_TEXT_COLOR)
                text_y += line_h_local

        y += h + extra_h + frame_gap

    # Rotate the whole strip 90º CW so it becomes horizontal.
    # Photos remain upright because we rotated them 90º CCW before layout.
    strip_horizontal = strip.rotate(-90, expand=True)
    return strip_horizontal


# ============================================================
# ORIENTATION WRAPPER (VERTICAL / HORIZONTAL)
# ============================================================

def build_strip_with_orientation(
    image_files: List[str],
    roll_code: str,
    orientation: str = "vertical",
    **kwargs,
) -> Image.Image:
    """Dispatch to the correct builder based on orientation."""
    if orientation.lower() == "horizontal":
        return build_horizontal_strip_upright(image_files, roll_code, **kwargs)
    return build_vertical_strip(image_files, roll_code, **kwargs)


# ============================================================
# CONTACT SHEET (MULTIPLE STRIPS)
# ============================================================

def chunk_list(items: List[str], size: int) -> List[List[str]]:
    """Split a list into chunks of size `size`."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_contact_sheet(
    image_files: List[str],
    roll_code: str,
    frames_per_strip: int = DEFAULT_FRAMES_PER_STRIP,
    orientation: str = "vertical",
    strip_width: int = DEFAULT_STRIP_WIDTH,
    **kwargs,
) -> Image.Image:
    """Split photos into multiple strips and merge them into one image.

    - vertical: strips side-by-side (columns)
    - horizontal: strips stacked (rows)
    """
    image_files = sorted(image_files)
    groups = chunk_list(image_files, frames_per_strip)

    strips: List[Image.Image] = []
    for group in groups:
        strip = build_strip_with_orientation(group, roll_code, orientation=orientation, strip_width=strip_width, **kwargs)
        strips.append(strip)

    if not strips:
        raise ValueError("No strip was generated.")

    if orientation.lower() == "vertical":
        total_w = sum(s.width for s in strips) + 40 * (len(strips) - 1)
        max_h = max(s.height for s in strips)
        sheet = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))

        x = 0
        for s in strips:
            sheet.paste(s, (x, 0), s)
            x += s.width + 40
    else:
        max_w = max(s.width for s in strips)
        total_h = sum(s.height for s in strips) + 40 * (len(strips) - 1)
        sheet = Image.new("RGBA", (max_w, total_h), (0, 0, 0, 0))

        y = 0
        for s in strips:
            sheet.paste(s, (0, y), s)
            y += s.height + 40

    return sheet


# ============================================================
# OUTPUT (METADATA + OPTIONAL PDF)
# ============================================================

def save_with_metadata(img: Image.Image, path: str, meta: Dict[str, str]):
    """Save an image file; if PNG, embed tEXt metadata."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(k, str(v))
        img.save(path, "PNG", pnginfo=pnginfo)
    else:
        img.save(path)


def save_pdf(img: Image.Image, pdf_path: str):
    """Save a single-page PDF version of the generated image.

    Note: PDF doesn't preserve alpha like PNG. We flatten transparency onto white.
    """
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        pdf_img = background
    else:
        pdf_img = img.convert("RGB")

    pdf_img.save(pdf_path, "PDF")


def save_output(img: Image.Image, output_path: str, meta: Dict[str, str], export_pdf: bool):
    """Save the output image and optionally also export a PDF copy."""
    save_with_metadata(img, output_path, meta)

    if export_pdf:
        base, _ext = os.path.splitext(output_path)
        pdf_path = base + ".pdf"
        save_pdf(img, pdf_path)
        print("PDF generated at:", pdf_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Kodak Gold film strip generator")

    parser.add_argument("input_dir", help="Folder containing input images")
    parser.add_argument("roll_code", help="Roll code printed between frames (e.g. DvG-250501)")
    parser.add_argument("output", help="Output file (PNG recommended)")

    parser.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--mode", choices=["strip", "contact"], default="strip")
    parser.add_argument("--frames-per-strip", type=int, default=DEFAULT_FRAMES_PER_STRIP)
    parser.add_argument("--strip-width", type=int, default=DEFAULT_STRIP_WIDTH)
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP)
    parser.add_argument("--font-path", default=DEFAULT_FONT_PATH)
    parser.add_argument("--meta", action="append", default=[], help="Add metadata: --meta key=value")
    parser.add_argument("--emulsion", default="GOLD",
                    help="Film emulsion name to print (e.g. GOLD, PORTRA, EKTAR)")
    parser.add_argument("--iso", type=int, default=None,
                    help="ISO speed to print next to the emulsion (e.g. 200)")
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Export an additional PDF copy next to the output image (same name, .pdf)",
    )
    parser.add_argument(
        "--exif-under-frame",
        action="store_true",
        help="Render a compact EXIF summary under each frame (camera, focal length, f-stop, shutter, ISO)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print("Input folder not found:", args.input_dir)
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png")
    image_files = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.lower().endswith(exts)
    ]

    if not image_files:
        print("No images found.")
        sys.exit(1)

    meta = parse_meta(args.meta)
    meta.setdefault("Emulsion", str(args.emulsion))
    if args.iso is not None:
        meta.setdefault("ISO", str(args.iso))

    # Common kwargs passed to the strip builders.
    # NOTE: Do not include strip_width here to avoid passing it twice in contact mode.
    common_kwargs = dict(
        frame_gap=args.frame_gap,
        font_path=args.font_path,
        emulsion=args.emulsion,
        iso=args.iso,
        exif_under_frame=args.exif_under_frame,
    )

    if args.mode == "strip":
        img = build_strip_with_orientation(
            image_files,
            args.roll_code,
            orientation=args.orientation,
            strip_width=args.strip_width,
            **common_kwargs,
        )
    else:
        img = build_contact_sheet(
            image_files,
            args.roll_code,
            frames_per_strip=args.frames_per_strip,
            orientation=args.orientation,
            strip_width=args.strip_width,
            **common_kwargs,
        )

    save_output(img, args.output, meta, export_pdf=args.export_pdf)
    print("Generated at:", args.output)


if __name__ == "__main__":
    main()