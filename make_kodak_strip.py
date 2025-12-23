import os
import sys
import argparse
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_STRIP_WIDTH = 1500        # Strip width in vertical mode
DEFAULT_STRIP_HEIGHT = 1500       # (Reserved if you want specific horizontal logic later)
DEFAULT_FRAME_GAP = 100           # Gap between frames
DEFAULT_HOLE_WIDTH = 70           # Perforation width
DEFAULT_HOLE_HEIGHT = 90          # Perforation height
DEFAULT_PERF_SPACING = 140        # Spacing between perforations
DEFAULT_TEXT_COLOR = (255, 215, 0, 255)  # "Kodak Gold"
DEFAULT_FONT_SIZE = 48
DEFAULT_TEXT_BAND_RATIO = 0.4     # Percentage of the band width used by text
DEFAULT_FRAMES_PER_STRIP = 4      # Used in contact-sheet mode
DEFAULT_FONT_PATH = ""            # Optional path to a TTF font


# ============================================================
# UTILS
# ============================================================

def get_resample():
    """Returns the LANCZOS resampling filter, compatible with different Pillow versions"""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Loads a TTF font or uses the default font."""
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def extract_suffix(filename: str) -> str:
    """
    Extract the suffix from a filename.
    Example: 'DVG-250501-0087.jpg' -> '0087'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("-")[-1]


def parse_meta(meta_list):
    """
    Converts a 'key=value' list in a dictionary.
    Example: ["Author=David", "Film=Kodak"] → {"Author": "David", "Film": "Kodak"}
    """
    result = {}
    for item in meta_list:
        if "=" in item:
            k, v = item.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def make_vertical_scaled(text: str,
                         font: ImageFont.FreeTypeFont,
                         target_width: int,
                         text_color=DEFAULT_TEXT_COLOR) -> Image.Image:
    """
    Creates an image with horizontal text, rotates it 90º and rescales to target_width.
    """
    # temporal canvas
    tmp = Image.new("RGBA", (2000, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    d.text((20, 20), text, fill=text_color, font=font)

    bbox = tmp.getbbox()
    if bbox:
        tmp = tmp.crop(bbox)

    rot = tmp.rotate(90, expand=True)
    w, h = rot.size
    if w == 0:
        return rot

    scale = target_width / w
    new_h = int(h * scale)
    resample = get_resample()
    return rot.resize((target_width, new_h), resample)


# ============================================================
# VERTICAL STRIP GENERATOR
# ============================================================

def build_vertical_strip(
    image_files,
    roll_code: str,
    strip_width: int = DEFAULT_STRIP_WIDTH,
    frame_gap: int = DEFAULT_FRAME_GAP,
    hole_w: int = DEFAULT_HOLE_WIDTH,
    hole_h: int = DEFAULT_HOLE_HEIGHT,
    perf_spacing: int = DEFAULT_PERF_SPACING,
    band_ratio: float = DEFAULT_TEXT_BAND_RATIO,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = DEFAULT_FONT_SIZE,
) -> Image.Image:

    # Images sorting
    image_files = sorted(image_files)

    # loads images and sufixes
    images = []
    suffixes = []
    for path in image_files:
        im = Image.open(path).convert("RGB")
        images.append(im)
        suffixes.append(extract_suffix(path))

    # Horizontal geometry
    left_margin = 40
    right_margin = 40
    strip_w = strip_width
    band_w = hole_w  # text band width equals perforation width

    left_hole_x0 = left_margin
    left_hole_x1 = left_hole_x0 + hole_w

    left_band_x0 = left_hole_x1
    left_band_x1 = left_band_x0 + band_w

    right_hole_x1 = strip_w - right_margin
    right_hole_x0 = right_hole_x1 - hole_w

    right_band_x1 = right_hole_x0
    right_band_x0 = right_band_x1 - band_w

    # inner width for photos:
    inner_w = strip_w - (left_margin + hole_w + band_w + band_w + hole_w + right_margin)
    photo_x = left_band_x1

    # Resize photos to the same width
    resized = []
    heights = []
    resample = get_resample()
    for im in images:
        w, h = im.size
        nh = int(h * (inner_w / w))
        resized.append(im.resize((inner_w, nh), resample))
        heights.append(nh)

    total_h = sum(heights) + frame_gap * (len(resized) + 1)

    # base canvas
    strip = Image.new("RGBA", (strip_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(strip)

    # Side perforations
    num_holes = max(1, total_h // perf_spacing)
    for i in range(num_holes):
        cy = int(i * perf_spacing + perf_spacing / 2)
        y0 = cy - hole_h // 2
        y1 = cy + hole_h // 2

        draw.rounded_rectangle(
            [left_hole_x0, y0, left_hole_x1, y1],
            radius=15, fill=(0, 0, 0, 0)
        )
        draw.rounded_rectangle(
            [right_hole_x0, y0, right_hole_x1, y1],
            radius=15, fill=(0, 0, 0, 0)
        )

    # Text
    font = load_font(font_path, font_size)
    target_width = int(band_w * band_ratio)

    y = frame_gap
    for index, im in enumerate(resized):
        h = im.size[1]
        suf = suffixes[index]

        # black window frame 
        draw.rectangle(
            [photo_x - 10, y - 10, photo_x + inner_w + 10, y + h + 10],
            fill=(0, 0, 0, 255)
        )

        strip.paste(im, (photo_x, y))
        cy = y + h // 2

        # LEFT — "KODAK GOLD"
        left_text = make_vertical_scaled("KODAK GOLD", font, target_width)
        lx = (left_band_x0 + left_band_x1) // 2 - left_text.size[0] // 2
        ly = cy - left_text.size[1] // 2
        strip.paste(left_text, (lx, ly), left_text)

        # RIGHT — Suffix (ex. 0022)
        right_text = make_vertical_scaled(suf, font, target_width)
        rx = (right_band_x0 + right_band_x1) // 2 - right_text.size[0] // 2
        ry = cy - right_text.size[1] // 2
        strip.paste(right_text, (rx, ry), right_text)

        # BETWEEN FRAMES — roll code
        if index < len(resized) - 1:
            gap_center = y + h + frame_gap // 2
            gap_text = make_vertical_scaled(roll_code, font, target_width)
            gx = (right_band_x0 + right_band_x1) // 2 - gap_text.size[0] // 2
            gy = gap_center - gap_text.size[1] // 2
            strip.paste(gap_text, (gx, gy), gap_text)

        y += h + frame_gap

    return strip


def build_horizontal_strip_upright(
    image_files,
    roll_code: str,
    strip_width: int = DEFAULT_STRIP_WIDTH,
    frame_gap: int = DEFAULT_FRAME_GAP,
    hole_w: int = DEFAULT_HOLE_WIDTH,
    hole_h: int = DEFAULT_HOLE_HEIGHT,
    perf_spacing: int = DEFAULT_PERF_SPACING,
    band_ratio: float = DEFAULT_TEXT_BAND_RATIO,
    font_path: str = DEFAULT_FONT_PATH,
    font_size: int = DEFAULT_FONT_SIZE,
) -> Image.Image:
    """
    HORIZONTAL orientation version:
    - Photos are rotated 90° CCW before layout, then the whole strip is rotated 90° CW at the end.
      This keeps photos upright in the final horizontal strip.
      Text and perforations remain correctly oriented.
    """

    image_files = sorted(image_files)

    # Load images and rotates them 90º CCW CCW (Pillow rotates CCW if using a positive angle)
    images = []
    suffixes = []
    for path in image_files:
        im = Image.open(path).convert("RGB")
        im = im.rotate(90, expand=True)  # CCW
        images.append(im)
        suffixes.append(extract_suffix(path))

    # Geometry (same as with horizontal)
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
    resized = []
    heights = []
    resample = get_resample()
    for im in images:
        w, h = im.size
        nh = int(h * (inner_w / w))
        resized.append(im.resize((inner_w, nh), resample))
        heights.append(nh)

    total_h = sum(heights) + frame_gap * (len(resized) + 1)

    # base canvas
    strip = Image.new("RGBA", (strip_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(strip)

    # Perforations (on the sides at this stage; they will end up top/bottom after rotation)
    num_holes = max(1, total_h // perf_spacing)
    for i in range(num_holes):
        cy = int(i * perf_spacing + perf_spacing / 2)
        y0 = cy - hole_h // 2
        y1 = cy + hole_h // 2

        draw.rounded_rectangle(
            [left_hole_x0, y0, left_hole_x1, y1],
            radius=15, fill=(0, 0, 0, 0)
        )
        draw.rounded_rectangle(
            [right_hole_x0, y0, right_hole_x1, y1],
            radius=15, fill=(0, 0, 0, 0)
        )

    font = load_font(font_path, font_size)
    target_width = int(band_w * band_ratio)

    y = frame_gap
    for index, im in enumerate(resized):
        h = im.size[1]
        suf = suffixes[index]

        # black window frame
        draw.rectangle(
            [photo_x - 10, y - 10, photo_x + inner_w + 10, y + h + 10],
            fill=(0, 0, 0, 255)
        )

        strip.paste(im, (photo_x, y))
        cy = y + h // 2

        # LEFT — "KODAK GOLD"
        left_text = make_vertical_scaled("KODAK GOLD", font, target_width)
        lx = (left_band_x0 + left_band_x1) // 2 - left_text.size[0] // 2
        ly = cy - left_text.size[1] // 2
        strip.paste(left_text, (lx, ly), left_text)

        # RIGTH — Suffix
        right_text = make_vertical_scaled(suf, font, target_width)
        rx = (right_band_x0 + right_band_x1) // 2 - right_text.size[0] // 2
        ry = cy - right_text.size[1] // 2
        strip.paste(right_text, (rx, ry), right_text)

        # BETWEEN FRAMES — roll_code
        if index < len(resized) - 1:
            gap_center = y + h + frame_gap // 2
            gap_text = make_vertical_scaled(roll_code, font, target_width)
            gx = (right_band_x0 + right_band_x1) // 2 - gap_text.size[0] // 2
            gy = gap_center - gap_text.size[1] // 2
            strip.paste(gap_text, (gx, gy), gap_text)

        y += h + frame_gap

    # Now rotate the entire strip 90° CW so it becomes horizontal,
    # (but the photos will be upright as they were previously rotated CCW)
    strip_horizontal = strip.rotate(-90, expand=True)
    return strip_horizontal

# ============================================================
# ORIENTATION GENERATOR (VERTICAL / HORIZONTAL)
# ============================================================

def build_strip_with_orientation(
    image_files,
    roll_code: str,
    orientation: str = "vertical",
    **kwargs,
):
    """
    Build a strip according to orientation:
    - vertical   → uses build_vertical_strip (original behavior)
    - horizontal → uses build_horizontal_strip_upright (upright photos, horizontal strip)
    """
    if orientation.lower() == "horizontal":
        return build_horizontal_strip_upright(
            image_files,
            roll_code,
            **kwargs,
        )
    else:
        return build_vertical_strip(
            image_files,
            roll_code,
            **kwargs,
        )

# ============================================================
# CONTACT SHEET GENERATOR (MULTIPLE STRIPS)
# ============================================================

def chunk_list(lst, size: int):
    """Divides the list in blocks of the specified size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def build_contact_sheet(
    image_files,
    roll_code: str,
    frames_per_strip: int = DEFAULT_FRAMES_PER_STRIP,
    orientation: str = "vertical",
    strip_width: int = DEFAULT_STRIP_WIDTH,
    **kwargs,
):
    """
    Split photos into multiple strips and assemble them into a single image.

    - frames_per_strip: number of frames per strip
    - vertical: strips placed side by side (columns)
    - horizontal: strips stacked (rows)
    """
    image_files = sorted(image_files)
    groups = chunk_list(image_files, frames_per_strip)

    strips = []
    for group in groups:
        strip = build_strip_with_orientation(
            group, roll_code, orientation=orientation,
            strip_width=strip_width, **kwargs
        )
        strips.append(strip)

    if not strips:
        raise ValueError("No strip was generated.")

    if orientation.lower() == "vertical":
        # lado a lado
        total_w = sum(s.width for s in strips) + 40 * (len(strips) - 1)
        max_h = max(s.height for s in strips)
        sheet = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))

        x = 0
        for s in strips:
            sheet.paste(s, (x, 0), s)
            x += s.width + 40

    else:
        # en filas
        max_w = max(s.width for s in strips)
        total_h = sum(s.height for s in strips) + 40 * (len(strips) - 1)
        sheet = Image.new("RGBA", (max_w, total_h), (0, 0, 0, 0))

        y = 0
        for s in strips:
            sheet.paste(s, (0, y), s)
            y += s.height + 40

    return sheet


# ============================================================
# METADATA PERSISTENCE
# ============================================================

def save_with_metadata(img: Image.Image, path: str, meta: Dict[str, str]):
    """
    Saves the PNG file with tEXt metadata.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(k, str(v))
        img.save(path, "PNG", pnginfo=pnginfo)
    else:
        img.save(path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Kodak Gold negative strip generator")

    parser.add_argument("input_dir", help="Folder containing the input images")
    parser.add_argument("roll_code", help="Roll code (e.g. DvG-250501)")
    parser.add_argument("output", help="Output file (PNG)")

    parser.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--mode", choices=["strip", "contact"], default="strip")
    parser.add_argument("--frames-per-strip", type=int, default=DEFAULT_FRAMES_PER_STRIP)
    parser.add_argument("--strip-width", type=int, default=DEFAULT_STRIP_WIDTH)
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP)
    parser.add_argument("--font-path", default=DEFAULT_FONT_PATH)
    parser.add_argument("--meta", action="append", default=[],
                        help="Añadir metadatos: --meta clave=valor")

    args = parser.parse_args()

    # Buscar imágenes
    if not os.path.isdir(args.input_dir):
        print("Folder not found:", args.input_dir)
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png")
    image_files = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.lower().endswith(exts)
    ]

    if not image_files:
        print("No images were found on the source directory.")
        sys.exit(1)

    meta = parse_meta(args.meta)

    common_kwargs = dict(
        strip_width=args.strip_width,
        frame_gap=args.frame_gap,
        font_path=args.font_path,
    )

    if args.mode == "strip":
        img = build_strip_with_orientation(
            image_files, args.roll_code,
            orientation=args.orientation,
            **common_kwargs
        )
    else:
        img = build_contact_sheet(
            image_files, args.roll_code,
            frames_per_strip=args.frames_per_strip,
            orientation=args.orientation,
            strip_width=args.strip_width,
            **common_kwargs
        )

    save_with_metadata(img, args.output, meta)
    print("Strip generated at:", args.output)


if __name__ == "__main__":
    main()