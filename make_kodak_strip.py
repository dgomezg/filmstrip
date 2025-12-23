import os
import sys
import argparse
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================

DEFAULT_STRIP_WIDTH = 1500        # Ancho de la tira en modo vertical
DEFAULT_STRIP_HEIGHT = 1500       # (Reservado si quieres lógica específica en horizontal)
DEFAULT_FRAME_GAP = 100           # Separación entre fotogramas
DEFAULT_HOLE_WIDTH = 70           # Ancho perforación
DEFAULT_HOLE_HEIGHT = 90          # Alto perforación
DEFAULT_PERF_SPACING = 140        # Distancia entre perforaciones
DEFAULT_TEXT_COLOR = (255, 215, 0, 255)  # "Kodak Gold"
DEFAULT_FONT_SIZE = 48
DEFAULT_TEXT_BAND_RATIO = 0.4     # % de ancho de banda que ocupa el texto
DEFAULT_FRAMES_PER_STRIP = 4      # Usado en modo contact-sheet
DEFAULT_FONT_PATH = ""            # Path a fuente TTF opcional


# ============================================================
# UTILIDADES
# ============================================================

def get_resample():
    """Devuelve LANCZOS compatible con diferentes versiones de Pillow."""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Carga una fuente TTF o usa la fuente por defecto."""
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def extract_suffix(filename: str) -> str:
    """
    Extrae el sufijo del nombre de archivo.
    Ej: 'DVG-250501-0087.jpg' -> '0087'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("-")[-1]


def parse_meta(meta_list):
    """
    Convierte una lista 'clave=valor' en un diccionario.
    Ej: ["Author=David", "Film=Kodak"] → {"Author": "David", "Film": "Kodak"}
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
    Crea una imagen con texto horizontal, la rota 90º y la escala a target_width.
    """
    # lienzo temporal grande
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
# GENERADOR DE TIRA VERTICAL
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

    # Ordenar imágenes
    image_files = sorted(image_files)

    # Cargar imágenes + sufijos
    images = []
    suffixes = []
    for path in image_files:
        im = Image.open(path).convert("RGB")
        images.append(im)
        suffixes.append(extract_suffix(path))

    # Geometría horizontal
    left_margin = 40
    right_margin = 40
    strip_w = strip_width
    band_w = hole_w  # la banda de texto tiene ancho igual al de las perforaciones

    left_hole_x0 = left_margin
    left_hole_x1 = left_hole_x0 + hole_w

    left_band_x0 = left_hole_x1
    left_band_x1 = left_band_x0 + band_w

    right_hole_x1 = strip_w - right_margin
    right_hole_x0 = right_hole_x1 - hole_w

    right_band_x1 = right_hole_x0
    right_band_x0 = right_band_x1 - band_w

    # ancho interno para fotos:
    inner_w = strip_w - (left_margin + hole_w + band_w + band_w + hole_w + right_margin)
    photo_x = left_band_x1

    # Redimensionar fotos al mismo ancho
    resized = []
    heights = []
    resample = get_resample()
    for im in images:
        w, h = im.size
        nh = int(h * (inner_w / w))
        resized.append(im.resize((inner_w, nh), resample))
        heights.append(nh)

    total_h = sum(heights) + frame_gap * (len(resized) + 1)

    # Lienzo base
    strip = Image.new("RGBA", (strip_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(strip)

    # Perforaciones laterales
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

    # Texto
    font = load_font(font_path, font_size)
    target_width = int(band_w * band_ratio)

    y = frame_gap
    for index, im in enumerate(resized):
        h = im.size[1]
        suf = suffixes[index]

        # marco negro tipo ventanilla
        draw.rectangle(
            [photo_x - 10, y - 10, photo_x + inner_w + 10, y + h + 10],
            fill=(0, 0, 0, 255)
        )

        strip.paste(im, (photo_x, y))
        cy = y + h // 2

        # IZQUIERDA — "KODAK GOLD"
        left_text = make_vertical_scaled("KODAK GOLD", font, target_width)
        lx = (left_band_x0 + left_band_x1) // 2 - left_text.size[0] // 2
        ly = cy - left_text.size[1] // 2
        strip.paste(left_text, (lx, ly), left_text)

        # DERECHA — sufijo (ej. 0087)
        right_text = make_vertical_scaled(suf, font, target_width)
        rx = (right_band_x0 + right_band_x1) // 2 - right_text.size[0] // 2
        ry = cy - right_text.size[1] // 2
        strip.paste(right_text, (rx, ry), right_text)

        # ENTRE FOTOGRAMAS — código de carrete
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
    Versión para orientación HORIZONTAL:
    - Las fotos se mantienen "derechas" para el espectador.
    - El truco: se rotan 90º CCW antes de maquetar, y al final toda la tira se rota 90º CW.
      Fotos: -90º + 90º = 0º (quedan sin girar).
      Texto / perforaciones sí quedan en horizontal.
    """

    image_files = sorted(image_files)

    # Cargar imágenes rotándolas 90º CCW (Pillow rota CCW con ángulo positivo)
    images = []
    suffixes = []
    for path in image_files:
        im = Image.open(path).convert("RGB")
        im = im.rotate(90, expand=True)  # CCW
        images.append(im)
        suffixes.append(extract_suffix(path))

    # Geometría (igual que la vertical)
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

    # Redimensionar fotos al mismo ancho
    resized = []
    heights = []
    resample = get_resample()
    for im in images:
        w, h = im.size
        nh = int(h * (inner_w / w))
        resized.append(im.resize((inner_w, nh), resample))
        heights.append(nh)

    total_h = sum(heights) + frame_gap * (len(resized) + 1)

    # Lienzo base
    strip = Image.new("RGBA", (strip_w, total_h), (0, 0, 0, 255))
    draw = ImageDraw.Draw(strip)

    # Perforaciones (laterales en esta fase; después quedarán arriba/abajo)
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

        # marco negro tipo ventanilla
        draw.rectangle(
            [photo_x - 10, y - 10, photo_x + inner_w + 10, y + h + 10],
            fill=(0, 0, 0, 255)
        )

        strip.paste(im, (photo_x, y))
        cy = y + h // 2

        # IZQUIERDA — "KODAK GOLD"
        left_text = make_vertical_scaled("KODAK GOLD", font, target_width)
        lx = (left_band_x0 + left_band_x1) // 2 - left_text.size[0] // 2
        ly = cy - left_text.size[1] // 2
        strip.paste(left_text, (lx, ly), left_text)

        # DERECHA — sufijo
        right_text = make_vertical_scaled(suf, font, target_width)
        rx = (right_band_x0 + right_band_x1) // 2 - right_text.size[0] // 2
        ry = cy - right_text.size[1] // 2
        strip.paste(right_text, (rx, ry), right_text)

        # ENTRE FOTOGRAMAS — roll_code
        if index < len(resized) - 1:
            gap_center = y + h + frame_gap // 2
            gap_text = make_vertical_scaled(roll_code, font, target_width)
            gx = (right_band_x0 + right_band_x1) // 2 - gap_text.size[0] // 2
            gy = gap_center - gap_text.size[1] // 2
            strip.paste(gap_text, (gx, gy), gap_text)

        y += h + frame_gap

    # Ahora rotamos toda la tira 90º CW para que quede horizontal,
    # pero las fotos neto se quedan derechas (porque antes las rotamos CCW)
    strip_horizontal = strip.rotate(-90, expand=True)
    return strip_horizontal

# ============================================================
# GENERADOR CON ORIENTACIÓN (VERTICAL / HORIZONTAL)
# ============================================================

def build_strip_with_orientation(
    image_files,
    roll_code: str,
    orientation: str = "vertical",
    **kwargs,
):
    """
    - vertical  → usa build_vertical_strip (comportamiento actual)
    - horizontal → usa build_horizontal_strip_upright (fotos derechas, tira horizontal)
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
# GENERADOR CONTACT SHEET (VARIAS TIRAS)
# ============================================================

def chunk_list(lst, size: int):
    """Divide la lista en bloques de tamaño size."""
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
    Divide las fotos en varias tiras y las une en una imagen:
      - vertical: tiras en columnas
      - horizontal: tiras apiladas como filas
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
        raise ValueError("No se generó ninguna tira.")

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
# GUARDADO CON METADATOS
# ============================================================


def save_with_metadata(img: Image.Image, path: str, meta: Dict[str, str]):
    """
    Guarda el PNG con metadatos tEXt.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(k, str(v))
        img.save(path, "PNG", pnginfo=pnginfo)
    else:
        img.save(path)


# ================================
# PDF/Output helpers
# ================================

def save_pdf(img: Image.Image, pdf_path: str):
    """Save a (single-page) PDF version of the generated image.

    Notes:
    - PDF does not support alpha the same way PNG does.
    - We flatten transparency onto a white background before exporting.
    """
    # Ensure we don't lose the transparent perforations in an unexpected way
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        pdf_img = background
    else:
        pdf_img = img.convert("RGB")

    # Pillow can write a single-page PDF directly
    pdf_img.save(pdf_path, "PDF")


def save_output(img: Image.Image, output_path: str, meta: Dict[str, str], export_pdf: bool):
    """Save output as PNG/JPG/etc and optionally export a PDF copy."""
    save_with_metadata(img, output_path, meta)

    if export_pdf:
        base, _ext = os.path.splitext(output_path)
        pdf_path = base + ".pdf"
        save_pdf(img, pdf_path)
        print("PDF generado en:", pdf_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generador de tiras Kodak Gold")

    parser.add_argument("input_dir", help="Carpeta con las imágenes de entrada")
    parser.add_argument("roll_code", help="Código del carrete (ej. DvG-250501)")
    parser.add_argument("output", help="Archivo de salida (png)")

    parser.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--mode", choices=["strip", "contact"], default="strip")
    parser.add_argument("--frames-per-strip", type=int, default=DEFAULT_FRAMES_PER_STRIP)
    parser.add_argument("--strip-width", type=int, default=DEFAULT_STRIP_WIDTH)
    parser.add_argument("--frame-gap", type=int, default=DEFAULT_FRAME_GAP)
    parser.add_argument("--font-path", default=DEFAULT_FONT_PATH)
    parser.add_argument("--meta", action="append", default=[],
                        help="Añadir metadatos: --meta clave=valor")
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="Export an additional PDF copy next to the output image (same name, .pdf)"
    )

    args = parser.parse_args()

    # Buscar imágenes
    if not os.path.isdir(args.input_dir):
        print("Carpeta no encontrada:", args.input_dir)
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png")
    image_files = [
        os.path.join(args.input_dir, f)
        for f in sorted(os.listdir(args.input_dir))
        if f.lower().endswith(exts)
    ]

    if not image_files:
        print("No se encontraron imágenes.")
        sys.exit(1)

    meta = parse_meta(args.meta)

    common_kwargs = dict(
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

    save_output(img, args.output, meta, export_pdf=args.export_pdf)
    print("Tira generada en:", args.output)


if __name__ == "__main__":
    main()