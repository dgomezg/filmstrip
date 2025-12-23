# 📸 *Kodak Gold* Negative Strip Generator — Python

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Pillow](https://img.shields.io/badge/Pillow-image_processing-yellow.svg)

This project generates **Kodak Gold 35mm–style negative strips** from a collection of digital photos.

It supports:

- ✅ **Vertical** or **horizontal** strips
- ✅ **Strip** mode (single strip)
- ✅ **Contact sheet** mode (multiple strips combined)
- ✅ **Transparent side perforations**, mimicking real film
- ✅ “KODAK <EMULSION> [ISO]” edge text, roll code `DvG-YYMMDD`, and frame numbering based on file names
- ✅ Additional EXIF-like metadata embedded into PNG files
- ✅ Export as pdf  
- ✅ Add ISO / emulsion text like classic Kodak rolls

Designed for photographers and designers who want album mockups or prints with an analog look and feel.

---

## 🗂 Example project structure

```text
my-negative-project/
│
├─ make_kodak_strip.py        # Main script (modular and configurable)
├─ README.md                  # This file
│
├─ photos/                    # Input images folder
│   ├─ DVG-240713-0022.jpg
│   ├─ DVG-240713-0023.jpg
│   ├─ DVG-240713-0024.jpg
│   ├─ DVG-240713-0025.jpg
│   ├─ DVG-240713-0026.jpg
│   ├─ DVG-240713-0027.jpg
│   ├─ DVG-240713-0028.jpg
│   ├─ DVG-240713-0029.jpg
│   └─ ...
│
└─ docs/
    ├─ horizontal_strip.png
    ├─ strip.png
    └─ vertical_strip.png
```

> Photos must follow the `DVG-YYMMDD-XXXX.ext` naming convention,  
> where `XXXX` is the suffix used as the frame number (e.g. `0087`).

---

## 🖼 Visual examples

Vertical strip:

![Vertical strip](./output/vertical_strip.png)

Horizontal strip:

![Horizontal strip](./output/horizontal_strip.png)

Contact sheet:

![Contact sheet](./output/contact_sheet_ejemplo.png)

---

## 🛠 Installation

### Requirements

- Python **3.8+**
- Pillow

Install dependencies:

```bash
pip install pillow
```

If you are using python installed with HomeBrew, you will have to use a virtual environment to run the script with pyhon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pillow
```

---

## 🚀 Basic usage

```bash
python make_kodak_strip.py <photos_folder> <roll_code> <output.png>
```

Example:

```bash
python make_kodak_strip.py ./photos DvG-240713 output/vertical_strip.png
```

- `./photos` → folder containing images
- `DvG-240713` → roll code printed between frames
- `vertical_strip.png` → resulting PNG file

---

## 📄 Automatic PDF export

The script can automatically generate a **PDF version** of the resulting strip or contact sheet.

This is useful for:
- Printing
- Sharing a single-file document
- Archiving finished layouts

### Usage

Add the `--export-pdf` flag to any command:

```bash
python make_kodak_strip.py ./photos DvG-240713 contact_sheet.png \
  --mode contact \
  --frames-per-strip 4 \
  --export-pdf
```

This will generate:
- `contact_sheet.png` (or the chosen output image format)
- `contact_sheet.pdf` (same base name, single-page PDF)

### Notes

- The PDF export is **single-page**.
- Since PDF does not support transparency in the same way as PNG, the image is flattened onto a **white background** before exporting.
- The PDF is generated using **Pillow**, so **no additional dependencies** are required.

---

## 📐 Naming convention

The script automatically extracts the frame number:

```text
DVG-240713-0022.jpg → suffix = 0022
```

This number is printed on the **right-hand text band**, aligned with each frame.

The roll code (e.g. `DvG-240713`) appears **between frames**, also on the right band.

---

## 🎞 Modes and advanced options

### ✔ Orientation

```bash
--orientation vertical    # default
--orientation horizontal
```

Example:

```bash
python make_kodak_strip.py ./photos DvG-240713 output/horizontal_strip.png --orientation horizontal
```

---

### ✔ Strip mode

Generates a single strip:

```bash
python make_kodak_strip.py ./photos DvG-240713 output/strip.png --mode strip
```

---

### ✔ Contact sheet mode

Splits photos into multiple strips:

```bash
python make_kodak_strip.py ./photos DvG-240713 contact_sheet.png   --mode contact   --frames-per-strip 4
```

- `--frames-per-strip` → number of frames per strip  
  - Vertical → strips placed side by side  
  - Horizontal → strips stacked  

---

### ✔ Metadata (EXIF-like text)

You can attach metadata to the PNG:

```bash
python make_kodak_strip.py ./photos DvG-240713 output/strip_meta.png   --meta Author="David Gomez"   --meta Project=Stage13Tour2024   --meta Film=KodakGold200
```

---

### ✔ ISO / emulsion edge text

You can customize the classic roll-style edge text printed on the strip using:

- `--emulsion` → film name (defaults to `GOLD`)
- `--iso` → ISO speed (optional)

Examples:

```bash
# Default label: "KODAK GOLD"
python make_kodak_strip.py ./photos DvG-240713 output/strip.png

# Label: "KODAK GOLD 200"
python make_kodak_strip.py ./photos DvG-240713 output/strip_iso.png \
  --emulsion GOLD \
  --iso 200

# Label: "KODAK PORTRA 400"
python make_kodak_strip.py ./photos DvG-240713 output/portra_400.png \
  --emulsion PORTRA \
  --iso 400
```

Notes:
- The label is printed on the **left band**, aligned with each frame.
- When `--iso` is provided, it is appended to the emulsion name.
- The values are also embedded as PNG metadata keys: `Emulsion` and `ISO`.

---

### ✔ Other useful parameters

- `--strip-width <px>` → base negative width
- `--frame-gap <px>` → spacing between frames
- `--font-path` → path to a custom TTF font
- `--orientation` → vertical / horizontal

Example:

```bash
python make_kodak_strip.py ./photos DvG-240713 output/strip_font.png   --font-path "/Library/Fonts/Arial.ttf"
```

---

## 🎨 Generated negative design

### Left band
- Aligned with each photo → `KODAK <EMULSION> [ISO]` (e.g. `KODAK GOLD 200`)

### Right band
- Aligned with each photo → file suffix (`0087`, `0097`, etc.)
- Between frames → roll code (`DvG-250501`)

### Side perforations
- Rounded
- Transparent
- Evenly spaced

### Frame and photos
- Black window around each photo
- Photos centered, colors unchanged
- Black separation between negatives

### Horizontal vs Vertical
- In horizontal mode, the same composition is generated and then rotated 90°.

---

## 🧩 Main script

The full implementation lives in [`make_kodak_strip.py`](./make_kodak_strip.py).

---

## 🧾 Film presets

The `presets.yml` file allows you to preconfigure styles:

```yaml
presets:
  kodak_gold:
    strip_width: 1500
    frame_gap: 100
    hole_width: 70
    hole_height: 90
    perf_spacing: 140
    text_band_ratio: 0.4
    text_color: "#FFD700"
  portra_400:
    strip_width: 1500
    frame_gap: 120
    hole_width: 70
    hole_height: 90
    perf_spacing: 140
    text_band_ratio: 0.35
    text_color: "#E8C598"
```

---

## 🐳 Docker usage

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY make_kodak_strip.py .

RUN pip install pillow

CMD ["python", "make_kodak_strip.py"]
```

Build:

```bash
docker build -t kodak-strip .
```

Run:

```bash
docker run -v "$(pwd)/photos:/app/photos"            -v "$(pwd)/out:/app/out"            kodak-strip            python make_kodak_strip.py photos DvG-250501 out/strip.png
```

---

## 🧰 Makefile

```make
run:
    python make_kodak_strip.py ./photos DvG-250501 vertical_strip.png

contact:
    python make_kodak_strip.py ./photos DvG-250501 contact_sheet.png \
      --mode contact --frames-per-strip 4

horizontal:
    python make_kodak_strip.py ./photos DvG-250501 horizontal_strip.png \
      --orientation horizontal
```

---

## 📬 Ideas for improvement

- Add Fuji / Portra / Ilford look presets
- Display real EXIF data under each frame
- Generate automatic thumbnails
