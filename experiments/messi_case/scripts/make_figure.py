"""Render paired MESSI help and harm examples from frozen outputs."""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1] / "shuanglan" / "polished" / "figures"
SIZE = (2737, 1659)
PANEL_SIZE = (714, 714)
PANEL_X = {"A": 162, "B": 1120, "C": 2000}
ROW_DATA = [
    ("100_0042_r16", "Correction with low image"),
    ("100_0002_r02", "Harm with low image"),
]
VIEWS = {
    "A": ("high_marked", "repeat high view"),
    "B": ("high_crop", "high-view digital zoom"),
    "C": ("low_crop", "real low-view crop"),
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def paste_framed(canvas, image, xy, size):
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x, y = xy
    px = x + (size[0] - image.width) // 2
    py = y + (size[1] - image.height) // 2
    canvas.paste(image, (px, py))
    ImageDraw.Draw(canvas).rectangle(
        (px, py, px + image.width - 1, py + image.height - 1),
        outline="black",
        width=3,
    )


def add_row_label(canvas, text, center_y):
    label_font = font(31)
    scratch = Image.new("RGBA", (700, 260), (255, 255, 255, 0))
    d = ImageDraw.Draw(scratch)
    bounds = d.multiline_textbbox((0, 0), text, font=label_font, spacing=5, align="center")
    crop = scratch.crop((0, 0, bounds[2] + 2, bounds[3] + 2))
    ImageDraw.Draw(crop).multiline_text(
        (1, 1), text, font=label_font, fill="black", spacing=5, align="center"
    )
    rotated = crop.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(rotated, (20, center_y - rotated.height // 2), rotated)


def main():
    with (ROOT / "results" / "per_sample.csv").open(newline="") as f:
        rows = {row["sample_id"]: row for row in csv.DictReader(f)}

    canvas = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(39)
    answer_font = font(37)
    colors = {True: "#147d52", False: "#b33030"}
    # The first row's A panel is a landscape view; B and C are matched square
    # crops. Centering A in the same row keeps the paired crops aligned.
    row_y = [107, 922]
    a_y = [176, 991]
    title_y = [17, 832]
    a_title_y = [90, 905]

    for row_i, (sample_id, description) in enumerate(ROW_DATA):
        row = rows[sample_id]
        for branch, (view_file, view_label) in VIEWS.items():
            image_path = ROOT / "derived" / "inputs" / f"{sample_id}_{view_file}.png"
            image = Image.open(image_path)
            if branch == "A":
                paste_framed(canvas, image, (PANEL_X[branch], a_y[row_i]), (866, 577))
                heading_y = a_title_y[row_i]
            else:
                paste_framed(canvas, image, (PANEL_X[branch], row_y[row_i]), PANEL_SIZE)
                heading_y = title_y[row_i]

            center_x = PANEL_X[branch] + (866 if branch == "A" else PANEL_SIZE[0]) // 2
            correct = row[f"{branch}_correct"] == "1"
            answer = row[f"{branch}_answer"].upper()
            status = "correct" if correct else "incorrect"
            color = colors[correct]
            draw.text((center_x, heading_y), f"{branch}: {view_label}",
                      font=title_font, fill=color, anchor="mt")
            draw.text((center_x, heading_y + 43), f"{answer} ({status})",
                      font=answer_font, fill=color, anchor="mt")

        label = f"{sample_id}\n{description}\nGT: {row['truth'].upper()}"
        center_y = row_y[row_i] + PANEL_SIZE[1] // 2
        add_row_label(canvas, label, center_y)

    PAPER.mkdir(parents=True, exist_ok=True)
    output = PAPER / "messi_case_pairs.png"
    canvas.save(output, dpi=(240, 240), optimize=True)
    print(output)


if __name__ == "__main__":
    main()
