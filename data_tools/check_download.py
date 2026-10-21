"""Sanity-check a TCGdex download: counts, skip reasons, image sizes, and a sample grid.

Usage:
    python -m data_tools.check_download [--data data/tcgdex] [--n 25]
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_grid(data_dir, rows, n, out_path, thumb_w=180):
    """Paste n random card images into one grid, with the card name under each."""
    sample = random.sample(rows, min(n, len(rows)))
    thumb_h = int(thumb_w * 88 / 63)  # card aspect ratio 63x88 mm
    label_h = 30
    cols = 5
    grid_rows = (len(sample) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_w, grid_rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(grid)

    for i, row in enumerate(sample):
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        with Image.open(data_dir / row["low_path"]) as img:
            grid.paste(img.convert("RGB").resize((thumb_w, thumb_h)), (x, y))
        draw.text((x + 4, y + thumb_h + 2), row["name"][:28], fill="black")
        draw.text((x + 4, y + thumb_h + 15), f"{row['local_id']}/{row['set_card_count']} {row['set_id']}", fill="gray")

    grid.save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/tcgdex")
    parser.add_argument("--n", type=int, default=25, help="cards in the sample grid")
    args = parser.parse_args()
    data_dir = Path(args.data)

    catalog = read_csv(data_dir / "catalog.csv")
    skipped = read_csv(data_dir / "skipped.csv")
    if not catalog:
        print(f"No catalog found in {data_dir}. Run download_tcgdex first.")
        return

    print(f"Cards in catalog: {len(catalog)}   skipped: {len(skipped)}\n")

    print("Cards per set:")
    for set_id, count in sorted(Counter(r["set_id"] for r in catalog).items()):
        print(f"  {set_id:12s} {count}")

    print("\nCategories:", dict(Counter(r["category"] for r in catalog)))
    print("Missing high-res image:", sum(1 for r in catalog if not r["high_path"]))

    if skipped:
        print("\nSkip reasons:")
        reasons = Counter(r["reason"].split(" for ")[0] for r in skipped)
        for reason, count in reasons.most_common():
            print(f"  {count:5d}  {reason}")

    # Image sizes: check a sample instead of opening every file.
    sizes = Counter()
    for r in random.sample(catalog, min(200, len(catalog))):
        with Image.open(data_dir / r["low_path"]) as img:
            sizes[("low", img.size)] += 1
        if r["high_path"]:
            with Image.open(data_dir / r["high_path"]) as img:
                sizes[("high", img.size)] += 1
    print("\nImage sizes (sample of up to 200 cards):")
    for (kind, size), count in sizes.most_common():
        print(f"  {kind:5s} {size[0]}x{size[1]}  x{count}")

    grid_path = data_dir / "sample_grid.png"
    make_grid(data_dir, catalog, args.n, grid_path)
    print(f"\nSample grid saved to {grid_path}")


if __name__ == "__main__":
    main()
