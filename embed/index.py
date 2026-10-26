"""Reference index: one embedding per catalog card, computed from its clean scan.

Adding a card means embedding one more image; the model is not retrained. The
index is cached to disk and rebuilt automatically when the checkpoint or the
catalog changes.

Usage:
    python -m embed.index                                   # whole catalog
    python -m embed.index --ckpt checkpoints/stage1/embedder.pt --out checkpoints/index_full.pt
"""

import argparse
import csv
from pathlib import Path

import cv2
import torch

from embed.augment import to_tensor
from embed.evaluate import embed_all
from embed.model import CardEmbedder


def load_embedder(ckpt_path, device):
    """Load the head-free embedding model saved by train_stage1 (embedder.pt)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = CardEmbedder(emb_dim=ckpt["config"]["emb_dim"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def image_to_uint8(img_rgb):
    """HxWx3 uint8 RGB -> 3xHxW uint8 tensor at the model's input size.

    embed_all expects uint8 (it normalizes itself); passing [0, 1] floats makes
    every image look the same.
    """
    return (to_tensor(img_rgb) * 255).round().to(torch.uint8)


def load_catalog_rows(data_dir):
    with open(Path(data_dir) / "catalog.csv", newline="") as f:
        return list(csv.DictReader(f))


def build_index(model, rows, data_dir, device, batch_size=256):
    """Embed each catalog card's clean low-res scan. Returns [N, D] normalized embeddings."""
    chunks = []
    for i in range(0, len(rows), batch_size):  # load in batches so all images never sit in memory
        images = []
        for row in rows[i:i + batch_size]:
            img = cv2.imread(str(Path(data_dir) / row["low_path"]))
            images.append(image_to_uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        chunks.append(embed_all(model, torch.stack(images), device).cpu())
        print(f"\rembedded {min(i + batch_size, len(rows))}/{len(rows)} cards", end="", flush=True)
    print()
    return torch.cat(chunks)


def load_or_build_index(model, ckpt_path, data_dir, device, cache_path="checkpoints/index_full.pt"):
    """Return (rows, embeddings), reusing the cached index if it still matches."""
    rows = load_catalog_rows(data_dir)
    key = {
        "ckpt": str(Path(ckpt_path).resolve()),
        "ckpt_mtime": Path(ckpt_path).stat().st_mtime,
        "card_ids": [r["card_id"] for r in rows],
    }
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached["key"] == key:
            return rows, cached["embeddings"]
        print("catalog or checkpoint changed; rebuilding index")
    embeddings = build_index(model, rows, data_dir, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "embeddings": embeddings}, cache_path)
    return rows, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    parser.add_argument("--data-dir", default="data/tcgdex")
    parser.add_argument("--out", default="checkpoints/index_full.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embedder(args.ckpt, device)
    rows, embeddings = load_or_build_index(model, args.ckpt, args.data_dir, device, args.out)
    print(f"index: {len(rows)} cards, {embeddings.shape[1]}-d embeddings -> {args.out}")


if __name__ == "__main__":
    main()
