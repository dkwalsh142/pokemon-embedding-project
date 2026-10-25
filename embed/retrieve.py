"""Retrieve demo cards by cosine similarity.

Demo list: whitespace-separated catalog card IDs (one per line is convenient).
Run from the project root, after downloading data and training Stage 1:

    python -m embed.retrieve --demo-list data/demo_cards.txt --query path/to/crop.jpg

The gallery is cached under data/. Use --rebuild after changing reference images.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from embed.augment import clean_view
from embed.evaluate import embed_all
from embed.model import CardEmbedder


def load_embedder(checkpoint="checkpoints/stage1/embedder.pt", device=None):
    """Load the inference checkpoint exported by train_stage1; return model, device."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CardEmbedder(emb_dim=ckpt["config"]["emb_dim"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), device


def read_image(path):
    """Read a clean reference or already-warped query as an RGB uint8 CHW tensor."""
    with Image.open(path) as img:
        return clean_view(np.array(img.convert("RGB")))


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CardRetriever:
    """A demo-only gallery with one entry per card ID, including reprints.

    Construct with the same checkpoint and demo list on subsequent runs to load
    cached embeddings. Catalog metadata is retained for downstream OCR matching.
    """

    def __init__(self, demo_list, checkpoint="checkpoints/stage1/embedder.pt",
                 data_dir="data/tcgdex", cache="data/demo_gallery.pt",
                 device=None, rebuild=False, batch_size=256):
        self.model, self.device = load_embedder(checkpoint, device)
        data_dir = Path(data_dir)
        ids = list(dict.fromkeys(Path(demo_list).read_text().split()))
        if not ids:
            raise ValueError("Demo list is empty; add catalog card IDs, one per line.")
        with open(data_dir / "catalog.csv", newline="") as f:
            catalog = {row["card_id"]: row for row in csv.DictReader(f)}
        missing = [card_id for card_id in ids if card_id not in catalog]
        if missing:
            raise ValueError(f"Demo IDs missing from catalog.csv: {', '.join(missing)}")
        self.cards = [catalog[card_id] for card_id in ids]
        paths = [data_dir / row["low_path"] for row in self.cards]
        # Invalidate on weight, metadata, image timestamp/size, or preprocessing changes.
        signature = {
            "version": 1,
            "checkpoint_sha256": checkpoint_digest(checkpoint),
            "cards": self.cards,
            "images": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                       for p in paths],
        }
        cache = Path(cache)
        cached = None
        if cache.exists() and not rebuild:
            cached = torch.load(cache, map_location="cpu", weights_only=True)
        if cached is not None and cached.get("signature") == signature:
            self.gallery = cached["embeddings"].to(self.device)
            self.gallery_source = "loaded"
        else:
            # Batch image loading as well as embedding to bound memory use.
            batches = []
            for start in range(0, len(paths), batch_size):
                images = torch.stack([read_image(p) for p in paths[start:start + batch_size]])
                batches.append(embed_all(self.model, images, self.device, batch_size).cpu())
            embeddings = torch.cat(batches)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"signature": signature, "embeddings": embeddings}, cache)
            self.gallery = embeddings.to(self.device)
            self.gallery_source = "built"

    @torch.no_grad()
    def retrieve(self, image_path, k=5):
        """Return ranked dicts with card_id, full catalog metadata, and cosine_score.

        image_path should be a straightened card crop. Scores are similarities,
        not calibrated probabilities. K is capped at the demo gallery size.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        query = embed_all(self.model, read_image(image_path).unsqueeze(0), self.device)
        scores, indices = (query @ self.gallery.T)[0].topk(min(k, len(self.cards)))
        return [
            {"card_id": self.cards[i]["card_id"], "metadata": dict(self.cards[i]),
             "cosine_score": score}
            for score, i in zip(scores.cpu().tolist(), indices.cpu().tolist())
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo-list", required=True)
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    parser.add_argument("--data", default="data/tcgdex")
    parser.add_argument("--cache", default="data/demo_gallery.pt")
    parser.add_argument("--query", help="straightened card crop; omit to only build/load gallery")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", help="e.g. cpu, cuda, or mps; defaults to CUDA if available, else CPU")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    retriever = CardRetriever(args.demo_list, args.ckpt, args.data, args.cache,
                              device=args.device, rebuild=args.rebuild)
    print(f"Gallery {retriever.gallery_source}: {len(retriever.cards)} cards, "
          f"{retriever.gallery.shape[1]} dimensions, device={retriever.device}")
    if args.query:
        print(f"\nTop-{min(args.k, len(retriever.cards))} matches for {args.query} (cosine, not probability):")
        for rank, result in enumerate(retriever.retrieve(args.query, args.k), 1):
            card = result["metadata"]
            print(f"  {rank:2d}. {result['card_id']:20s} {result['cosine_score']:.4f}  "
                  f"{card['name']} | {card['set_name']} | "
                  f"{card['local_id']}/{card['set_card_count']}")


if __name__ == "__main__":
    main()
