"""Print OCR name matches and fused rankings for retrieved candidates.

python -m ocr.evaluate --demo-list data/demo_cards.txt \
    --query data/tcgdex/images/low/sv01/sv01-202.webp
"""

import argparse

from ocr.match import score_candidates
from ocr.reader import read_name
from fuse.rank import rank_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="upright, straightened card crop")
    parser.add_argument("--demo-list", required=True)
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    parser.add_argument("--data", default="data/tcgdex")
    parser.add_argument("--cache", default="data/demo_gallery.pt")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ocr-weight", type=float, default=0.2,
                        help="OCR weight from 0 to 1; embedding weight is 1 minus this (default: 0.2)")
    parser.add_argument("--device")
    parser.add_argument("--save-crop", help="optional path to inspect the processed name band")
    args = parser.parse_args()
    if not 0 <= args.ocr_weight <= 1:
        parser.error("--ocr-weight must be between 0 and 1")
    if args.k < 1:
        parser.error("--k must be at least 1")
    reading = read_name(args.query, args.save_crop)
    print(f"OCR text: {reading['text']!r}", flush=True)
    print(f"OCR available: {reading['ocr_available']}", flush=True)
    if reading["reason"]:
        print(reading["reason"], flush=True)
    from embed.retrieve import CardRetriever
    retriever = CardRetriever(args.demo_list, args.ckpt, args.data, args.cache, device=args.device)
    candidates = score_candidates(retriever.retrieve(args.query, args.k), reading)
    print("\nEmbedding order (scores are similarities, not probabilities):")
    print(f"{'Rank':4} {'Card ID':20} {'Cosine':>8} {'OCR name':>9}  Name")
    for rank, candidate in enumerate(candidates, 1):
        score = candidate["ocr_score"]
        label = f"{score:.4f}" if score is not None else "N/A"
        print(f"{rank:4} {candidate['card_id']:20} {candidate['cosine_score']:8.4f} "
              f"{label:>9}  {candidate['metadata']['name']}")
    if reading["ocr_available"]:
        best = max(candidates, key=lambda c: c["ocr_score"])
        print(f"\nBest name match within shortlist: {best['card_id']} ({best['metadata']['name']})")
    else:
        print("\nOCR unavailable; embedding ranking remains available.")

    ranked = rank_candidates(candidates, ocr_weight=args.ocr_weight)
    if ranked[0]["ocr_used"]:
        print(f"\nFusion: {1 - args.ocr_weight:.2f} * cosine + {args.ocr_weight:.2f} * OCR name")
    else:
        print("\nFusion: embedding only (OCR unavailable or weight is zero)")
    print(f"{'Rank':4} {'Was':>4} {'Card ID':20} {'Cosine':>8} {'OCR name':>9} {'Final':>8}  Name")
    original_ranks = {c["card_id"]: i for i, c in enumerate(candidates, 1)}
    for rank, candidate in enumerate(ranked, 1):
        score = candidate["ocr_score"]
        label = f"{score:.4f}" if score is not None else "N/A"
        print(f"{rank:4} {original_ranks[candidate['card_id']]:4} {candidate['card_id']:20} "
              f"{candidate['cosine_score']:8.4f} {label:>9} {candidate['final_score']:8.4f}  "
              f"{candidate['metadata']['name']}")
    best = ranked[0]
    print(f"\nFinal prediction: {best['card_id']} ({best['metadata']['name']})")


if __name__ == "__main__":
    main()
