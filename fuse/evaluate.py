"""Compare embedding-only and fused predictions on labeled card crops.

Without --manifest, generate seeded degraded views of the demo references.
With --manifest, evaluate real crops listed in a CSV: image_path,card_id.
Paths in the CSV are relative to the CSV's directory (or absolute).
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import tempfile

from PIL import Image, ImageDraw, ImageOps

from fuse.rank import rank_candidates
from ocr.match import score_candidates
from ocr.reader import read_name


def load_manifest(path, gallery_ids):
    """Validate ground truth before running inference; never infer it from OCR."""
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not {"image_path", "card_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs image_path and card_id columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    cases = []
    for line, row in enumerate(rows, 2):
        card_id = row["card_id"].strip()
        image_path = row["image_path"].strip()
        if card_id not in gallery_ids:
            raise ValueError(f"Manifest line {line}: {card_id!r} is not in the demo gallery")
        image = path.parent / image_path
        if not image_path or not image.is_file():
            raise ValueError(f"Manifest line {line}: image not found: {image}")
        cases.append({"image_path": str(image.resolve()), "card_id": card_id,
                      "condition": row.get("condition", "").strip() or "real_photo",
                      "seed": None})
    return cases


def synthetic_cases(cards, data_dir, views, seed):
    """Describe clean controls, random camera views, and covered-name controls."""
    for index, card in enumerate(cards):
        common = {"image_path": str((Path(data_dir) / card["low_path"]).resolve()),
                  "card_id": card["card_id"]}
        yield dict(common, condition="clean", seed=None)
        for view in range(views):
            yield dict(common, condition="camera", seed=seed + index * views + view)
        yield dict(common, condition="artwork_covered", seed=None)
        yield dict(common, condition="name_covered", seed=None)


def make_query(case, synthetic):
    with Image.open(case["image_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if not synthetic:
        return image
    if case["condition"] == "camera":
        import numpy as np
        from embed.augment import camera_augment
        tensor = camera_augment(np.array(image), random.Random(case["seed"]))
        return Image.fromarray(tensor.permute(1, 2, 0).numpy())
    if case["condition"] == "name_covered":
        # Cover the entire top band, not only the exact OCR rectangle. Run real
        # OCR on this image; do not force the availability flag to be false.
        ImageDraw.Draw(image).rectangle((0, 0, image.width, round(image.height * .16)),
                                       fill="white")
    if case["condition"] == "artwork_covered":
        # Deliberately severe occlusion stress test: retain the original name
        # but remove most visual evidence. Not a typical camera condition.
        ImageDraw.Draw(image).rectangle((0, round(image.height * .17), image.width, image.height),
                                       fill="gray")
    return image


def summarize(results):
    n = len(results)
    ambiguous = [r for r in results if not r["embedding_correct"] and r["in_top_k"]]
    fallback = [r for r in results if not r["ocr_available"]]
    return {
        "queries": n,
        "embedding_top1": sum(r["embedding_correct"] for r in results) / n if n else None,
        "embedding_top_k": sum(r["in_top_k"] for r in results) / n if n else None,
        "fused_top1": sum(r["fused_correct"] for r in results) / n if n else None,
        "ocr_available": sum(r["ocr_available"] for r in results),
        "ambiguous_queries": len(ambiguous),
        "ocr_fixes": sum(not r["embedding_correct"] and r["fused_correct"] for r in results),
        "ocr_regressions": sum(r["embedding_correct"] and not r["fused_correct"] for r in results),
        "top_k_misses": sum(not r["in_top_k"] for r in results),
        "fallback_queries": len(fallback),
        "fallback_ranking_violations": sum(not r["fallback_preserved"] for r in fallback),
    }


def print_summary(label, results, k):
    scores = summarize(results)
    print(f"\n{label}: {scores['queries']} queries")
    print(f"  Embedding top-1: {scores['embedding_top1']:.1%} | "
          f"top-{k}: {scores['embedding_top_k']:.1%} | fused top-1: {scores['fused_top1']:.1%}")
    print(f"  Ambiguous (wrong top-1, correct in shortlist): {scores['ambiguous_queries']} | "
          f"fixed: {scores['ocr_fixes']} | regressions: {scores['ocr_regressions']}")
    print(f"  OCR available: {scores['ocr_available']} | fallback cases: {scores['fallback_queries']} | "
          f"fallback ranking violations: {scores['fallback_ranking_violations']}")
    if not scores["ambiguous_queries"]:
        print("  No recoverable embedding mistakes in this group; OCR improvement is unproven.")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-list", required=True)
    parser.add_argument("--manifest", help="CSV of labeled real, upright, straightened card crops")
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    parser.add_argument("--data", default="data/tcgdex")
    parser.add_argument("--cache", default="data/demo_gallery.pt")
    parser.add_argument("--device")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ocr-weight", type=float, default=.2)
    parser.add_argument("--views", type=int, default=3, help="synthetic camera views per card")
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--out", default="data/fusion_eval.json", help="JSON report with every candidate score")
    args = parser.parse_args()
    if args.k < 1 or args.views < 1 or not 0 <= args.ocr_weight <= 1:
        parser.error("k and views must be positive; ocr-weight must be between 0 and 1")
    import pytesseract
    try:
        pytesseract.get_tesseract_version()
    except (OSError, RuntimeError) as exc:
        parser.error(f"Install Tesseract before evaluating OCR: {exc}")
    from embed.retrieve import CardRetriever, checkpoint_digest
    retriever = CardRetriever(args.demo_list, args.ckpt, args.data, args.cache, device=args.device)
    synthetic = not args.manifest
    try:
        cases = list(synthetic_cases(retriever.cards, args.data, args.views, args.seed)) if synthetic else \
            load_manifest(args.manifest, {c["card_id"] for c in retriever.cards})
    except ValueError as exc:
        parser.error(str(exc))
    k = min(args.k, len(retriever.cards))
    print("Synthetic scan tests: diagnostic proxies, not real-photo accuracy." if synthetic else
          "Labeled crop evaluation: use held-out real photos for final accuracy.", flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="card-fusion-") as temporary:
        query_path = Path(temporary) / "query.png"
        for i, case in enumerate(cases, 1):
            make_query(case, synthetic).save(query_path)
            candidates = retriever.retrieve(query_path, k)
            reading = read_name(query_path)
            ranked = rank_candidates(score_candidates(candidates, reading), args.ocr_weight)
            truth = case["card_id"]
            embedding_correct = candidates[0]["card_id"] == truth
            fused_correct = ranked[0]["card_id"] == truth
            result = dict(case, ocr=reading, ocr_available=ranked[0]["ocr_available"],
                          embedding_prediction=candidates[0]["card_id"],
                          fused_prediction=ranked[0]["card_id"],
                          embedding_correct=embedding_correct, fused_correct=fused_correct,
                          in_top_k=any(c["card_id"] == truth for c in candidates),
                          fallback_preserved=all(a["card_id"] == b["card_id"] and
                                                 a["cosine_score"] == b["final_score"]
                                                 for a, b in zip(candidates, ranked)),
                          candidates=ranked)
            results.append(result)
            outcome = "FIXED" if not embedding_correct and fused_correct else \
                "REGRESSION" if embedding_correct and not fused_correct else \
                "correct" if fused_correct else "wrong"
            print(f"[{i}/{len(cases)}] {truth} {case['condition']}: {outcome} | "
                  f"embedding={result['embedding_prediction']} fused={result['fused_prediction']} | "
                  f"OCR={reading['text']!r}", flush=True)
    summaries = {"all": print_summary("All", results, k), "by_condition": {}}
    for condition in sorted({r["condition"] for r in results}):
        group = [r for r in results if r["condition"] == condition]
        summaries["by_condition"][condition] = print_summary(condition, group, k)
    if synthetic:
        covered = [r for r in results if r["condition"] == "name_covered"]
        print(f"\nCovered-name cases yielding no OCR: "
              f"{sum(not r['ocr_available'] for r in covered)}/{len(covered)}")
    reasons = Counter(r["ocr"]["reason"] for r in results if r["ocr"]["reason"])
    if reasons:
        print(f"OCR unavailable reasons: {dict(reasons)}")
    report = {"config": vars(args), "mode": "synthetic" if synthetic else "manifest",
              "effective_k": k, "gallery_ids": [c["card_id"] for c in retriever.cards],
              "checkpoint_sha256": checkpoint_digest(args.ckpt),
              "summary": summaries, "results": results}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport: {output}")


if __name__ == "__main__":
    main()
