"""2-D maps of the card embeddings, for the /embeddings page of the demo app.

Takes the cached reference index (one 256-d embedding per catalog card) and projects it:
    UMAP  -> connects each card to its nearest look-alikes, then lays the cards out in
             2-D so connected cards stay together. Local neighbours are meaningful;
             distances between far-apart clusters are not.
    PCA   -> best flat (linear) projection, as a baseline. Usually one blob.
    grid  -> the UMAP layout snapped onto a non-overlapping grid ("card wall"), so every
             card gets its own tile and similar cards stay next to each other.

Also stores each card's 10 nearest neighbours in the full 256-d space (what retrieval
actually sees) and k-means region labels, so 8k points don't read as one big clump.

Usage:
    python -m embed.index                 # first, if the index is not cached yet
    python -m embed.visualize             # writes checkpoints/embedding_map.json
    python -m embed.visualize --n-neighbors 30 --min-dist 0.5
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import umap

from embed.index import load_catalog_rows

CARD_ASPECT = 88 / 63  # height / width


def load_eras(data_dir):
    """set_id -> series name ("Sun & Moon", "Black & White", ...) from the cached TCGdex set files."""
    eras = {}
    for set_file in (Path(data_dir) / "sets").glob("*.json"):
        set_data = json.loads(set_file.read_text())
        eras[set_data["id"]] = set_data.get("serie", {}).get("name", "Other")
    return eras


def load_index(index_path, rows):
    cached = torch.load(index_path, map_location="cpu")
    if cached["key"]["card_ids"] != [r["card_id"] for r in rows]:
        raise SystemExit("index does not match catalog.csv; run: python -m embed.index")
    return torch.nn.functional.normalize(cached["embeddings"].float(), dim=1)


def pca_2d(emb):
    """Project onto the top-2 principal components. Returns ([N, 2], explained variance share)."""
    centered = emb - emb.mean(dim=0)
    torch.manual_seed(0)  # pca_lowrank is randomized; keep the numbers the same run to run
    _, s, v = torch.pca_lowrank(centered, q=2, center=False)
    explained = (s ** 2).sum() / (centered ** 2).sum()
    return (centered @ v[:, :2]).numpy(), explained.item()


def nearest_neighbors(emb, k=10, chunk=1024):
    """Top-k cosine neighbours of every card in the full embedding space (self excluded)."""
    result = []
    for start in range(0, len(emb), chunk):
        sims = emb[start:start + chunk] @ emb.T
        rows = torch.arange(start, min(start + chunk, len(emb)))
        sims[torch.arange(len(rows)), rows] = -2  # never return a card as its own neighbour
        result.append(sims.topk(k, dim=1).indices)
    return torch.cat(result).tolist()


def grid_layout(xy):
    """Snap 2-D points onto a grid, one card per cell: sort by y into equal rows, then by x."""
    n = len(xy)
    cols = math.ceil(math.sqrt(n * CARD_ASPECT))  # makes the wall roughly square on screen
    grid = np.zeros((n, 2), dtype=int)
    by_y = np.argsort(xy[:, 1])
    for row, start in enumerate(range(0, n, cols)):
        members = by_y[start:start + cols]
        members = members[np.argsort(xy[members, 0])]
        grid[members, 0] = np.arange(len(members))
        grid[members, 1] = row
    return grid


def kmeans(xy, k, iters=50, seed=0):
    """Plain k-means on the 2-D points. Returns the cluster index of each point."""
    rng = np.random.default_rng(seed)
    centres = xy[rng.choice(len(xy), k, replace=False)]
    for _ in range(iters):
        dists = ((xy[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
        for c in range(k):
            if (assign == c).any():
                centres[c] = xy[assign == c].mean(axis=0)
    return assign


LABEL_FIELDS = ("name", "set_name", "era", "rarity", "category")


def region_label(members, all_counts, n_total):
    """What makes this region stand out: its two most over-represented traits (from different fields).

    Score = share in region * log(share in region / share in whole catalog). A value
    that is common everywhere (e.g. "Pokemon") scores ~0; one concentrated here wins.
    Traits less than 1.3x as common here as in the whole catalog are not shown.
    """
    best = {}  # field -> (score, text)
    for field in LABEL_FIELDS:
        for value, count in Counter(c[field] for c in members).items():
            share = count / len(members)
            lift = share / (all_counts[field][value] / n_total)
            score = share * math.log(lift)
            if share >= 0.25 and lift >= 1.3 and score > best.get(field, (0.0, ""))[0]:
                best[field] = (score, f"{value} {share:.0%}")
    top = sorted(best.values(), reverse=True)[:2]
    return " · ".join(text for _, text in top)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/tcgdex")
    parser.add_argument("--index", default="checkpoints/index_full.pt")
    parser.add_argument("--out", default="checkpoints/embedding_map.json")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP: how many neighbours define 'local'")
    parser.add_argument("--min-dist", type=float, default=0.3, help="UMAP: how tightly points may pack")
    parser.add_argument("--spread", type=float, default=1.0, help="UMAP: overall scale of the layout")
    parser.add_argument("--regions", type=int, default=25, help="number of k-means region labels")
    args = parser.parse_args()

    rows = load_catalog_rows(args.data_dir)
    emb = load_index(args.index, rows)
    print(f"{len(rows)} cards, {emb.shape[1]}-d embeddings")

    pca_xy, explained = pca_2d(emb)
    print(f"PCA: the top 2 components explain {explained:.1%} of the variance")

    print("running UMAP (takes a minute)...")
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, spread=args.spread,
                        metric="cosine", random_state=0)
    umap_xy = reducer.fit_transform(emb.numpy())

    eras = load_eras(args.data_dir)
    cards = [{
        "card_id": r["card_id"], "name": r["name"], "set_id": r["set_id"], "set_name": r["set_name"],
        "category": r["category"], "rarity": r["rarity"] or "None", "era": eras.get(r["set_id"], "Other"),
    } for r in rows]
    cluster = kmeans(umap_xy, args.regions)
    all_counts = {field: Counter(c[field] for c in cards) for field in LABEL_FIELDS}
    regions = [region_label([cards[i] for i in np.flatnonzero(cluster == c)], all_counts, len(cards))
               if (cluster == c).any() else "" for c in range(args.regions)]

    out = {
        "cards": cards,
        "umap": np.round(umap_xy, 3).tolist(),
        "pca": np.round(pca_xy, 3).tolist(),
        "grid": grid_layout(umap_xy).tolist(),
        "neighbors": nearest_neighbors(emb),
        "cluster": cluster.tolist(),
        "regions": regions,
        "pca_explained": round(explained, 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
