"""Catalog loading, duplicate-art merging, train/eval splits and datasets for Stage 1.

Splits (all decided per *class*, never per image):
  - held-out: ~5% of classes removed from training entirely -> tests generalization
    to cards the model never saw. Picks the best checkpoint.
  - demo-mirror: 200 classes the model DOES train on, queried with fresh augmented
    views it never saw -> mirrors the real demo, where demo cards are in training.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from embed.augment import HEIGHT, WIDTH, camera_augment, clean_view

# Eval query views use seeds from this range; training never seeds its RNG, so
# the model never sees these exact views.
EVAL_SEED_BASE = 10_000_000


def load_catalog(data_dir, set_prefixes):
    """catalog.csv rows whose set id starts with any of the given prefixes."""
    with open(Path(data_dir) / "catalog.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if any(r["set_id"].startswith(p) for p in set_prefixes)]


def average_hash(path, size=16):
    """Tiny fingerprint of an image: which pixels of a 16x16 grey thumbnail are above average."""
    with Image.open(path) as img:
        grey = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return np.packbits(grey > grey.mean()).tobytes()


def group_duplicates(rows, data_dir):
    """Merge cards with identical artwork (e.g. reprints, basic energy) into one class.

    Returns a list of classes; each class is a list of catalog rows. Telling such
    cards apart is left to OCR (name + collector number).
    """
    groups = defaultdict(list)
    for r in tqdm(rows, desc="hashing images", unit="card"):
        groups[average_hash(Path(data_dir) / r["low_path"])].append(r)
    classes = list(groups.values())
    merged = [c for c in classes if len(c) > 1]
    print(f"{len(rows)} cards -> {len(classes)} classes "
          f"({len(merged)} groups of identical art merged)")
    for c in merged[:5]:
        print("   e.g.", ", ".join(f"{r['name']} ({r['card_id']})" for r in c))
    return classes


def load_images(classes, data_dir):
    """Load one clean image per class into a single uint8 array [N, H, W, 3].

    One big array (rather than a list of PIL images) keeps memory shared between
    DataLoader worker processes.
    """
    images = np.empty((len(classes), HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i, c in enumerate(tqdm(classes, desc="loading images", unit="card")):
        with Image.open(Path(data_dir) / c[0]["low_path"]) as img:
            images[i] = np.asarray(img.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS))
    return images


def make_splits(classes, seed, holdout_frac, demo_size, demo_ids=None, use_holdout=True):
    """Split class indices into train / held-out / demo-mirror.

    demo_ids: optional set of card ids (the real demo list). Classes containing any
    of them form the demo-mirror set and are never held out.
    """
    rng = random.Random(seed)
    all_idx = list(range(len(classes)))

    if demo_ids:
        demo = [i for i in all_idx if any(r["card_id"] in demo_ids for r in classes[i])]
    else:
        demo = rng.sample(all_idx, min(demo_size, len(all_idx) // 4))

    demo_set = set(demo)
    candidates = [i for i in all_idx if i not in demo_set]
    n_holdout = round(holdout_frac * len(all_idx)) if use_holdout else 0
    held_out = rng.sample(candidates, n_holdout)

    held_set = set(held_out)
    train = [i for i in all_idx if i not in held_set]
    return {"train": train, "held_out": sorted(held_out), "demo": sorted(demo)}


class TrainDataset(Dataset):
    """(augmented view, label) pairs. Each card appears `views` times per epoch."""

    def __init__(self, images, class_indices, views=8):
        self.images = images
        self.class_indices = class_indices  # which rows of `images` to train on
        self.views = views

    def __len__(self):
        return len(self.class_indices) * self.views

    def __getitem__(self, i):
        label = i % len(self.class_indices)  # labels are 0..n_train-1
        view = camera_augment(self.images[self.class_indices[label]])  # fresh randomness
        return view, label


class RetrievalSet:
    """A demo-style retrieval test: clean gallery images + fixed augmented queries.

    Gallery i is the clean scan of class_indices[i]; each query's label is the
    gallery index it should match. Everything is uint8 to keep memory small.
    """

    def __init__(self, name, images, class_indices, views=5, seed_offset=0):
        self.name = name
        self.class_indices = class_indices
        self.gallery = torch.stack([clean_view(images[c]) for c in class_indices])
        queries, labels = [], []
        for g, c in enumerate(tqdm(class_indices, desc=f"building {name} queries", unit="card")):
            for v in range(views):
                seed = EVAL_SEED_BASE + seed_offset + c * 100 + v
                queries.append(camera_augment(images[c], random.Random(seed)))
                labels.append(g)
        self.queries = torch.stack(queries)
        self.query_labels = torch.tensor(labels)


def prepare(cfg, build_train=True):
    """Everything train_stage1 and evaluate need, rebuilt identically from a config dict.

    cfg keys: data, sets, seed, holdout_frac, demo_size, demo_list, no_holdout, limit_cards
    """
    rows = load_catalog(cfg["data"], cfg["sets"])
    if not rows:
        raise SystemExit(f"No cards for sets {cfg['sets']} in {cfg['data']}/catalog.csv")
    classes = group_duplicates(rows, cfg["data"])
    if cfg.get("limit_cards"):
        classes = random.Random(cfg["seed"]).sample(classes, min(cfg["limit_cards"], len(classes)))

    demo_ids = None
    if cfg.get("demo_list"):
        demo_ids = set(Path(cfg["demo_list"]).read_text().split())

    splits = make_splits(classes, cfg["seed"], cfg["holdout_frac"], cfg["demo_size"],
                         demo_ids, use_holdout=not cfg.get("no_holdout"))
    print(f"classes: {len(splits['train'])} train, {len(splits['held_out'])} held out, "
          f"{len(splits['demo'])} demo-mirror (subset of train)")

    images = load_images(classes, cfg["data"])
    eval_sets = [RetrievalSet("demo", images, splits["demo"])]
    if splits["held_out"]:
        eval_sets.append(RetrievalSet("held_out", images, splits["held_out"]))

    train_ds = TrainDataset(images, splits["train"], cfg.get("views", 8)) if build_train else None
    return classes, splits, train_ds, eval_sets
