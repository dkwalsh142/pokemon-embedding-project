"""Stage 1: train ResNet-50 + embedding layer as an ArcFace classifier, then keep the embedder.

Phase A: backbone frozen, train only the embedding layer and ArcFace head.
Phase B: unfreeze the backbone at a 10x lower learning rate than the new layers.

After every epoch we run two retrieval evals (see embed/data.py):
  demo     - cards in training, new augmented views (mirrors the real demo)
  held_out - cards never trained on (generalization; picks best.pt)

Usage:
    python -m embed.train_stage1                                   # full run on all sv sets
    python -m embed.train_stage1 --limit-cards 300 --epochs-a 1 --epochs-b 1   # quick dry run
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embed.augment import normalize
from embed.data import prepare
from embed.evaluate import retrieval_eval
from embed.model import ArcFaceHead, CardEmbedder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data", default="data/tcgdex")
    p.add_argument("--sets", nargs="+", default=["sv"], help="set id prefixes to train on")
    p.add_argument("--views", type=int, default=8, help="augmented views of each card per epoch")
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--no-holdout", action="store_true", help="train on every card (final run)")
    p.add_argument("--demo-size", type=int, default=200)
    p.add_argument("--demo-list", help="text file of card ids (the real demo cards)")
    p.add_argument("--limit-cards", type=int, help="use only N classes (quick tests)")
    # model
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--arc-s", type=float, default=30.0)
    p.add_argument("--arc-m", type=float, default=0.3)
    # training
    p.add_argument("--epochs-a", type=int, default=3)
    p.add_argument("--epochs-b", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/stage1")
    return p.parse_args()


def worker_init(_):
    torch.set_num_threads(1)  # each DataLoader worker augments on one core


def train_one_epoch(model, head, loader, optimizer, scheduler, device, backbone_frozen):
    model.train()
    if backbone_frozen:
        model.backbone.eval()  # also freeze BatchNorm running statistics
    total_loss, correct, seen = 0.0, 0, 0
    for views, labels in loader:
        x = normalize(views.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            emb = model(x)
        logits = head(emb, labels)  # ArcFace runs in float32
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        with torch.no_grad():  # accuracy without the margin, i.e. nearest class centre
            correct += (head(emb.detach()).argmax(1) == labels).sum().item()
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    return total_loss / seen, correct / seen


def main():
    args = parse_args()
    cfg = vars(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    classes, splits, train_ds, eval_sets = prepare(cfg)
    (out / "split.json").write_text(json.dumps(
        {name: [[r["card_id"] for r in classes[i]] for i in idx] for name, idx in splits.items()}, indent=1))

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True,
                        worker_init_fn=worker_init)

    model = CardEmbedder(emb_dim=args.emb_dim).to(device)
    head = ArcFaceHead(args.emb_dim, len(splits["train"]), s=args.arc_s, m=args.arc_m).to(device)
    new_params = list(model.embedding.parameters()) + list(head.parameters())

    log_path = out / "log.csv"
    log_file = open(log_path, "w", newline="")
    log = csv.writer(log_file)
    metric_cols = [f"{r.name}_{k}" for r in eval_sets for k in ("top1", "top5", "top10")]
    log.writerow(["epoch", "phase", "loss", "train_acc", *metric_cols, "seconds"])

    best_score = -1.0
    total_epochs = args.epochs_a + args.epochs_b
    optimizer = scheduler = None
    for epoch in range(1, total_epochs + 1):
        phase = "A" if epoch <= args.epochs_a else "B"

        if epoch == 1:  # Phase A: only the new layers learn
            model.freeze_backbone(True)
            optimizer = torch.optim.AdamW(new_params, lr=args.lr_head, weight_decay=args.weight_decay)
        if epoch == args.epochs_a + 1:  # Phase B: whole network, backbone at a lower LR
            model.freeze_backbone(False)
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr_backbone},
                {"params": new_params, "lr": args.lr_head},
            ], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs_b * len(loader))

        start = time.time()
        loss, acc = train_one_epoch(model, head, loader, optimizer, scheduler, device,
                                    backbone_frozen=(phase == "A"))
        scores = {f"{r.name}_{k}": v for r in eval_sets for k, v in retrieval_eval(model, r, device).items()}
        seconds = time.time() - start

        log.writerow([epoch, phase, f"{loss:.4f}", f"{acc:.4f}",
                      *[f"{scores[c]:.4f}" for c in metric_cols], f"{seconds:.0f}"])
        log_file.flush()
        print(f"epoch {epoch:2d}/{total_epochs} [{phase}] loss {loss:.3f}  train acc {acc:.1%}  | "
              + "  ".join(f"{r.name} top1 {scores[r.name + '_top1']:.1%} top5 {scores[r.name + '_top5']:.1%}"
                          for r in eval_sets)
              + f"  ({seconds:.0f}s)")

        state = {"model": model.state_dict(), "head": head.state_dict(), "epoch": epoch, "config": cfg}
        torch.save(state, out / "last.pt")

        # Choose the best checkpoint on held-out cards: unlike the demo-mirror score
        # it does not saturate. With --no-holdout the last epoch is kept.
        score = scores.get("held_out_top1", epoch)
        if score > best_score:
            best_score = score
            torch.save(state, out / "best.pt")
            # The Stage 1 deliverable: backbone + embedding layer only (ArcFace head dropped).
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, out / "embedder.pt")

    log_file.close()
    best = torch.load(out / "embedder.pt", map_location="cpu")["epoch"]
    print(f"\nDone. Best epoch {best}; embedder saved to {out / 'embedder.pt'}, log in {log_path}")
    print("Note: both scores use the training augmenter, so they are proxies; "
          "real photos of the demo cards are the final metric.")


if __name__ == "__main__":
    main()
