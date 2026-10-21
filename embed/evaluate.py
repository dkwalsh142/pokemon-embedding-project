"""Top-K retrieval evaluation, shared by Stage 1 and Stage 2 so both are scored identically.

For each query (augmented view) we rank the gallery (clean scans) by cosine
similarity and check whether the correct card is among the K closest.

Usage:
    python -m embed.evaluate --ckpt checkpoints/stage1/embedder.pt
"""

import argparse

import torch

from embed.augment import normalize


@torch.no_grad()
def embed_all(model, images_uint8, device, batch_size=256):
    """Embed a uint8 image tensor [N, 3, H, W] in batches -> [N, D] normalized embeddings."""
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        batch = normalize(images_uint8[i:i + batch_size].to(device))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            emb = model.embed(batch)
        out.append(emb.float())
    return torch.nn.functional.normalize(torch.cat(out), dim=1)


@torch.no_grad()
def retrieval_eval(model, rset, device, ks=(1, 5, 10)):
    """Return {"top1": ..., "top5": ..., "top10": ...} for one RetrievalSet."""
    gallery = embed_all(model, rset.gallery, device)
    queries = embed_all(model, rset.queries, device)
    sims = queries @ gallery.T  # cosine similarity, since both sides are normalized
    ranked = sims.topk(max(ks), dim=1).indices.cpu()
    hits = ranked == rset.query_labels.view(-1, 1)
    return {f"top{k}": hits[:, :k].any(dim=1).float().mean().item() for k in ks}


def main():
    from embed.data import prepare
    from embed.model import CardEmbedder

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CardEmbedder(emb_dim=cfg["emb_dim"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    _, _, _, eval_sets = prepare(cfg, build_train=False)
    for rset in eval_sets:
        scores = retrieval_eval(model, rset, device)
        print(f"{rset.name:9s} ({len(rset.gallery)} cards, {len(rset.queries)} queries): "
              + "  ".join(f"{k} {v:.1%}" for k, v in scores.items()))


if __name__ == "__main__":
    main()
