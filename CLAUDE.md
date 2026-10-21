# Project Briefing: Pokémon Card Identification (Computer Vision)

## What this is

A class project (proof-of-concept demo, single final deadline). An overhead camera watches two people playing the physical Pokémon Trading Card Game, and the system identifies which cards are played. The deliverable is a live or recorded demo with a UI that shows predictions, confidence, and the reasoning behind each one.

This is not a production system. Optimize for a working, explainable end-to-end pipeline over robustness or scale. When a choice is between "clever" and "easy to explain to a professor," pick the latter.

## Pipeline at a glance

```
Video frame
   │
   ▼
Detect card quads (OpenCV) → perspective-warp to canonical crop → crop QC
   │
   ├────────────────────────┐
   ▼                        ▼
OCR on crop           Embedding model (self-trained)
   │                        │
   │                        ▼
   │                 Cosine nearest-neighbour vs. reference index
   │                        │
   │                        ▼
   │                 Top-K candidate shortlist
   │                        │
   └──────────┬─────────────┘
              ▼
   OCR similarity vs. top-K reference strings only
              ▼
   Weighted fusion score → final prediction → UI + "played" event log
```

Three subsystems: (1) detection and continuity, (2) embedding retrieval as the primary signal, (3) OCR as a tie-breaker, combined by a simple fusion step.

## Scope

- **Demo reference set:** ~100–300 physical cards (exact list not chosen yet). At inference, retrieval searches only this set.
- **Training set:** much larger — ideally thousands of cards from the official catalog, including all demo cards (see Training).
- **Environment:** fixed overhead camera, fixed lighting, plain high-contrast mat (solid black or green).
- **Out of scope:** arbitrary lighting/backgrounds, hand tracking, occlusion recovery, player attribution, multi-camera, identifying against the full catalog at inference.

## Data

**Source: TCGdex** (https://tcgdex.dev) — free, no API key, card text and images in one place, Python SDK or plain REST. Do not use pokemontcg.io; it's now part of Scrydex and no longer free.

- Images: append a quality/format suffix to the card's image base URL, e.g. `/high.png` or `/low.webp`. Use a training-sized image plus a high-res one for OCR reference.
- Field names: collector number is `localId`, set total is the set's `cardCount`, attack text is `effect`.
- Some cards (especially older sets) lack an image or fields. The download script must **skip and log** these, not crash.
- Be polite: short sleep between downloads, cache everything locally, pull the catalog once.
- Order: first pull the sets the demo cards come from and sanity-check images and text, then pull the wider catalog.

**Storage:** store originals only. Augmentations are generated on the fly in the data loader, not saved to disk. The full catalog fits in roughly 1–3 GB.

## Detection & continuity (`detect/`)

- Classic OpenCV, no ML: grayscale → threshold/Canny → `findContours` → filter by rectangularity and card aspect ratio (~2.5:3.5, i.e. 63×88 mm) → four-corner perspective warp to a canonical rectangle.
- **"Played" event:** a detected card region that stays in a stable position for N consecutive frames. Sampling a frame periodically and re-running detection is fine; no true multi-object tracking needed.
- **Crop QC:** a bad crop poisons both OCR and the embedding at once, so add a cheap sanity check (aspect ratio, edge/area checks) and drop or flag bad crops.

## Embedding model (`embed/`)

Self-trained on purpose (the learning experience is part of the assignment) — do not substitute CLIP or another off-the-shelf embedding as the final model. Built in two stages.

### Stage 1: classifier, then strip the head

Architecture:
1. **ImageNet-pretrained backbone** (ResNet-50, EfficientNet, or similar) with its ImageNet head removed; global average pooling.
2. **Embedding layer:** a new linear layer, 128–512 dims (start at 256), **no activation** (a ReLU here zeros negatives and distorts cosine similarity). Dropout before it.
3. **Classification head:** one output per training card. A cosine-softmax head (ArcFace/CosFace) is preferred over plain softmax given many classes with few source images each.

Training:
- Phase A: freeze backbone, train only layers 2–3 for a few epochs.
- Phase B: unfreeze some or all of the backbone at a 10–100× lower learning rate.
- Moderate weight decay. 99% train accuracy is not the goal; general embeddings are.
- Optional short final fine-tune at low LR weighted toward demo cards and any real photos. Keep it short so the model doesn't narrow to the small set.

After training, discard layer 3. Backbone + embedding layer is the embedding model. **L2-normalize** every embedding.

### Stage 2: triplet fine-tuning (bootstrapped from Stage 1)

- Start from the Stage 1 weights. Its embedding space already places look-alike cards near each other, so mined negatives are meaningful from step one.
- Loss: `max(0, d(a,p) - d(a,n) + margin)` on normalized embeddings, margin ~0.2–0.5.
- **Online mining** (batch-hard or semi-hard) with **P×K batch sampling** (P cards, K samples each) so every anchor has in-batch positives. Augment each sample independently; optionally include 2+ augmented views of the same source image as extra positives.
- With batch-all or random positives, watch for easy self-positives (two augmentations of one image) dominating the loss.
- Re-mine from the model's own live embeddings; the classifier is only a bootstrap.
- If triplet training won't converge, contrastive loss or the Stage 1 classifier embedding are acceptable fallbacks. Prototype early rather than committing blindly.
- For reproducibility, log source image IDs plus augmentation seeds/params for interesting triplets rather than saving augmented images.

### Augmentation (the main risk is the domain gap)

Training images are clean, flat scans; the demo input is overhead camera crops. Augmentations must mimic the camera: perspective warp, blur, glare/hotspots, white-balance shifts, JPEG compression, partial occlusion, small crop misalignment. Mix in real photos of demo cards when available.

Augmentations must be label-preserving: no horizontal flips (text), no crops that cut off the card.

### Reference index

One embedding per demo card, computed from its **clean reference image** (not an augmented photo), tagged with card identity. Adding a card = embed one more image, no retraining. Retrieval uses cosine similarity, returning top-K (K tuned empirically: too small drops the right card before OCR sees it, too large dilutes OCR).

## OCR tie-break (`ocr/`)

- Tesseract or EasyOCR on the warped crop, targeting the **name** field first.
- For each reference card build a reference string from TCGdex fields: start with name; stretch goal adds collector number as `localId/cardCount`. Name + number separates the same Pokémon across sets and rarities — the embedding model's hardest case. Attack names are a backup signal if the name is unreadable.
- Compare OCR output **only against the top-K shortlist**, not the whole catalog.
- Failed or empty OCR sets `ocr_available = False` explicitly. Never impute a neutral score.

## Fusion (`fuse/`)

- Weighted combination of embedding cosine similarity and OCR string similarity per candidate (or a tiny logistic regression). No complex stacking model.
- If `ocr_available` is false, fall back to the embedding score alone.
- Every prediction must keep its component scores so the UI can show them.

## UI (`app/`)

Streamlit is the default.
- Live/replay view with card boxes overlaid, labeled with predicted name + confidence.
- Per-card panel: top-3 candidates with embedding score, OCR score, and combined score. The point is to show the reasoning trail (top-K → OCR tie-break → final), not a black-box label.
- Timeline/log of "played" events.
- Runs without manual intervention once cards are on the mat.

## Evaluation rules

- The metric that matters: **top-K retrieval and final accuracy on held-out real photos of demo cards**, searched against a demo-only reference index. Accuracy on augmented official scans overstates performance.
- **Split discipline:** all augmented variants of a card stay on the same side of any split. Never let variants of one source image leak across train/val.
- Report Stage 1 vs. Stage 2 embeddings on the same eval so the triplet step's value is visible.

## Milestones (work back from the single final deadline)

1. Data download script (TCGdex, cached, skip-and-log) + detection/warp/QC + "played" logic on the real mat setup.
2. Stage 1 classifier-embedding trained; reference index built; top-K validated on held-out real photos.
3. Stage 2 triplet fine-tuning; compare against Stage 1.
4. OCR tie-break including the missing-OCR path.
5. Fusion + Streamlit UI end-to-end on live or recorded footage.
6. Dry run and buffer for lighting, camera angle, timing.

## Not yet decided (ask before assuming)

- Exact demo card list and which sets they come from.
- Deadline date.
- Framework: not settled. Default to PyTorch + timm unless told otherwise.
- Backbone choice, embedding size, margin, K, and fusion weights — all to be tuned empirically.
- Whether OCR goes beyond the name field (depends on how many near-identical reprints are in the demo set).
- How much of the full catalog to train on given compute/time (a few thousand cards weighted toward the demo sets is the fallback).

## Working conventions for Claude Code

- Keep each stage runnable and testable on its own (detection on a still image, embedding on a folder, OCR on a crop).
- Prefer simple, readable code over clever abstractions; the owner needs to explain all of it in class.
- Don't commit downloaded images or model weights; cache under a gitignored `data/` and `checkpoints/`.
