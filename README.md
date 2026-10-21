# Pokémon Card Identification

A computer-vision class project. An overhead camera watches two people play the physical Pokémon Trading Card Game, and the system identifies each card as it is played. A UI shows every prediction, its confidence, and the reasoning behind it.

This is a proof-of-concept demo. The goal is a working, explainable end-to-end pipeline, not a production system.

## How it works

```
Video frame
   │
   ▼
Detect card outlines (OpenCV) → straighten each card → check the crop is usable
   │
   ├────────────────────────┐
   ▼                        ▼
Read text (OCR)       Embedding model (trained for this project)
   │                        │
   │                        ▼
   │                 Find the most similar reference cards
   │                        │
   │                        ▼
   │                 Shortlist of top-K candidates
   │                        │
   └──────────┬─────────────┘
              ▼
   Compare OCR text against the shortlist only
              ▼
   Combine both scores → final prediction → UI + log of played cards
```

1. **Detection.** Classic OpenCV with no machine learning. The program finds card-shaped rectangles on the mat, warps each one flat, and drops bad crops. A card counts as "played" once it stays in the same place for several frames.
2. **Embedding retrieval (main signal).** A model turns each card crop into a vector, and the closest vectors from a reference set of demo cards become the shortlist. The model is trained in two stages:
   - **Stage 1:** an ImageNet-pretrained backbone is trained as a classifier over thousands of cards, then its classification head is removed.
   - **Stage 2:** the result is fine-tuned with triplet loss.
3. **OCR (tie-breaker).** The program reads the card's name, and later its collector number, then compares that text against the shortlisted candidates only. This helps tell apart reprints of the same Pokémon.
4. **Fusion.** A simple weighted combination of the embedding score and the OCR score. If OCR fails, the embedding score is used alone.
5. **UI.** A Streamlit app shows the camera view with labelled card boxes, the top-3 candidates for each card with every score, and a timeline of played cards.

### Demo setup

- Fixed overhead camera, fixed lighting, and a plain solid-colour mat.
- At demo time the system only searches a reference set of about 100–300 physical cards. Training uses a much larger part of the official catalogue.

## Data

Card images and text come from [TCGdex](https://tcgdex.dev), which is free and needs no API key. Only the original images are stored. Training augmentations (perspective, blur, glare, colour shifts and so on, chosen to mimic the camera) are generated on the fly.

## Project status

| Milestone | Status |
|---|---|
| TCGdex download script | ✅ Done |
| Stage 1 classifier embedding (code, augmentations, retrieval eval) | ✅ Code done, full training run next |
| Detection, perspective warp, crop check, "played" logic | ⏳ Next |
| Reference index and evaluation on real photos | Planned |
| Stage 2 triplet fine-tuning, compared with Stage 1 | Planned |
| OCR tie-breaker | Planned |
| Fusion and Streamlit UI | Planned |

## Getting started

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### Download card data

```bash
# Specific sets (IDs from https://api.tcgdex.net/v2/en/sets)
python -m data_tools.download_tcgdex --sets sv01 base1

# Every Scarlet & Violet set, training images only (the high-res PNGs are only needed for OCR)
python -m data_tools.download_tcgdex --prefix sv --no-high

# Every set (several GB; add --no-high to skip the high-res PNGs)
python -m data_tools.download_tcgdex --all

# Summary plus a grid of sample cards (data/tcgdex/sample_grid.png)
python -m data_tools.check_download
```

Downloads are cached in `data/tcgdex/`, so re-running only fetches what is missing. Cards with no image or broken data are skipped and listed in `skipped.csv`; they don't stop the run.

| Output | Contents |
|---|---|
| `catalog.csv` | One row per card: ID, set, collector number, set size, name, category, rarity, attack names, image paths |
| `skipped.csv` | Cards that couldn't be downloaded, with the reason |
| `images/low/` | 245×337 WebP images for training |
| `images/high/` | 600×825 PNG images used as OCR references |

## Repository layout

```
data_tools/   TCGdex download and sanity-check scripts
embed/        embedding model, camera augmentations, Stage 1 training, retrieval eval
data/         downloaded data (not committed)
checkpoints/  trained weights (not committed)
```

Planned: `detect/`, `ocr/`, `fuse/` and `app/` for the remaining pipeline stages.

### Train the Stage 1 embedding

```bash
# Quick check that everything runs (about a minute)
python -m embed.train_stage1 --sets sv01 --epochs-a 1 --epochs-b 1 --views 2

# Full run on every downloaded sv set; writes checkpoints/stage1/
python -m embed.train_stage1

# Picture of each camera augmentation on its own (data/augment_effects.png)
python -m embed.augment --per-effect --cards 3
```

## Evaluation

The main measure is top-K retrieval accuracy and final accuracy on held-out **real photos** of the demo cards, searched against the demo-only reference set. Accuracy on augmented catalogue scans would overstate performance. Stage 1 and Stage 2 embeddings are compared on the same test.
