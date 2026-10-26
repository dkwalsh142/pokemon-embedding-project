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
| Detection, perspective warp, crop check, "played" logic | ✅ Works on a dark mat; weak on white backgrounds |
| Reference index and evaluation on real photos | 🟡 Index done; 17/17 top-1 on a first handful of real photos |
| Stage 2 triplet fine-tuning, compared with Stage 1 | Planned |
| OCR tie-breaker | Planned |
| Live webcam UI (browser + Flask) | ✅ First version, embedding only |
| Fusion of embedding and OCR scores | Planned |

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
embed/        embedding model, camera augmentations, Stage 1 training, retrieval eval, reference index
detect/       card detection, perspective warp, crop QC, "played" tracker
app/          live webcam demo: recognizer (frame -> predictions), Flask server, browser page
data/         downloaded data (not committed)
checkpoints/  trained weights (not committed)
```

Planned: `ocr/` and `fuse/` for the remaining pipeline stages.

### Train the Stage 1 embedding

```bash
# Quick check that everything runs (about a minute)
python -m embed.train_stage1 --sets sv01 --epochs-a 1 --epochs-b 1 --views 2

# Full run on every downloaded sv set; writes checkpoints/stage1/
python -m embed.train_stage1

# Picture of each camera augmentation on its own (data/augment_effects.png)
python -m embed.augment --per-effect --cards 3
```

### Live webcam demo

The browser reads the webcam and sends frames to a small Flask server, which runs detection and retrieval on the GPU. Each card is outlined with its best match, and once a card has stayed still for a few frames it pops into the "Played cards" sidebar with its top-3 candidates.

```bash
# Optional: embed every catalog card up front (otherwise the server does it on first start)
python -m embed.index

# Check the recognizer on a still photo
python -m app.recognizer photo.jpg

# Start the server, then open http://localhost:8501 and allow camera access
python -m app.server
```

You can also drag photos onto the page (or use "Identify images…"). Every card found is boxed and listed in the sidebar with its crop, best match and top-3. To fix or add boxes, click empty space to add a card-shaped box (locked to the 63:88 card ratio). Drag a box to move it, drag ↻ to rotate it, or drag a corner to resize it. Each box is identified again when you let go. × or Delete removes a box.

When running on a remote machine over SSH, forward the port (VS Code does this automatically, or `ssh -L 8501:localhost:8501 host`). Open the page as `http://localhost:8501`, not by IP address: browsers only allow camera access on localhost or https.

Known limitations:
- Detection expects a dark, plain surface. On white paper it misses most cards.
- Cards with no TCGdex image (for example the plain basic energies) are not in the index and get matched confidently to a look-alike. The collector-number OCR step should fix this.
- Scores below 0.6 are flagged "uncertain". OCR is not built yet, so the combined score is the embedding score.

## Evaluation

The main measure is top-K retrieval accuracy and final accuracy on held-out **real photos** of the demo cards, searched against the demo-only reference set. Accuracy on augmented catalogue scans would overstate performance. Stage 1 and Stage 2 embeddings are compared on the same test.
