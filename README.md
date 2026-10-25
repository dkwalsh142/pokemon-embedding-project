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
| Detection, perspective warp, crop check, "played" logic | ⏳ Next |
| Stage 1 classifier embedding and reference index | Planned |
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
data/         downloaded data (not committed)
```

Planned: `detect/`, `embed/`, `ocr/`, `fuse/` and `app/` for the remaining pipeline stages.

## Evaluation

The main measure is top-K retrieval accuracy and final accuracy on held-out **real photos** of the demo cards, searched against the demo-only reference set. Accuracy on augmented catalogue scans would overstate performance. Stage 1 and Stage 2 embeddings are compared on the same test.

### Compare embedding retrieval with OCR fusion

Requires a trained `checkpoints/stage1/embedder.pt`, downloaded cards, a demo
list of catalog IDs (one per line), and Tesseract with English language data.
On macOS, install the OCR engine with `brew install tesseract`; the Python
wrapper is included in `requirements.txt`.

Run deterministic synthetic diagnostics against the demo gallery:

```bash
python -m fuse.evaluate --demo-list data/demo_cards.txt --views 3
```

Each card gets a clean control, three seeded camera-like degraded views, a view
with most artwork covered (severe occlusion stress test), and a view with its
name band covered. Queries are generated temporarily; original
images are unchanged. The report records the source image, condition, and seed.
These use the training augmenter and are diagnostic proxies, not real-photo
accuracy. Use `--seed` to repeat with different degradations.

The command prints embedding top-1, top-K recall, fused top-1, OCR fixes,
regressions, and whether missing OCR preserves the full embedding ranking.
It explicitly counts ambiguous cases where embedding top-1 is wrong but the
correct card remains in the shortlist. If there are none, the test cannot show
whether OCR rescues mistakes. Incorrect answers absent from top-K cannot be
recovered by fusion. Covered text can occasionally produce OCR garbage; the
report shows actual OCR availability rather than assuming extraction failed.

For real photos, first crop and straighten each card, then create a CSV such as
`data/photo_tests.csv` (paths relative to that CSV):

```csv
image_path,card_id,condition
photos/scovillain_01.jpg,sv01-202,real_photo
photos/scovillain_blur.jpg,sv01-202,blurred
photos/scovillain_name_hidden.jpg,sv01-202,name_hidden
```

Use actual files and independently assigned ground-truth card IDs. Every labeled
card must belong to the demo gallery; query photos are never added to it.

```bash
python -m fuse.evaluate --demo-list data/demo_cards.txt \
  --manifest data/photo_tests.csv --out data/real_photo_eval.json
```

`--k 5` and `--ocr-weight 0.2` are defaults. Keep K fixed when comparing weights;
use a separate validation collection for tuning and leave final test photos
untouched. The default JSON report is `data/fusion_eval.json`; it contains all
queries, OCR text and errors, candidate metadata and scores, and aggregate
metrics by condition. Use different `--out` paths to retain multiple runs.

Automated metric and fusion checks:

```bash
python -m unittest fuse.test_rank fuse.test_evaluate
```
