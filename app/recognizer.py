"""Frame -> detected cards -> top-K catalog matches -> "played" events.

Ties the stages together without any web code, so it can be tested on a still image:
    detect_cards (OpenCV) -> embed each crop -> cosine similarity vs. the reference
    index -> top-K candidates -> PlayedTracker decides when a card counts as played.

OCR is not built yet, so every candidate has ocr_available=False and its combined
score is just the embedding score. The fields are already there for the fusion step.

Usage:
    python -m app.recognizer photo.jpg        # print the top-3 for each card in a still photo
    python -m app.recognizer photo.jpg --normalize   # same, after normalizing the lighting
"""

import time

import cv2
import numpy as np
import torch

from detect.cards import MAX_AREA_FRAC, detect_cards, order_corners, warp
from detect.lighting import normalize_lighting
from detect.played import PlayedTracker
from embed.evaluate import embed_all
from embed.index import image_to_uint8, load_embedder, load_or_build_index


class Recognizer:
    def __init__(self, ckpt="checkpoints/stage1/embedder.pt", data_dir="data/tcgdex",
                 top_k=3, unsure_below=0.6, stable_frames=5, max_move_px=15.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_embedder(ckpt, self.device)
        self.rows, embeddings = load_or_build_index(self.model, ckpt, data_dir, self.device)
        self.index = embeddings.to(self.device)
        self.rows_by_id = {r["card_id"]: r for r in self.rows}
        self.top_k = top_k
        self.unsure_below = unsure_below  # top-1 score under this is flagged "uncertain"
        self.tracker_args = dict(stable_frames=stable_frames, max_move_px=max_move_px)
        self.reset()

    def reset(self):
        self.tracker = PlayedTracker(**self.tracker_args)
        self.events = []

    def match(self, crops):
        """List of RGB crops -> list of top-K candidate lists.

        A card the other player put down faces the other way, and order_corners then
        returns it rotated 180 degrees. So we embed each crop both ways round and keep
        the better similarity for every catalog card.
        """
        if not crops:
            return []
        upright = torch.stack([image_to_uint8(c) for c in crops])
        flipped = torch.flip(upright, dims=[2, 3])  # 180-degree rotation
        sims = torch.maximum(embed_all(self.model, upright, self.device) @ self.index.T,
                             embed_all(self.model, flipped, self.device) @ self.index.T)
        scores, idx = sims.topk(self.top_k, dim=1)
        results = []
        for row_scores, row_idx in zip(scores.tolist(), idx.tolist()):
            candidates = []
            for score, i in zip(row_scores, row_idx):
                card = self.rows[i]
                candidates.append({
                    "card_id": card["card_id"],
                    "name": card["name"],
                    "set_name": card["set_name"],
                    "number": f"{card['local_id']}/{card['set_card_count']}",
                    "embedding_score": round(score, 4),
                    "ocr_score": None,
                    "ocr_available": False,
                    "combined": round(score, 4),  # embedding only until OCR + fusion exist
                })
            results.append(candidates)
        return results

    def detect_and_match(self, frame_bgr, max_area_frac=MAX_AREA_FRAC):
        """Detections in the frame, plus the top-K candidates for each one that passed QC."""
        detections = detect_cards(frame_bgr, max_area_frac)
        good = [d for d in detections if d.ok]
        top_by_det = {id(d): top for d, top in zip(good, self.match([d.crop for d in good]))}
        return detections, top_by_det

    def identify(self, image_bgr, drawn_box=False):
        """Still image (e.g. an uploaded photo) -> every card in it with its top-K. No tracker.

        If no card outline is found, the whole image is treated as one card, so a tight
        close-up or a scan still gets an answer. Each card includes its warped crop
        (RGB array) so the UI can show what was actually compared.

        drawn_box=True means the image is a box the user drew around one card, so the
        card may fill nearly all of it.
        """
        detections, top_by_det = self.detect_and_match(image_bgr, 0.98 if drawn_box else MAX_AREA_FRAC)
        cards = [{
            "corners": det.corners.round(1).tolist(),
            "ok": det.ok,
            "reason": det.reason,
            "whole_image": False,
            "top": top_by_det.get(id(det), []),
            "crop": det.crop,
        } for det in detections]

        if not any(card["ok"] for card in cards):
            h, w = image_bgr.shape[:2]
            corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            crop = warp(image_bgr, corners)
            cards.append({
                "corners": corners.tolist(),
                "ok": True,
                "reason": "",
                "whole_image": True,
                "top": self.match([crop])[0],
                "crop": crop,
            })
        return cards

    def identify_corners(self, image_bgr, corners):
        """The user clicked (or rotated) a card's 4 corners, in any order: warp exactly that quad."""
        corners = order_corners(np.float32(corners))  # TL, TR, BR, BL, like detected cards
        crop = warp(image_bgr, corners)
        return [{
            "corners": corners.round(1).tolist(),
            "ok": True,
            "reason": "",
            "whole_image": False,
            "top": self.match([crop])[0],
            "crop": crop,
        }]

    def process(self, frame_bgr):
        """One frame -> {"cards": [...], "played": [events that fired on this frame]}."""
        detections, top_by_det = self.detect_and_match(frame_bgr)

        newly_played = self.tracker.update(detections)
        track_by_det = {id(t.detection): t for t in self.tracker.tracks if t.missed == 0}

        cards = []
        for det in detections:
            track = track_by_det.get(id(det))
            cards.append({
                "corners": det.corners.round(1).tolist(),
                "ok": det.ok,
                "reason": det.reason,
                "track_id": track.track_id if track else None,
                "top": top_by_det.get(id(det), []),
            })

        played = []
        for track in newly_played:
            top = top_by_det[id(track.detection)]
            event = {
                "event_id": len(self.events),
                "time": time.time(),
                "track_id": track.track_id,
                "top": top,
                "uncertain": top[0]["combined"] < self.unsure_below,
            }
            self.events.append(event)
            played.append(event)
        return {"cards": cards, "played": played}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    parser.add_argument("--normalize", action="store_true", help="normalize lighting first (detect/lighting.py)")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"could not read {args.image}")
    if args.normalize:
        frame = normalize_lighting(frame)
    for i, card in enumerate(Recognizer(args.ckpt).identify(frame)):
        if not card["ok"]:
            print(f"card {i}: dropped ({card['reason']})")
            continue
        where = " (no outline found; matched the whole image)" if card["whole_image"] else ""
        print(f"card {i}{where}: " + " | ".join(
            f"{c['name']} {c['card_id']} {c['combined']:.2f}" for c in card["top"]))


if __name__ == "__main__":
    main()
