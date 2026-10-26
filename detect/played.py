"""Decide when a card has been "played": it sits still in the same spot for N sampled frames.

No real multi-object tracking. Each sampled frame, every detection that passed crop QC
is matched to the nearest card from the previous frame (by centre distance). A card
that stays within a small distance for `stable_frames` frames in a row fires one
"played" event. A card that moves starts counting again; a card missing for a few
frames is forgotten, so picking it up and putting it down again counts as a new play.

Usage:
    python -m detect.played game.mp4                        # print played events
    python -m detect.played game.mp4 --crops-dir data/real_raw   # also save each played card's crop
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    track_id: int
    centre: np.ndarray
    detection: object           # latest Detection for this card
    stable_count: int = 1
    missed: int = 0
    played: bool = False
    history: list = field(default_factory=list)


class PlayedTracker:
    def __init__(self, stable_frames=5, max_move_px=15.0, forget_after=3):
        self.stable_frames = stable_frames  # sampled frames a card must stay still
        self.max_move_px = max_move_px      # how far its centre may drift and still count as still
        self.forget_after = forget_after    # missed frames before a track is dropped
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """Feed one sampled frame's detections. Returns the tracks that became "played" now."""
        unmatched = [d for d in detections if d.ok]
        newly_played = []

        for track in self.tracks:
            if not unmatched:
                track.missed += 1
                continue
            dists = [np.linalg.norm(d.centre - track.centre) for d in unmatched]
            best = int(np.argmin(dists))
            # A card that moved further than a card's width is a different card.
            if dists[best] > unmatched[best].short_side:
                track.missed += 1
                continue
            det = unmatched.pop(best)
            track.stable_count = track.stable_count + 1 if dists[best] <= self.max_move_px else 1
            track.centre, track.detection, track.missed = det.centre, det, 0
            if not track.played and track.stable_count >= self.stable_frames:
                track.played = True
                newly_played.append(track)

        self.tracks = [t for t in self.tracks if t.missed <= self.forget_after]
        for det in unmatched:  # cards seen for the first time
            self.tracks.append(Track(self.next_id, det.centre, det))
            self.next_id += 1
        return newly_played


def main():
    import argparse
    from pathlib import Path

    import cv2

    from detect.cards import detect_cards

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video file, or a camera index such as 0")
    parser.add_argument("--every", type=int, default=5, help="run detection on every k-th frame")
    parser.add_argument("--stable", type=int, default=5, help="sampled frames a card must stay still")
    parser.add_argument("--max-move", type=float, default=15.0, help="pixels a still card may drift")
    parser.add_argument("--crops-dir", help="save the warped crop of each played card here")
    args = parser.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    tracker = PlayedTracker(stable_frames=args.stable, max_move_px=args.max_move)

    frame_idx, n_played = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every == 0:
            for track in tracker.update(detect_cards(frame)):
                n_played += 1
                x, y = track.centre
                print(f"{frame_idx / fps:7.2f}s  card {track.track_id} played at ({x:.0f}, {y:.0f})")
                if args.crops_dir:
                    Path(args.crops_dir).mkdir(parents=True, exist_ok=True)
                    path = Path(args.crops_dir) / f"frame{frame_idx:06d}_card{track.track_id}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(track.detection.crop, cv2.COLOR_RGB2BGR))
        frame_idx += 1
    print(f"{frame_idx} frames, {n_played} cards played")


if __name__ == "__main__":
    main()
