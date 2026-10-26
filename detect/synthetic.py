"""Fake overhead frames, so detection can be tested before the real camera setup exists.

Pastes catalog scans onto a plain mat at random positions, angles (including upside
down) and sizes, with a little perspective, blur and noise. Ground-truth corners are
known, so we can measure how many cards the detector finds and how accurately.

Usage:
    python -m detect.synthetic --scenes 100            # detector recall + corner error
    python -m detect.synthetic --video data/fake_game.mp4   # clip of cards being played one by one
"""

import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 1920, 1080
MATS = {"black": (12, 12, 12), "green": (40, 90, 25)}  # BGR


def load_cards(data_dir, n, rng):
    with open(Path(data_dir) / "catalog.csv", newline="") as f:
        rows = rng.sample(list(csv.DictReader(f)), n)
    images = []
    for row in rows:
        img = cv2.imread(str(Path(data_dir) / row["low_path"]))  # Pillow-free; OpenCV reads WebP
        images.append((row["card_id"], img))
    return images


def card_corners(centre, long_side, angle_deg, rng, skew=0.03):
    """Corners (card's own TL, TR, BR, BL) of a card placed on the mat, with slight perspective."""
    w, h = long_side * 63 / 88, long_side
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    local += np.array([[rng.uniform(-skew, skew) * w, rng.uniform(-skew, skew) * h] for _ in range(4)])
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (local @ rot.T + centre).astype(np.float32)


def paste(frame, card_bgr, corners):
    """Warp the card image into the frame at `corners`."""
    h, w = card_bgr.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, corners)
    size = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(card_bgr, matrix, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, size)
    frame[mask > 0] = warped[mask > 0]


def random_layout(n, rng, sizes=(180, 380)):
    """Non-overlapping (centre, long_side, angle) for n cards."""
    placed = []
    for _ in range(200 * n):
        if len(placed) == n:
            break
        long_side = rng.uniform(*sizes)
        r = long_side * 0.62  # bounding circle radius
        centre = np.array([rng.uniform(r, FRAME_W - r), rng.uniform(r, FRAME_H - r)])
        if all(np.linalg.norm(centre - c) > r + pr + 10 for c, _, _, pr in placed):
            placed.append((centre, long_side, rng.uniform(0, 360), r))
    return [(c, s, a) for c, s, a, _ in placed]


def camera_noise(frame, rng):
    frame = cv2.GaussianBlur(frame, (0, 0), rng.uniform(0.5, 1.5))
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, 3, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def make_scene(cards, rng, mat="green"):
    """Frame with all `cards` (list of (card_id, image)) laid out. Returns frame, ground truth."""
    frame = np.full((FRAME_H, FRAME_W, 3), MATS[mat], np.uint8)
    truth = []
    for (card_id, img), (centre, size, angle) in zip(cards, random_layout(len(cards), rng)):
        corners = card_corners(centre, size, angle, rng)
        paste(frame, img, corners)
        truth.append((card_id, corners))
    return camera_noise(frame, rng), truth


def corner_error(found, true):
    """Mean corner distance in px. The detector can't tell up from down, so try both."""
    flipped = np.roll(true, 2, axis=0)
    return min(np.linalg.norm(found - true, axis=1).mean(), np.linalg.norm(found - flipped, axis=1).mean())


def evaluate(args):
    from detect.cards import detect_cards, draw_overlay

    rng = random.Random(args.seed)
    n_true = n_found = n_ok = n_false = 0
    errors = []
    for i in range(args.scenes):
        cards = load_cards(args.data, rng.randint(2, 8), rng)
        frame, truth = make_scene(cards, rng, mat=rng.choice(list(MATS)))
        dets = detect_cards(frame)
        used = set()
        for _, corners in truth:
            n_true += 1
            dists = [np.linalg.norm(d.centre - corners.mean(axis=0)) for d in dets]
            j = int(np.argmin(dists)) if dists else -1
            if j >= 0 and dists[j] < 20 and j not in used:
                used.add(j)
                n_found += 1
                n_ok += dets[j].ok
                errors.append(corner_error(dets[j].corners, corners))
        n_false += len(dets) - len(used)
        if i == 0:
            cv2.imwrite(args.overlay, draw_overlay(frame, dets))

    print(f"{args.scenes} scenes, {n_true} cards")
    print(f"  found:          {n_found / n_true:.1%}")
    print(f"  found + QC ok:  {n_ok / n_true:.1%}")
    print(f"  false detections: {n_false}")
    if errors:
        print(f"  corner error: mean {np.mean(errors):.1f}px, worst {np.max(errors):.1f}px")
    print(f"Example overlay: {args.overlay}")


def make_video(args):
    """Cards slide in one at a time, then sit still; a played event should fire for each."""
    rng = random.Random(args.seed)
    cards = load_cards(args.data, 5, rng)
    layout = random_layout(len(cards), rng)
    fps, slide_s, rest_s = 15, 1.0, 2.0
    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    on_mat = []
    for (card_id, img), (centre, size, angle) in zip(cards, layout):
        start = np.array([centre[0], FRAME_H + size])  # slides in from the bottom edge
        corners_final = card_corners(np.zeros(2), size, angle, rng)
        for f in range(int((slide_s + rest_s) * fps)):
            t = min(1.0, f / (slide_s * fps))
            pos = start + (centre - start) * t
            frame = np.full((FRAME_H, FRAME_W, 3), MATS["green"], np.uint8)
            for other_img, other_corners in on_mat:
                paste(frame, other_img, other_corners)
            paste(frame, img, corners_final + pos.astype(np.float32))
            writer.write(camera_noise(frame, rng))
        on_mat.append((img, corners_final + centre.astype(np.float32)))
        print(f"{len(on_mat)}: {card_id}")
    writer.release()
    print(f"Saved {args.video}: {len(cards)} cards played, {fps} fps")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/tcgdex")
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay", default="data/synthetic_overlay.png")
    parser.add_argument("--video", help="write a fake game clip here instead of evaluating")
    args = parser.parse_args()
    make_video(args) if args.video else evaluate(args)
