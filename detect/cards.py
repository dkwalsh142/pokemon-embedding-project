"""Find cards on the mat, warp each one flat, and check the crop is usable.

Classic OpenCV, no machine learning:
    grayscale -> blur -> Canny edges -> close gaps -> outer contours
    -> keep 4-corner shapes that are rectangular and card-shaped (63:88)
    -> order the corners -> perspective-warp to a 224x320 crop (the model's input size)
    -> crop QC (size, rectangularity, brightness, sharpness)

Every detection is returned, including the ones that fail QC, so the UI can show
why a card was dropped.

Usage:
    python -m detect.cards frame.jpg                     # saves frame_overlay.png
    python -m detect.cards frame.jpg --crops-dir crops/  # also saves each warped card
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from embed.augment import HEIGHT, WIDTH

CARD_ASPECT = 63 / 88       # short side / long side
ASPECT_TOLERANCE = 0.12     # allowed relative error on the aspect ratio
MIN_AREA_FRAC = 0.002       # card must cover at least this fraction of the frame...
MAX_AREA_FRAC = 0.4         # ...and at most this much
MIN_RECTANGULARITY = 0.85   # contour area / area of its 4-corner polygon

# Crop QC thresholds (tune on the real setup)
MIN_SHORT_SIDE_PX = 60      # card narrower than this in the frame has too little detail
BRIGHTNESS_RANGE = (35, 230)
MIN_SHARPNESS = 30.0        # variance of the Laplacian; lower means blurry


@dataclass
class Detection:
    corners: np.ndarray     # 4x2 float32 in frame pixels: top-left, top-right, bottom-right, bottom-left
    crop: np.ndarray        # HEIGHT x WIDTH x 3 RGB uint8, the card warped flat
    ok: bool                # passed crop QC
    reason: str             # why QC failed ("" if ok)
    rectangularity: float

    @property
    def centre(self):
        return self.corners.mean(axis=0)

    @property
    def short_side(self):
        sides = [np.linalg.norm(self.corners[i] - self.corners[(i + 1) % 4]) for i in range(4)]
        return min(sides)


def order_corners(pts):
    """Order 4 points so the card reads upright: TL, TR, BR, BL with TL->TR a short side.

    Sorting by angle around the centre gives clockwise order (image y points down).
    We then rotate the order so the first edge is a short side, and of the two short
    sides pick the higher one in the image as the top. A card the other player put
    down upside-down therefore comes out rotated 180 degrees; retrieval handles that.
    """
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]
    if np.linalg.norm(pts[0] - pts[1]) > np.linalg.norm(pts[1] - pts[2]):
        pts = np.roll(pts, -1, axis=0)  # make edge 0->1 a short side
    top_a = (pts[0][1] + pts[1][1]) / 2
    top_b = (pts[2][1] + pts[3][1]) / 2
    if top_b < top_a:
        pts = np.roll(pts, -2, axis=0)
    return pts.astype(np.float32)


def warp(frame_bgr, corners):
    """Perspective-warp the card to a flat WIDTH x HEIGHT RGB crop."""
    dst = np.float32([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]])
    matrix = cv2.getPerspectiveTransform(corners, dst)
    crop = cv2.warpPerspective(frame_bgr, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def aspect_ratio(corners):
    """Short side / long side, averaging opposite edges."""
    w = (np.linalg.norm(corners[0] - corners[1]) + np.linalg.norm(corners[3] - corners[2])) / 2
    h = (np.linalg.norm(corners[0] - corners[3]) + np.linalg.norm(corners[1] - corners[2])) / 2
    return min(w, h) / max(w, h)


def find_quads(frame_bgr, max_area_frac=MAX_AREA_FRAC):
    """Return (corners, rectangularity) for every card-shaped contour in the frame.

    max_area_frac can be raised when the frame is a user-drawn box around one card,
    where the card fills most of the image.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # join broken edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    quads = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not MIN_AREA_FRAC * frame_area < area < max_area_frac * frame_area:
            continue
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) != 4:
            continue
        corners = order_corners(approx.reshape(4, 2).astype(np.float32))
        if abs(aspect_ratio(corners) - CARD_ASPECT) / CARD_ASPECT > ASPECT_TOLERANCE:
            continue
        rectangularity = area / cv2.contourArea(corners)
        if rectangularity < MIN_RECTANGULARITY:
            continue
        quads.append((corners, min(rectangularity, 1.0)))
    return quads


def crop_qc(crop_rgb, corners, rectangularity):
    """Cheap checks that the warped crop is worth sending to the model. Returns (ok, reason)."""
    short_side = min(np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4))
    if short_side < MIN_SHORT_SIDE_PX:
        return False, f"too small ({short_side:.0f}px wide)"
    if rectangularity < MIN_RECTANGULARITY:
        return False, f"not rectangular ({rectangularity:.2f})"
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    brightness = gray.mean()
    if not BRIGHTNESS_RANGE[0] < brightness < BRIGHTNESS_RANGE[1]:
        return False, f"bad exposure (mean {brightness:.0f})"
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness < MIN_SHARPNESS:
        return False, f"blurry (sharpness {sharpness:.0f})"
    return True, ""


def detect_cards(frame_bgr, max_area_frac=MAX_AREA_FRAC):
    """Frame (BGR, as OpenCV reads it) -> list of Detection, one per card-shaped region."""
    detections = []
    for corners, rectangularity in find_quads(frame_bgr, max_area_frac):
        crop = warp(frame_bgr, corners)
        ok, reason = crop_qc(crop, corners, rectangularity)
        detections.append(Detection(corners, crop, ok, reason, rectangularity))
    return detections


def draw_overlay(frame_bgr, detections, labels=None):
    """Copy of the frame with each card outlined (green = passed QC, red = dropped) and labelled."""
    out = frame_bgr.copy()
    thickness = max(2, round(max(out.shape[:2]) / 600))
    for i, det in enumerate(detections):
        colour = (0, 200, 0) if det.ok else (0, 0, 230)
        pts = det.corners.astype(np.int32)
        cv2.polylines(out, [pts], True, colour, thickness)
        cv2.circle(out, tuple(pts[0]), 3 * thickness, colour, -1)  # marks the card's top-left corner
        text = labels[i] if labels else (f"card {i}" if det.ok else det.reason)
        x, y = pts[:, 0].min(), pts[:, 1].min() - 8
        cv2.putText(out, text, (int(x), int(max(y, 20))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * thickness, colour, thickness)
    return out


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--out", help="overlay path (default: <image>_overlay.png)")
    parser.add_argument("--crops-dir", help="save each warped card here")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"could not read {args.image}")
    detections = detect_cards(frame)

    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_overlay.png"))
    cv2.imwrite(out, draw_overlay(frame, detections))
    for i, det in enumerate(detections):
        print(f"card {i}: centre ({det.centre[0]:.0f}, {det.centre[1]:.0f}), "
              f"{det.short_side:.0f}px wide, " + ("ok" if det.ok else f"dropped: {det.reason}"))
        if args.crops_dir:
            Path(args.crops_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.crops_dir) / f"card_{i}.png"), cv2.cvtColor(det.crop, cv2.COLOR_RGB2BGR))
    print(f"{len(detections)} cards found, {sum(d.ok for d in detections)} passed QC. Overlay: {out}")


if __name__ == "__main__":
    main()
