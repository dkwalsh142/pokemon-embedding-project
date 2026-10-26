"""Normalize the lighting of a camera frame before detection and embedding.

The reference index is built from clean, evenly lit scans, but the camera sees a
warm or cool lamp, a dim room, or an overexposed table. Two simple steps:

  1. White balance on near-gray pixels: the white text boxes and silver frames of
     the cards (and white paper) should be neutral, so scale each channel until
     their average is gray. Colourful pixels are left out: plain gray-world
     (averaging everything) turns the image blue, because the yellow card borders
     pull the average towards yellow. Very dark pixels (the black mat) and
     blown-out ones (glare) are left out too. Under a strong tint nothing looks gray,
     so we start from the brightest pixels (assumed white) and refine a few times.
  2. Exposure: scale brightness so the bright end of the image (99th percentile)
     sits near white, like a scan.

Both are a single multiply per channel, so the card's own colours are not distorted,
and both gains are clamped so one very colourful card can't be "corrected" to gray.

Usage:
    python -m detect.lighting photo.jpg        # saves photo_lighting.png (before | after)
"""

import cv2
import numpy as np

DARK = 30         # pixels darker than this (0-255) are mat/shadow: ignored for white balance
BRIGHT = 250      # pixels with a channel at or above this are glare: ignored
GRAY_SATURATION = 0.15  # (max - min) / max below this counts as a near-gray pixel
MAX_WB_GAIN = 1.4  # white balance never changes a channel by more than this factor
MAX_EXPOSURE_GAIN = 2.0
WHITE_POINT = 245  # target for the 99th-percentile brightness


def lighting_gains(image_bgr):
    """Per-channel (B, G, R) multipliers that white-balance and expose the image."""
    img = image_bgr.astype(np.float32)
    usable = (img.mean(axis=2) > DARK) & (img.max(axis=2) < BRIGHT)
    if usable.sum() < 500:  # almost nothing but mat or glare: leave the image alone
        return np.ones(3, np.float32)

    # First guess: the brightest usable pixels (white paper, text boxes) should be white.
    # Under a strong tint even those look coloured, so they wouldn't pass the gray test yet.
    pixels = img[usable]
    brightest = pixels[pixels.mean(axis=1) >= np.percentile(pixels.mean(axis=1), 95)]
    wb = np.clip(brightest.mean() / brightest.mean(axis=0), 1 / MAX_WB_GAIN, MAX_WB_GAIN)

    # Refine: find the near-gray pixels of the corrected image and balance on those.
    for _ in range(3):
        corrected = pixels * wb
        saturation = (corrected.max(axis=1) - corrected.min(axis=1)) / np.maximum(corrected.max(axis=1), 1)
        gray = pixels[saturation < GRAY_SATURATION]
        if len(gray) < 500:
            break
        wb = np.clip(gray.mean() / gray.mean(axis=0), 1 / MAX_WB_GAIN, MAX_WB_GAIN)

    brightness = (img * wb).mean(axis=2)
    exposure = np.clip(WHITE_POINT / max(np.percentile(brightness, 99), 1), 1 / MAX_EXPOSURE_GAIN, MAX_EXPOSURE_GAIN)
    return (wb * exposure).astype(np.float32)


def normalize_lighting(image_bgr):
    """Return a white-balanced, exposure-corrected copy of a BGR uint8 image."""
    gains = lighting_gains(image_bgr)
    return np.clip(image_bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read {args.image}")
    b, g, r = lighting_gains(image)
    print(f"gains: R x{r:.2f}  G x{g:.2f}  B x{b:.2f}")
    out = Path(args.image).with_name(Path(args.image).stem + "_lighting.png")
    cv2.imwrite(str(out), np.hstack([image, normalize_lighting(image)]))
    print(f"saved {out} (left: original, right: normalized)")


if __name__ == "__main__":
    main()
