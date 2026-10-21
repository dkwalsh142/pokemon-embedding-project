"""Camera-like augmentation for clean card scans.

Training images are flat, perfect scans; the demo sees overhead-camera crops after
a perspective warp. Each function below imitates one thing the camera does to a
card. All of them are label-preserving: no flips (text would be mirrored) and the
card is never cropped away.

Every random choice is drawn from a `random.Random` object, so passing a seeded
one reproduces the exact same augmented view (used for the fixed eval queries).

Preview:
    python -m embed.augment --n 15                 # random full-pipeline views
    python -m embed.augment --per-effect --cards 3   # one labelled column per effect
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

WIDTH, HEIGHT = 224, 320  # keeps the card's 63:88 shape
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(img):
    """uint8 HxWx3 numpy array -> float CHW tensor in [0, 1], resized to the model input."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
    if x.shape[1:] != (HEIGHT, WIDTH):
        x = TF.resize(x, [HEIGHT, WIDTH], antialias=True)
    return x


def normalize(x_uint8):
    """uint8 CHW (or batched BCHW) tensor -> ImageNet-normalized float tensor (what the model eats).

    Works on whatever device the input is on, so batches can be normalized on the GPU.
    """
    mean, std = IMAGENET_MEAN.to(x_uint8.device), IMAGENET_STD.to(x_uint8.device)
    return (x_uint8.float() / 255 - mean) / std


# ---------- individual camera effects (float CHW in [0, 1] -> same) ----------

def mat_colour(rng):
    """Background visible around a misaligned crop: near-black or dark green mat."""
    if rng.random() < 0.5:
        return [rng.uniform(0.0, 0.08)] * 3
    return [rng.uniform(0.0, 0.08), rng.uniform(0.15, 0.35), rng.uniform(0.05, 0.15)]


def misalign(x, rng, fill):
    """Small rotate/shift/scale: the detector's crop is never perfectly aligned."""
    return TF.affine(
        x,
        angle=rng.uniform(-3, 3),
        translate=[rng.uniform(-0.03, 0.03) * WIDTH, rng.uniform(-0.03, 0.03) * HEIGHT],
        scale=rng.uniform(0.94, 1.03),
        shear=[0.0, 0.0],
        fill=fill,
    )


def perspective(x, rng, fill, strength=0.04):
    """Leftover keystone distortion when the detected corners are slightly off."""
    w, h = WIDTH - 1, HEIGHT - 1
    corners = [[0, 0], [w, 0], [w, h], [0, h]]
    moved = [[cx + rng.uniform(-strength, strength) * WIDTH,
              cy + rng.uniform(-strength, strength) * HEIGHT] for cx, cy in corners]
    return TF.perspective(x, corners, moved, fill=fill)


def colour_shift(x, rng):
    """White balance (per-channel gain), exposure and contrast of the camera."""
    gain = torch.tensor([rng.uniform(0.85, 1.15) for _ in range(3)]).view(3, 1, 1)
    x = x * gain * rng.uniform(0.7, 1.2)
    mean = x.mean()
    x = (x - mean) * rng.uniform(0.75, 1.15) + mean
    return x.clamp(0, 1)


def glare(x, rng):
    """Soft bright hotspot from the overhead light reflecting off the card."""
    cy, cx = rng.uniform(0, HEIGHT), rng.uniform(0, WIDTH)
    sigma = rng.uniform(0.08, 0.3) * WIDTH
    ys = torch.arange(HEIGHT).view(-1, 1).float()
    xs = torch.arange(WIDTH).view(1, -1).float()
    blob = torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
    return x + rng.uniform(0.25, 0.7) * blob * (1 - x)  # push towards white


def gaussian_blur(x, rng):
    """Slight defocus."""
    sigma = rng.uniform(0.3, 1.6)
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(x, [k, k], [sigma, sigma])


def motion_blur(x, rng):
    """Smear along a random direction (card still moving when the frame is taken)."""
    k = rng.choice([5, 7, 9])
    angle = rng.uniform(0, math.pi)
    kernel = torch.zeros(k, k)
    for t in np.linspace(-1, 1, 2 * k):
        r = round(k // 2 + t * (k // 2) * math.sin(angle))
        c = round(k // 2 + t * (k // 2) * math.cos(angle))
        kernel[r, c] = 1
    kernel = (kernel / kernel.sum()).expand(3, 1, k, k)
    return F.conv2d(x.unsqueeze(0), kernel, padding=k // 2, groups=3).squeeze(0)


def low_resolution(x, rng):
    """The card covers only part of the camera frame, so detail is lost."""
    factor = rng.uniform(0.35, 0.9)
    small = TF.resize(x, [int(HEIGHT * factor), int(WIDTH * factor)], antialias=True)
    return TF.resize(small, [HEIGHT, WIDTH], antialias=True)


# Fraction of the card area covered at each occlusion level, and how often each is picked.
OCCLUSION_LEVELS = {"light": (0.02, 0.05), "medium": (0.05, 0.12), "heavy": (0.12, 0.25)}
OCCLUSION_WEIGHTS = {"light": 0.5, "medium": 0.35, "heavy": 0.15}


def occlusion(x, rng, level=None):
    """A finger, hand or another card covering part of the card from one edge or corner.

    `level` is "light", "medium" or "heavy" (see OCCLUSION_LEVELS); None picks one at random.
    """
    if level is None:
        level = rng.choices(list(OCCLUSION_WEIGHTS), weights=list(OCCLUSION_WEIGHTS.values()))[0]
    area = rng.uniform(*OCCLUSION_LEVELS[level]) * WIDTH * HEIGHT

    x = x.clone()
    colour = torch.tensor([rng.uniform(0.5, 0.9), rng.uniform(0.35, 0.7), rng.uniform(0.3, 0.6)])  # skin
    if rng.random() < 0.4:
        colour = torch.full((3,), rng.uniform(0.0, 0.3))  # dark sleeve or card back
    side = rng.choice(["top", "bottom", "left", "right", "corner"])
    if side in ("top", "bottom"):  # strip along a short edge
        w = int(WIDTH * rng.uniform(0.3, 0.9))
        h = min(int(area / w), HEIGHT // 2)
        x0 = rng.randint(0, WIDTH - w)
        y0 = 0 if side == "top" else HEIGHT - h
    elif side in ("left", "right"):  # strip along a long edge
        h = int(HEIGHT * rng.uniform(0.2, 0.6))
        w = min(int(area / h), WIDTH // 2)
        y0 = rng.randint(0, HEIGHT - h)
        x0 = 0 if side == "left" else WIDTH - w
    else:  # roughly square block over one corner, like an overlapping card
        w = min(int(math.sqrt(area * rng.uniform(0.6, 1.6))), WIDTH // 2)
        h = min(int(area / w), HEIGHT // 2)
        x0 = rng.choice([0, WIDTH - w])
        y0 = rng.choice([0, HEIGHT - h])
    x[:, y0:y0 + h, x0:x0 + w] = colour.view(3, 1, 1)
    return x


def sensor_noise(x, rng):
    gen = torch.Generator().manual_seed(rng.getrandbits(32))
    return (x + torch.randn(x.shape, generator=gen) * rng.uniform(0.005, 0.03)).clamp(0, 1)


def jpeg(x_uint8, rng):
    """Video/JPEG compression artefacts."""
    return TF.jpeg(x_uint8, quality=rng.randint(30, 90))


# ---------- full pipeline ----------

def camera_augment(img, rng=None):
    """Clean scan (uint8 HxWx3) -> camera-like view (uint8 CHW tensor).

    rng=None draws fresh randomness (training); a seeded random.Random gives a
    reproducible view (eval queries).
    """
    rng = rng or random.Random()
    x = to_tensor(img)
    fill = mat_colour(rng)

    x = misalign(x, rng, fill)
    if rng.random() < 0.7:
        x = perspective(x, rng, fill)
    x = colour_shift(x, rng)
    if rng.random() < 0.5:
        x = glare(x, rng)
    blur = rng.random()
    if blur < 0.5:
        x = gaussian_blur(x, rng)
    elif blur < 0.7:
        x = motion_blur(x, rng)
    if rng.random() < 0.6:
        x = low_resolution(x, rng)
    if rng.random() < 0.3:
        x = occlusion(x, rng)
    if rng.random() < 0.6:
        x = sensor_noise(x, rng)

    x_uint8 = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    if rng.random() < 0.7:
        x_uint8 = jpeg(x_uint8, rng)
    return x_uint8


def clean_view(img):
    """Reference image for the index: just resized, no augmentation (uint8 CHW)."""
    return (to_tensor(img) * 255).round().to(torch.uint8)


def effect_gallery(images, rng):
    """One column per camera effect applied on its own, plus the full pipeline.

    Under each card is a 2x zoom of the name bar, where blur, noise and JPEG
    artefacts are easy to see. Returns a PIL image.
    """
    from PIL import Image, ImageDraw, ImageFont

    def uint8_effect(fn):
        return lambda x, r: fn((x * 255).round().to(torch.uint8), r).float() / 255

    effects = [
        ("clean scan", lambda x, r: x),
        ("misalign", lambda x, r: misalign(x, r, mat_colour(r))),
        ("perspective", lambda x, r: perspective(x, r, mat_colour(r))),
        ("colour shift", colour_shift),
        ("glare", glare),
        ("gaussian blur", gaussian_blur),
        ("motion blur", motion_blur),
        ("low resolution", low_resolution),
        *[(f"occlusion ({lvl})", lambda x, r, lvl=lvl: occlusion(x, r, lvl)) for lvl in OCCLUSION_LEVELS],
        ("sensor noise", sensor_noise),
        ("JPEG", uint8_effect(jpeg)),
        ("all combined", None),
    ]
    header, zoom_h = 36, HEIGHT // 3  # zoom shows the top-left 112x53 px at 2x
    row_h = HEIGHT + zoom_h
    grid = Image.new("RGB", (len(effects) * WIDTH, header + len(images) * row_h), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=20)

    for c, (label, fn) in enumerate(effects):
        draw.text((c * WIDTH + WIDTH // 2, header // 2), label, fill="black", font=font, anchor="mm")
        for r, img in enumerate(images):
            if fn is None:
                view = camera_augment(img, rng)
            else:
                view = (fn(to_tensor(img), rng).clamp(0, 1) * 255).round().to(torch.uint8)
            card = Image.fromarray(view.permute(1, 2, 0).numpy())
            zoom = card.crop((0, 0, WIDTH // 2, zoom_h // 2)).resize((WIDTH, zoom_h), Image.NEAREST)
            y = header + r * row_h
            grid.paste(card, (c * WIDTH, y))
            grid.paste(zoom, (c * WIDTH, y + HEIGHT))
    for c in range(1, len(effects)):  # thin separators between columns
        draw.line([(c * WIDTH, 0), (c * WIDTH, grid.height)], fill="white", width=3)
    return grid


if __name__ == "__main__":
    import argparse
    import csv
    from pathlib import Path

    from PIL import Image

    parser = argparse.ArgumentParser(description="Save a grid of augmented views of a few random cards.")
    parser.add_argument("--data", default="data/tcgdex")
    parser.add_argument("--n", type=int, default=15, help="augmented views per card")
    parser.add_argument("--cards", type=int, default=4)
    parser.add_argument("--per-effect", action="store_true",
                        help="one labelled column per effect instead of random full-pipeline views")
    parser.add_argument("--seed", type=int, help="fix the random cards and effects")
    parser.add_argument("--out", help="default: data/augment_preview.png (data/augment_effects.png with --per-effect)")
    args = parser.parse_args()
    random.seed(args.seed)

    with open(Path(args.data) / "catalog.csv", newline="") as f:
        rows = random.sample(list(csv.DictReader(f)), args.cards)
    images = [np.array(Image.open(Path(args.data) / row["low_path"]).convert("RGB")) for row in rows]

    if args.per_effect:
        out = args.out or "data/augment_effects.png"
        effect_gallery(images, random.Random(args.seed)).save(out)
        print(f"Saved {out}: each column is one effect on its own; under each card is a 2x zoom of the name bar")
        raise SystemExit

    args.out = args.out or "data/augment_preview.png"
    cols = args.n + 1
    grid = Image.new("RGB", (cols * WIDTH, args.cards * HEIGHT), "white")
    for r, img in enumerate(images):
        views = [clean_view(img)] + [camera_augment(img) for _ in range(args.n)]
        for c, v in enumerate(views):
            grid.paste(Image.fromarray(v.permute(1, 2, 0).numpy()), (c * WIDTH, r * HEIGHT))
    grid.save(args.out)
    print(f"Saved {args.out}: each row is one card, clean scan first, then {args.n} augmented views")
