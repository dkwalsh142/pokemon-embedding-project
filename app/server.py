"""Live demo server: the browser sends webcam frames, we send back boxes and predictions.

The webcam is on the laptop, but the model and GPU are on this machine, so the page
(app/static/index.html) reads the camera in the browser and POSTs one JPEG at a time
to /frame. Each connection gets its own thread (so one stalled upload can't freeze
the server), but a lock lets only one frame into the model and PlayedTracker at a time.

Usage:
    python -m app.server                 # then open http://localhost:8501
    python -m app.server --port 8600

Over SSH, VS Code forwards the port automatically (or: ssh -L 8501:localhost:8501 host).
Open it as http://localhost:..., because browsers only allow camera access on
localhost or https.
"""

import argparse
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from app.recognizer import Recognizer
from detect.lighting import lighting_gains

STATIC = Path(__file__).parent / "static"
EMBEDDING_MAP = Path("checkpoints/embedding_map.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # full-res phone photos are a few MB
recognizer = None  # created in main(), after the arguments are parsed
data_dir = None
model_lock = threading.Lock()  # one request at a time in the model and tracker


@app.get("/")
def index():
    return send_file(STATIC / "index.html")


@app.post("/frame")
def frame():
    image = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        abort(400, "body must be a JPEG or PNG image")
    image, gains = maybe_normalize(image)
    with model_lock:
        result = recognizer.process(image)
    result["width"], result["height"] = image.shape[1], image.shape[0]
    result["gains"] = gains
    return jsonify(result)


@app.post("/identify")
def identify():
    """A dropped-in still image: identify every card in it. Does not touch the played log.

    With ?x=&y=&w=&h= (image pixels) only that user-drawn box is searched. Inside it we
    still look for the card's outline first; if none is found the box itself is used.
    With ?corners=x1,y1,...,x4,y4 the user clicked the card's corners: warp exactly that.
    """
    image = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        abort(400, "could not read the image (JPEG, PNG or WebP; not HEIC)")
    image, gains = maybe_normalize(image)  # before cropping a drawn box, so the whole scene sets the white balance

    if "corners" in request.args:
        try:
            values = [float(v) for v in request.args["corners"].split(",")]
        except ValueError:
            abort(400, "corners must be 8 numbers")
        if len(values) != 8:
            abort(400, "corners must be 8 numbers")
        with model_lock:
            cards = recognizer.identify_corners(image, np.reshape(values, (4, 2)))
        for card in cards:
            card["manual"] = True
            card["crop"] = crop_data_url(card["crop"])
        return jsonify(cards=cards, width=image.shape[1], height=image.shape[0], gains=gains)

    x0 = y0 = 0
    region = image
    if "x" in request.args:
        try:
            x, y, w, h = (int(float(request.args[k])) for k in ("x", "y", "w", "h"))
        except (KeyError, ValueError):
            abort(400, "region needs numeric x, y, w, h")
        x0, y0 = max(0, x), max(0, y)
        region = image[y0:min(y + h, image.shape[0]), x0:min(x + w, image.shape[1])]
        if region.shape[0] < 10 or region.shape[1] < 10:
            abort(400, "region is too small")

    with model_lock:
        cards = recognizer.identify(region, drawn_box=region is not image)
    for card in cards:
        card["manual"] = region is not image
        card["corners"] = [[px + x0, py + y0] for px, py in card["corners"]]  # back to full-image pixels
        card["crop"] = crop_data_url(card["crop"])
    return jsonify(cards=cards, width=image.shape[1], height=image.shape[0], gains=gains)


def maybe_normalize(image):
    """?normalize=1: white-balance and expose the whole image before detection and matching.

    Returns (image, gains). gains is [R, G, B] (or None when off), so the page can apply
    the same correction to what it displays.
    """
    if request.args.get("normalize") != "1":
        return image, None
    gains = lighting_gains(image)
    corrected = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return corrected, [round(float(g), 3) for g in gains[::-1]]  # BGR -> RGB


def crop_data_url(crop_rgb):
    """Warped crop -> small inline JPEG, so the sidebar can show what was compared."""
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()


@app.get("/card/<card_id>")
def card_image(card_id):
    """Clean reference scan for a card. The path comes from the catalog, never from the URL.

    ?size=low returns the small scan (~20 KB), for pages that show hundreds of thumbnails.
    """
    row = recognizer.rows_by_id.get(card_id)
    if row is None:
        abort(404)
    if request.args.get("size") == "low":
        return send_file(data_dir / row["low_path"], max_age=86400)
    high = row["high_path"] and data_dir / row["high_path"]
    return send_file(high if high and high.exists() else data_dir / row["low_path"])


@app.get("/embeddings")
def embeddings_page():
    return send_file(STATIC / "embeddings.html")


@app.get("/embeddings.json")
def embeddings_map():
    """2-D maps of the index, precomputed by embed/visualize.py (UMAP is too slow per request)."""
    if not EMBEDDING_MAP.exists():
        abort(404, "no embedding map yet; run: python -m embed.visualize")
    return send_file(EMBEDDING_MAP.resolve(), mimetype="application/json")


@app.post("/reset")
def reset():
    with model_lock:
        recognizer.reset()
    return jsonify(ok=True)


def main():
    global recognizer, data_dir
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--ckpt", default="checkpoints/stage1/embedder.pt")
    parser.add_argument("--data-dir", default="data/tcgdex")
    parser.add_argument("--stable", type=int, default=5, help="frames a card must stay still to count as played")
    parser.add_argument("--max-move", type=float, default=40.0,
                        help="pixels a 'still' card may drift between frames (hand-held cards jitter)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    recognizer = Recognizer(args.ckpt, args.data_dir, stable_frames=args.stable, max_move_px=args.max_move)
    print(f"index ready: {len(recognizer.rows)} cards. Open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
