"""Read a card's name band using pytesseract (requires the Tesseract executable)."""

import pytesseract
from PIL import Image, ImageOps


def read_name(image_path, save_crop=None):
    """Return text, availability, and failure reason; optionally save the OCR input.

    The fixed name band is a starting point for standard Pokemon card layouts.
    Other layouts may need different bounds. No catalog text is used here.
    """
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        # Exclude the stage marker on the left and HP/type symbol on the right.
        crop = image.crop((round(w * .15), round(h * .035),
                           round(w * .72), round(h * .085)))
    crop = ImageOps.autocontrast(ImageOps.grayscale(crop))
    crop = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.LANCZOS)
    if save_crop:
        crop.save(save_crop)
    low, high = crop.getextrema()
    if low == high:
        return {"text": "", "ocr_available": False, "reason": "Name region has no contrast"}
    try:
        text = pytesseract.image_to_string(crop, lang="eng", config="--psm 7", timeout=30)
    except pytesseract.TesseractNotFoundError:
        return {"text": "", "ocr_available": False,
                "reason": "Tesseract is missing. On macOS, install with: brew install tesseract"}
    except (pytesseract.TesseractError, RuntimeError, OSError) as exc:
        return {"text": "", "ocr_available": False, "reason": f"Tesseract failed: {exc}"}
    text = " ".join(text.split())
    available = any(char.isalpha() for char in text)
    return {"text": text, "ocr_available": available,
            "reason": "" if available else "No readable name text detected"}
