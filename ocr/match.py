"""Simple name similarity against the embedding shortlist only."""

from difflib import SequenceMatcher
import unicodedata


def normalize_name(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def score_candidates(candidates, reading):
    """Preserve retrieval order and add an optional 0..1 OCR name score.

    These string similarities are not probabilities. Missing OCR produces None,
    so a later fusion step can explicitly fall back to embedding alone.
    """
    query = normalize_name(reading["text"])
    available = reading["ocr_available"] and bool(query)
    return [dict(candidate, ocr_available=available,
                 ocr_score=SequenceMatcher(
                     None, query, normalize_name(candidate["metadata"]["name"]),
                     autojunk=False).ratio() if available else None)
            for candidate in candidates]
