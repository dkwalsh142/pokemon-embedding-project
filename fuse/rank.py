"""Weighted reranking of an embedding shortlist using OCR name scores."""

import math


def rank_candidates(candidates, ocr_weight=0.2):
    """Return new candidate dicts sorted by final_score, highest first.

    final_score = (1 - ocr_weight) * cosine_score + ocr_weight * ocr_score

    Cosine ranges from -1 to 1; OCR similarity ranges from 0 to 1. This is a
    heuristic score, not a probability; the default weight needs real-photo
    validation. Original component scores and metadata are retained.

    OCR must be available for the entire shortlist to compare fused scores
    fairly. Otherwise use raw cosine for every candidate. Stable sorting keeps
    the original embedding order for ties. Input dictionaries are not modified.
    """
    if not math.isfinite(ocr_weight) or not 0 <= ocr_weight <= 1:
        raise ValueError("ocr_weight must be between 0 and 1")
    candidates = list(candidates)
    use_ocr = ocr_weight > 0 and bool(candidates) and all(
        c.get("ocr_available", False) and c.get("ocr_score") is not None
        for c in candidates
    )
    ranked = []
    for candidate in candidates:
        cosine = candidate["cosine_score"]
        if not math.isfinite(cosine):
            raise ValueError("cosine_score must be finite")
        score = cosine
        if use_ocr:
            ocr = candidate["ocr_score"]
            if not math.isfinite(ocr) or not 0 <= ocr <= 1:
                raise ValueError("ocr_score must be between 0 and 1")
            score = (1 - ocr_weight) * cosine + ocr_weight * ocr
        ranked.append(dict(candidate, final_score=score, ocr_used=use_ocr))
    return sorted(ranked, key=lambda c: c["final_score"], reverse=True)
