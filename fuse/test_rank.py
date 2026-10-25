"""Run with: python -m unittest fuse.test_rank"""

import unittest

from fuse.rank import rank_candidates


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"card_id": "wrong", "cosine_score": .85, "ocr_score": .1, "ocr_available": True},
            {"card_id": "correct", "cosine_score": .8, "ocr_score": 1., "ocr_available": True},
        ]

    def test_ocr_can_correct_close_visual_match(self):
        ranked = rank_candidates(self.candidates)
        self.assertEqual(ranked[0]["card_id"], "correct")
        self.assertEqual(ranked[0]["cosine_score"], .8)
        self.assertTrue(ranked[0]["ocr_used"])
        self.assertNotIn("final_score", self.candidates[0])

    def test_missing_ocr_preserves_embedding_scores_for_entire_shortlist(self):
        for candidate in self.candidates:
            candidate.update(ocr_available=False, ocr_score=None)
        ranked = rank_candidates(self.candidates)
        self.assertEqual([c["card_id"] for c in ranked], ["wrong", "correct"])
        for candidate in ranked:
            self.assertEqual(candidate["final_score"], candidate["cosine_score"])
            self.assertFalse(candidate["ocr_used"])
        self.candidates[0].update(ocr_available=True, ocr_score=.1)
        self.assertFalse(rank_candidates(self.candidates)[0]["ocr_used"])

    def test_weight_endpoints(self):
        self.assertEqual(rank_candidates(self.candidates, 0)[0]["card_id"], "wrong")
        self.assertEqual(rank_candidates(self.candidates, 1)[0]["card_id"], "correct")

    def test_stable_ties_and_empty_shortlist(self):
        self.candidates[1].update(cosine_score=.85, ocr_score=.1)
        self.assertEqual([c["card_id"] for c in rank_candidates(self.candidates)],
                         ["wrong", "correct"])
        self.assertEqual(rank_candidates([]), [])

    def test_invalid_weight(self):
        for weight in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                rank_candidates(self.candidates, weight)


if __name__ == "__main__":
    unittest.main()
