"""Metric and input validation checks: python -m unittest fuse.test_evaluate"""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from fuse.evaluate import load_manifest, make_query, summarize
from ocr.reader import read_name


class EvaluationTests(unittest.TestCase):
    def test_covered_name_triggers_real_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "card.png"
            Image.new("RGB", (245, 337), "green").save(path)
            make_query({"image_path": str(path), "condition": "name_covered"}, True).save(path)
            reading = read_name(path)
            self.assertFalse(reading["ocr_available"])
            self.assertEqual(reading["reason"], "Name region has no contrast")

    def test_metrics_distinguish_fixes_regressions_and_shortlist_misses(self):
        results = [
            dict(embedding_correct=False, fused_correct=True, in_top_k=True,
                 ocr_available=True, fallback_preserved=False),
            dict(embedding_correct=True, fused_correct=False, in_top_k=True,
                 ocr_available=True, fallback_preserved=False),
            dict(embedding_correct=False, fused_correct=False, in_top_k=False,
                 ocr_available=False, fallback_preserved=True),
            dict(embedding_correct=True, fused_correct=True, in_top_k=True,
                 ocr_available=False, fallback_preserved=True),
        ]
        summary = summarize(results)
        self.assertEqual(summary["embedding_top1"], .5)
        self.assertEqual(summary["fused_top1"], .5)
        self.assertEqual(summary["embedding_top_k"], .75)
        self.assertEqual(summary["ambiguous_queries"], 1)
        self.assertEqual(summary["ocr_fixes"], 1)
        self.assertEqual(summary["ocr_regressions"], 1)
        self.assertEqual(summary["top_k_misses"], 1)
        self.assertEqual(summary["fallback_queries"], 2)
        self.assertEqual(summary["fallback_ranking_violations"], 0)

    def test_manifest_paths_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "crop.png").touch()
            manifest = root / "photos.csv"
            manifest.write_text("image_path,card_id\ncrop.png,sv01-202\n")
            cases = load_manifest(manifest, {"sv01-202"})
            self.assertEqual(cases[0]["image_path"], str((root / "crop.png").resolve()))
            with self.assertRaisesRegex(ValueError, "not in the demo gallery"):
                load_manifest(manifest, {"other"})
            manifest.write_text("image_path,card_id\nmissing.png,sv01-202\n")
            with self.assertRaisesRegex(ValueError, "image not found"):
                load_manifest(manifest, {"sv01-202"})


if __name__ == "__main__":
    unittest.main()
