"""Download card data and images from TCGdex (https://tcgdex.dev).

Everything is cached under the output folder, so re-running only fetches what
is missing. Cards with no image or broken data are skipped and logged, never
crash the run.

Usage:
    python -m data_tools.download_tcgdex --sets sv01 sv02
    python -m data_tools.download_tcgdex --all
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

API = "https://api.tcgdex.net/v2/en"

CATALOG_COLUMNS = [
    "card_id", "set_id", "set_name", "local_id", "set_card_count",
    "name", "category", "rarity", "attack_names", "low_path", "high_path",
]

log = logging.getLogger("download_tcgdex")


class Downloader:
    def __init__(self, out_dir, sleep, get_high):
        self.out = Path(out_dir)
        self.sleep = sleep
        self.get_high = get_high
        self.session = requests.Session()

    # ---------- low-level helpers ----------

    def fetch(self, url):
        """GET a URL with a few retries. Sleeps after each request to be polite."""
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=30)
                time.sleep(self.sleep)
                if resp.status_code == 404:
                    raise FileNotFoundError(f"HTTP 404 for {url}")
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                wait = 2 ** (attempt + 1)  # 2s, then 4s
                log.warning("Request failed (%s), retrying in %ss", e, wait)
                time.sleep(wait)

    def get_json(self, url, cache_path):
        """Return JSON from the local cache, or download and cache it."""
        cache_path = Path(cache_path)
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        data = self.fetch(url).json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        return data

    def download_image(self, url, path):
        """Download an image unless it is already on disk. Checks it opens with Pillow."""
        path = Path(path)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(self.fetch(url).content)
        try:
            with Image.open(tmp) as img:
                img.verify()
        except Exception as e:
            tmp.unlink()
            raise ValueError(f"corrupt image: {e}")
        tmp.rename(path)

    # ---------- per-set / per-card logic ----------

    def image_paths(self, set_id, card_id):
        low = self.out / "images" / "low" / set_id / f"{card_id}.webp"
        high = self.out / "images" / "high" / set_id / f"{card_id}.png"
        return low, high

    def process_card(self, set_id, card_id):
        """Fetch one card's JSON and images. Raises with a readable reason on failure."""
        card = self.get_json(f"{API}/cards/{card_id}", self.out / "cards" / f"{card_id}.json")
        for field in ("name", "localId"):
            if not card.get(field):
                raise ValueError(f"missing field '{field}'")
        image_base = card.get("image")
        if not image_base:
            raise ValueError("no image URL")

        low, high = self.image_paths(set_id, card_id)
        self.download_image(f"{image_base}/low.webp", low)
        if self.get_high:
            self.download_image(f"{image_base}/high.png", high)

    def process_set(self, set_id):
        """Download every card in a set. Returns a list of (card_id, reason) skips."""
        set_data = self.get_json(f"{API}/sets/{set_id}", self.out / "sets" / f"{set_id}.json")
        cards = set_data.get("cards", [])
        skipped = []
        for brief in tqdm(cards, desc=set_id, unit="card"):
            card_id = brief["id"]
            try:
                self.process_card(set_id, card_id)
            except Exception as e:
                skipped.append((card_id, str(e)))
                log.debug("Skipped %s: %s", card_id, e)

        # Save this set's skip list so the combined skipped.csv covers all runs.
        self.write_csv(self.out / "skipped" / f"{set_id}.csv", ["card_id", "reason"], skipped)
        log.info("%s (%s): %d ok, %d skipped",
                 set_id, set_data.get("name"), len(cards) - len(skipped), len(skipped))
        return skipped

    # ---------- catalog building ----------

    def build_catalog(self):
        """Rebuild catalog.csv and skipped.csv from everything cached on disk."""
        rows = []
        for card_file in sorted((self.out / "cards").glob("*.json")):
            card = json.loads(card_file.read_text())
            set_info = card.get("set", {})
            set_id = set_info.get("id", "")
            low, high = self.image_paths(set_id, card["id"])
            if not low.exists():
                continue  # no usable image -> it is listed in skipped.csv instead
            rows.append([
                card["id"],
                set_id,
                set_info.get("name", ""),
                card.get("localId", ""),
                set_info.get("cardCount", {}).get("official", ""),
                card.get("name", ""),
                card.get("category", ""),
                card.get("rarity", ""),
                ";".join(a.get("name", "") for a in card.get("attacks", [])),
                low.relative_to(self.out).as_posix(),
                high.relative_to(self.out).as_posix() if high.exists() else "",
            ])
        self.write_csv(self.out / "catalog.csv", CATALOG_COLUMNS, rows)

        skipped = []
        for skip_file in sorted((self.out / "skipped").glob("*.csv")):
            with open(skip_file, newline="") as f:
                skipped.extend(list(csv.reader(f))[1:])
        self.write_csv(self.out / "skipped.csv", ["card_id", "reason"], skipped)

        log.info("catalog.csv: %d cards, skipped.csv: %d cards", len(rows), len(skipped))

    @staticmethod
    def write_csv(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sets", nargs="+", help="TCGdex set IDs, e.g. sv01 base1")
    group.add_argument("--all", action="store_true", help="download every set in the catalog")
    parser.add_argument("--out", default="data/tcgdex", help="output folder (default: data/tcgdex)")
    parser.add_argument("--sleep", type=float, default=0.2, help="seconds to wait after each request")
    parser.add_argument("--no-high", action="store_true", help="skip the high-res PNG images")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every skipped card")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    dl = Downloader(args.out, args.sleep, get_high=not args.no_high)
    all_sets = dl.get_json(f"{API}/sets", dl.out / "sets.json")
    known_ids = {s["id"] for s in all_sets}

    set_ids = [s["id"] for s in all_sets] if args.all else args.sets
    for set_id in set_ids:
        if set_id not in known_ids:
            log.error("Unknown set id '%s' (see %s for valid ids)", set_id, dl.out / "sets.json")
            continue
        try:
            dl.process_set(set_id)
        except Exception as e:
            log.error("Could not process set %s: %s", set_id, e)

    dl.build_catalog()


if __name__ == "__main__":
    main()
