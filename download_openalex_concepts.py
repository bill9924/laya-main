#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_openalex_concepts.py

One-time, resumable bulk download of ALL OpenAlex Concepts into a local
cache file, via cursor pagination over the plain /concepts LIST endpoint
(NOT /concepts?search=...).

Why this exists: /concepts?search=... only does short-phrase/substring
matching against concept names (confirmed empirically - see project notes),
so it cannot rank a free-text patent abstract. 01_retrieve_concepts.py
instead ranks the *actual* OpenAlex Concepts (downloaded here, verbatim,
nothing invented) against each patent locally. This script only builds
that local cache; it does no patent-specific work.

Output:
    openalex_concepts_data/openalex_concepts.jsonl            one JSON concept object per line
    openalex_concepts_data/openalex_concepts.jsonl.progress.json   resume state (cursor, count)

Usage:
    python download_openalex_concepts.py
    python download_openalex_concepts.py --mailto you@example.com   # faster "polite pool"
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

OPENALEX_CONCEPTS_URL = "https://api.openalex.org/concepts"
PER_PAGE = 200
REQUEST_TIMEOUT = 30
DEFAULT_SLEEP = 0.15
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fetch_concepts")


def fetch_page(cursor: str, mailto: Optional[str]) -> dict:
    params = {"per_page": PER_PAGE, "cursor": cursor}
    if mailto:
        params["mailto"] = mailto

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(OPENALEX_CONCEPTS_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = 2 ** attempt
            logger.warning("Request failed (attempt %d/%d): %s -- retrying in %ds",
                            attempt, MAX_RETRIES, e, wait)
            time.sleep(wait)
    raise last_exc  # exhausted retries


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not parse %s; starting fresh.", progress_path)
    return {"cursor": "*", "fetched_count": 0, "done": False}


def save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Bulk-download all OpenAlex Concepts into a local cache.")
    parser.add_argument("--output", type=Path,
                         default=Path("openalex_concepts_data/openalex_concepts.jsonl"))
    parser.add_argument("--progress-file", type=Path, default=None,
                         help="Defaults to <output>.progress.json")
    parser.add_argument("--mailto", type=str, default=None,
                         help="Email for OpenAlex's polite pool (faster/more reliable).")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    args = parser.parse_args()

    output_path: Path = args.output
    progress_path: Path = args.progress_file or output_path.with_suffix(output_path.suffix + ".progress.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress = load_progress(progress_path)

    if progress.get("done"):
        logger.info("Cache already complete (%d concepts) at %s. Delete %s to re-fetch.",
                     progress["fetched_count"], output_path, progress_path)
        return

    resuming = progress.get("fetched_count", 0) > 0 and output_path.exists()
    if not resuming:
        progress = {"cursor": "*", "fetched_count": 0, "done": False}

    mode = "a" if resuming else "w"
    logger.info("%s download (already have %d concepts).",
                "Resuming" if resuming else "Starting fresh", progress["fetched_count"])

    cursor = progress["cursor"]
    fetched_count = progress["fetched_count"]

    with open(output_path, mode, encoding="utf-8") as f:
        while True:
            data = fetch_page(cursor, args.mailto)
            results = data.get("results", [])
            total = data.get("meta", {}).get("count")

            for concept in results:
                f.write(json.dumps(concept, ensure_ascii=False) + "\n")
            fetched_count += len(results)
            f.flush()

            next_cursor = data.get("meta", {}).get("next_cursor")
            logger.info("Fetched %d / %s concepts", fetched_count, total)

            progress = {"cursor": next_cursor, "fetched_count": fetched_count, "done": False}
            save_progress(progress_path, progress)

            if not next_cursor or not results:
                break
            cursor = next_cursor
            time.sleep(args.sleep)

    progress["done"] = True
    save_progress(progress_path, progress)
    logger.info("Done. %d concepts saved to %s", fetched_count, output_path)


if __name__ == "__main__":
    main()