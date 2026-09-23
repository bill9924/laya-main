#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patent2candidate.py  (local-matching version)

Stage 1 of the Patent -> OpenAlex Concepts pipeline.

Why this version does NOT call /concepts?search=... per patent:
  Empirical testing showed that endpoint does short-phrase/substring
  matching against a small (~65k) vocabulary of concept names, not
  full-text relevance search like /works. A free-text patent abstract
  essentially never appears verbatim inside a concept's short
  display_name/description, so per-patent remote search returns empty
  results for the large majority of patents, however the query is built
  or trimmed. That's a hard limitation of this deprecated endpoint.

  To still satisfy "use real OpenAlex Concepts (not Topics, not a
  self-invented taxonomy), title+description as input, top_k candidates
  for every patent", this script instead:
    1. Loads the ACTUAL OpenAlex Concepts from a local cache built by
       download_openalex_concepts.py (openalex_concepts_data/openalex_concepts.jsonl)
       - a verbatim copy of real OpenAlex data, not an invented category system.
    2. Ranks all cached concepts against each patent's title+description
       using TF-IDF + cosine similarity, computed locally.
    3. Takes the top_k highest-similarity concepts as candidates.

  This guarantees every patent gets top_k candidates (the only exception
  is a patent with literally empty title AND description - nothing to
  build a query from). The "score" field on each candidate is OUR
  locally-computed cosine similarity in [0, 1] - it is NOT something
  OpenAlex returned, and is unrelated to OpenAlex's own (API-only,
  unreliable on this endpoint) relevance_score. Every other concept
  field (id, display_name, description, level, works_count,
  cited_by_count, wikidata) is copied verbatim from the cached data -
  nothing invented there either.

Prerequisite (run once):
    python download_openalex_concepts.py

Usage:
    python patent2candidate.py
    python patent2candidate.py --top-k 20
    python patent2candidate.py --limit 20      # quick smoke test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_TOP_K = 10
SAVE_EVERY = 200  # pure local computation now (no network calls per patent),
                   # so periodic checkpoints just guard against a crash mid-run

CONCEPT_FIELDS_TO_KEEP = [
    "id", "display_name", "description", "level",
    "works_count", "cited_by_count", "wikidata",
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("retrieve_concepts")


# ---------------------------------------------------------------------------
# 1. Load patents
# ---------------------------------------------------------------------------

def load_patents(input_path: Path) -> List[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        patents = json.load(f)

    if not isinstance(patents, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(patents).__name__}")
    if not patents:
        logger.warning("Input file %s is empty.", input_path)
        return patents

    sample_keys = sorted(patents[0].keys())
    logger.info("Loaded %d patents from %s", len(patents), input_path)
    logger.info("Sample record keys: %s", sample_keys)

    required = {"title", "description"}
    missing = required - set(sample_keys)
    if missing:
        raise KeyError(
            f"Patent records are missing required field(s) {missing}. "
            f"Actual keys found on the first record: {sample_keys}."
        )
    return patents


def get_patent_id(patent: dict, index: int) -> str:
    for key in ("patent_id", "id", "publication_number"):
        value = patent.get(key)
        if value:
            return str(value)
    logger.warning(
        "Patent at index %d has no patent_id/id/publication_number field; "
        "falling back to a positional placeholder id.", index,
    )
    return f"__no_id__index_{index}"


def build_query(patent: dict) -> str:
    title = (patent.get("title") or "").strip()
    description = (patent.get("description") or "").strip()
    if title and description:
        return f"{title}. {description}"
    return title or description


# ---------------------------------------------------------------------------
# 2. Load local OpenAlex Concepts cache + build TF-IDF index
# ---------------------------------------------------------------------------

def load_concepts_cache(cache_path: Path) -> List[dict]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Run `python download_openalex_concepts.py` first "
            f"to build the local OpenAlex Concepts cache."
        )
    concepts = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                concepts.append(json.loads(line))
    if not concepts:
        raise ValueError(f"{cache_path} is empty. Re-run download_openalex_concepts.py.")
    logger.info("Loaded %d cached OpenAlex concepts from %s", len(concepts), cache_path)
    return concepts


def concept_text(concept: dict) -> str:
    name = concept.get("display_name") or ""
    desc = concept.get("description") or ""
    return f"{name}. {desc}".strip()


def build_index(concepts: List[dict]):
    texts = [concept_text(c) for c in concepts]
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=1)
    matrix = vectorizer.fit_transform(texts)  # shape: (n_concepts, vocab_size)
    return vectorizer, matrix


# ---------------------------------------------------------------------------
# 3. Per-patent local ranking (replaces the old search_concepts())
# ---------------------------------------------------------------------------

def rank_concepts(query: str, vectorizer, concept_matrix, concepts: List[dict], top_k: int) -> List[dict]:
    if not query:
        return []
    query_vec = vectorizer.transform([query])                     # (1, vocab_size)
    sims = cosine_similarity(query_vec, concept_matrix)[0]        # (n_concepts,)
    k = min(top_k, len(concepts))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]                 # sort just the top-k

    candidates = []
    for idx in top_idx:
        c = concepts[int(idx)]
        entry = {field: c[field] for field in CONCEPT_FIELDS_TO_KEEP if field in c}
        entry["score"] = round(float(sims[idx]), 6)  # locally-computed cosine similarity
        candidates.append(entry)
    return candidates


def process_patent(patent: dict, patent_id: str, vectorizer, concept_matrix, concepts, top_k: int) -> dict:
    query = build_query(patent)
    record = {
        "patent_id": patent_id,
        "title": patent.get("title", ""),
        "description": patent.get("description", ""),
        "query": query,
        "candidate_concepts": [],
    }
    if not query:
        record["error"] = "empty title and description; no query could be built"
        return record

    record["candidate_concepts"] = rank_concepts(query, vectorizer, concept_matrix, concepts, top_k)
    return record


# ---------------------------------------------------------------------------
# 4. Checkpointing / resume + save
# ---------------------------------------------------------------------------

def load_existing_results(output_path: Path) -> List[dict]:
    if not output_path.exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read existing %s (%s); starting fresh.", output_path, e)
        return []
    if not isinstance(existing, list):
        logger.warning("%s did not contain a JSON list; starting fresh.", output_path)
        return []
    return existing


def save_results(results: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (local matching): rank real OpenAlex Concepts per patent via TF-IDF."
    )
    parser.add_argument("--input", type=Path, default=Path("dataset/patent.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/candidates.json"))
    parser.add_argument("--concepts-cache", type=Path,
                         default=Path("openalex_concepts_data/openalex_concepts.jsonl"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap on number of NEW patents to process this run (for a quick test).")
    args = parser.parse_args()

    patents = load_patents(args.input)
    total = len(patents)

    concepts = load_concepts_cache(args.concepts_cache)
    logger.info("Building TF-IDF index over %d concepts...", len(concepts))
    vectorizer, concept_matrix = build_index(concepts)
    logger.info("Index ready.")

    results = load_existing_results(args.output)
    done_ids = {r["patent_id"] for r in results if "patent_id" in r}
    if done_ids:
        logger.info("Found %d already-processed patents in %s; skipping those.",
                     len(done_ids), args.output)

    processed_this_run = 0

    for idx, patent in enumerate(patents, start=1):
        patent_id = get_patent_id(patent, idx - 1)
        if patent_id in done_ids:
            continue
        if args.limit is not None and processed_this_run >= args.limit:
            logger.info("Reached --limit=%d new patents for this run; stopping.", args.limit)
            break

        query = build_query(patent)
        preview = query[:60].replace("\n", " ")
        logger.info("[%d/%d] Processing patent: %s", idx, total, patent_id)
        logger.info("    Query: %s%s", preview, "..." if len(query) > 60 else "")

        record = process_patent(patent, patent_id, vectorizer, concept_matrix, concepts, args.top_k)

        if record.get("error"):
            logger.info("[ERROR] Patent %s", patent_id)
            logger.info("Reason: %s", record["error"])
        else:
            top_names = [c.get("display_name") for c in record["candidate_concepts"][:3]]
            logger.info("    Retrieved concepts: %d (top: %s)", len(record["candidate_concepts"]), top_names)

        results.append(record)
        done_ids.add(patent_id)
        processed_this_run += 1

        if processed_this_run % SAVE_EVERY == 0:
            save_results(results, args.output)

    save_results(results, args.output)
    logger.info("Done. %d new patents processed this run; %d total records in %s",
                processed_this_run, len(results), args.output)


if __name__ == "__main__":
    main()
