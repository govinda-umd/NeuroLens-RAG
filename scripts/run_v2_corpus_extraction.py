"""Corpus-wide claim extraction over section-based chunks (v2 pipeline,
step 1-3 of docs/v2/rag-cav-verification-loop-design.md).

Long-running (~60-70 min on the local Llama-3.2-3B-Instruct-4bit model) --
run via `nohup` / background, not interactively. Writes incrementally to
`--out` so a partial run is still usable if interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurolens import verification_v2 as v2
from neurolens.concepts import map_phrase_to_known_concept
from neurolens.pipeline import build_concept_extraction_prompt, make_mlx_generate_fn
from neurolens.retrieval import ingest_pdf_directory_by_section

PAPERS_DIR = Path(__file__).resolve().parents[1] / "data" / "papers"


def extract_claim(chunk_text: str, generate_fn) -> str | None:
    response = generate_fn(build_concept_extraction_prompt(chunk_text)).strip()
    if response.upper().startswith("NONE") or len(response) == 0 or len(response) > 150:
        return None
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/corpus_claim_extraction_v2_sections.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print("Chunking corpus by section...", flush=True)
    chunks = ingest_pdf_directory_by_section(PAPERS_DIR)
    filtered = v2.keyword_prefilter(chunks)
    print(f"{len(chunks)} section chunks total, {len(filtered)} survive the broad keyword pre-filter", flush=True)

    print("Loading local LLM (mlx-community/Llama-3.2-3B-Instruct-4bit)...", flush=True)
    generate_fn = make_mlx_generate_fn()

    out_path = Path(args.out)
    results = []
    t_start = time.time()
    for i, chunk in enumerate(filtered):
        phrases = []
        for _ in range(args.repeats):
            claim = extract_claim(chunk.text, generate_fn)
            if claim is not None:
                mapped = map_phrase_to_known_concept(claim)
                phrases.append({"phrase": claim, "mapped_concept": mapped})

        # map_phrase_to_known_concept returns str | list[str] | None (a
        # laterality phrase matches both right_side and left_side, since the
        # keyword list can't tell direction) -- flatten to one vote per
        # concept per repeat before counting, rather than assuming a single
        # hashable value.
        concept_votes = []
        for p in phrases:
            mapped = p["mapped_concept"]
            if mapped is None:
                continue
            concept_votes.extend(mapped if isinstance(mapped, list) else [mapped])
        concept_counts = Counter(concept_votes)
        consistent_concept = None
        for concept, count in concept_counts.most_common(1):
            if count >= 2:
                consistent_concept = concept

        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "repeats": phrases,
                "consistent_concept": consistent_concept,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(filtered) - (i + 1)) / rate if rate > 0 else float("nan")
            print(
                f"[{i + 1}/{len(filtered)}] elapsed={elapsed / 60:.1f}min "
                f"est_remaining={remaining / 60:.1f}min consistent_so_far="
                f"{sum(1 for r in results if r['consistent_concept'])}",
                flush=True,
            )
            out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Done. Wrote {len(results)} chunk records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
