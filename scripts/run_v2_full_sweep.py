"""Full claim-first verification sweep (v2) over every self-consistent
literature claim mined from the section-based corpus extraction.

Caches everything reusable across claims (corpus embeddings, the 6 trained
representations at resample 0, their fitted differentiable heads, and any
CAV direction already fit for a given (representation, concept) pair) so
the per-claim marginal cost is dominated by the 2 LLM calls (stance +
narration), not by reloading models.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from neurolens import verification_v2 as v2
from neurolens.concepts import EXTENDED_CONCEPT_DEFINITIONS, train_cav, tcav_score
from neurolens.data_setup import make_dataloaders
from neurolens.pipeline import make_mlx_generate_fn
from neurolens.retrieval import (
    embed_chunks,
    ingest_pdf_directory,
    load_embedding_model,
    load_reranker,
    retrieve_and_rerank,
)

EXTRACTION_PATH = Path("results/corpus_claim_extraction_v2_sections.json")
OUT_PATH = Path("results/v2_full_sweep_results.json")
RESAMPLE = 0


def unique_consistent_claims() -> list[tuple[str, str]]:
    records = json.loads(EXTRACTION_PATH.read_text())
    seen = set()
    claims = []
    for r in records:
        if not r.get("consistent_concept"):
            continue
        for rep in r["repeats"]:
            mapped = rep["mapped_concept"]
            hit = r["consistent_concept"] == mapped or (isinstance(mapped, list) and r["consistent_concept"] in mapped)
            if hit and rep["phrase"] not in seen:
                seen.add(rep["phrase"])
                claims.append((rep["phrase"], r["consistent_concept"]))
                break
    return claims


def main() -> None:
    device = torch.device("cpu")
    t_start = time.time()

    print("Loading corpus + embeddings + reranker + LLM...", flush=True)
    chunks = ingest_pdf_directory(Path("data/papers"))
    embedding_model = load_embedding_model()
    embeddings = embed_chunks(chunks, embedding_model)
    reranker = load_reranker()
    generate_fn = make_mlx_generate_fn()

    print("Loading TCAV lookup + splits + resample-0 dataloaders...", flush=True)
    lookup = v2.load_precomputed_tcav_lookup()
    splits = v2.load_splits()
    split0 = splits[RESAMPLE]
    train_loader, _, test_loader, info = make_dataloaders(
        v2.PROCESSED_ROOT, batch_size=64,
        train_subjects=split0["train_subjects"], val_subjects=split0["val_subjects"], test_subjects=split0["test_subjects"],
    )

    print("Pre-loading all 6 representations at resample 0...", flush=True)
    reps = {}
    for case, arch in v2.REPRESENTATIONS:
        model = v2.load_representation(case, arch, RESAMPLE, info, embedding_model, device)
        wrapped = v2.fit_differentiable_head(model, case, train_loader, device, num_classes=info["num_classes"])
        feats, labels = v2.get_features_and_labels(wrapped, case, train_loader, device)
        reps[(case, arch)] = {"model": wrapped, "feats": feats, "labels": labels, "cav_cache": {}}
        print(f"  {case}/{arch} ready", flush=True)

    claims = unique_consistent_claims()
    print(f"{len(claims)} unique self-consistent claims to run", flush=True)

    results = []
    for i, (claim_text, extracted_concept) in enumerate(claims):
        t0 = time.time()
        concept_weights = v2.soft_concept_mapping(claim_text, embedding_model)
        ranked = v2.representation_rank_bootstrap(concept_weights, lookup)
        finding = v2.interpret_representation_ranking(ranked)
        winner_row = ranked.iloc[0]
        case, arch = winner_row["case"], winner_row["arch"]
        dominant_concept = max(concept_weights, key=concept_weights.get)

        rep = reps[(case, arch)]
        if dominant_concept not in rep["cav_cache"]:
            positive_classes, negative_classes = EXTENDED_CONCEPT_DEFINITIONS[dominant_concept]
            rep["cav_cache"][dominant_concept] = train_cav(rep["feats"], rep["labels"], positive_classes, negative_classes)
        cav = rep["cav_cache"][dominant_concept]

        attr = v2.concept_attribution(rep["model"], case, test_loader, cav["direction"], device)
        rsn = v2.rsn_consensus_from_attribution(attr)

        positive_classes, _ = EXTENDED_CONCEPT_DEFINITIONS[dominant_concept]
        per_class = [
            tcav_score(rep["model"], rep["feats"], rep["labels"], c, cav["direction"], device)["tcav_score"]
            for c in positive_classes
        ]
        per_class = [s for s in per_class if s is not None]
        this_tcav = float(sum(per_class) / len(per_class)) if per_class else float("nan")

        second_query = v2.build_second_query(claim_text, rsn["consensus_network"], dominant_concept)
        retrieved = retrieve_and_rerank(
            second_query, model=embedding_model, embeddings=embeddings, chunks=chunks,
            reranker=reranker, candidate_k=20, top_k=1,
        )
        top_chunk_text = retrieved.iloc[0]["text"]
        rerank_score = float(retrieved.iloc[0]["rerank_score"])
        stance_result = v2.compute_stance(second_query, top_chunk_text, rerank_score, generate_fn)
        stance = stance_result["stance"]

        verdict = v2.deterministic_verdict(stance, this_tcav)
        narration = generate_fn(
            v2.build_synthesis_prompt(claim_text, stance or "UNCLEAR", verdict, this_tcav, case, arch, dominant_concept, rsn["consensus_network"])
        ).strip()

        results.append(
            {
                "claim": claim_text,
                "extracted_concept": extracted_concept,
                "dominant_concept": dominant_concept,
                "concept_weight": concept_weights[dominant_concept],
                "winning_case": case,
                "winning_arch": arch,
                "combined_tcav_30resamples": float(winner_row["mean_combined_tcav"]),
                "p_rank1": float(winner_row["p_rank1"]),
                "frac_ties_at_max": float(winner_row["frac_ties_at_max"]),
                "representation_finding": finding,
                "cav_probe_accuracy": cav["probe_accuracy"],
                "tcav_this_resample": this_tcav,
                "consensus_network": rsn["consensus_network"],
                "second_pass_source": retrieved.iloc[0]["source_file"],
                "second_pass_rerank_score": float(retrieved.iloc[0]["rerank_score"]),
                "stance": stance,
                "stance_gate": stance_result["gate"],
                "stance_quote": stance_result["quote"],
                "verdict": verdict,
                "narration": narration,
            }
        )
        print(f"[{i + 1}/{len(claims)}] ({time.time() - t0:.1f}s) {case}/{arch} {dominant_concept} -> {verdict}", flush=True)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Wrote {len(results)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
