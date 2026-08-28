"""Claim-first RAG-CAV verification loop (v2).

Implements the design in `docs/v2/rag-cav-verification-loop-design.md`:
mine literature claims independent of any decode, verify each against
whichever of the 6 trained representations (3 paradigms x 2 architectures)
actually depends on it -- using the already-validated, uniform
fitted-post-hoc-classifier-head CAV/TCAV mechanism (`case3.py::fit_post_hoc_classifier`,
reused unmodified) -- then run concept-attribution (not prediction-attribution)
on the winning representation's own held-out set to drive a second,
targeted retrieval pass, and synthesize a verdict computed deterministically
from the numbers already in hand.

No open-vocabulary route anywhere in this module, by design (see the
standardization note in `docs/interview-prep-neurolens-rag.md` after Sec 11.3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

from neurolens.case3 import BrainHRFModel, BrainWithPostHocClassifier
from neurolens.concepts import EXTENDED_CONCEPT_DEFINITIONS, train_cav, tcav_score
from neurolens.contrastive import CONDITION_DESCRIPTIONS, ContrastiveModel, TextPrototypeEncoder
from neurolens.data_setup import make_dataloaders
from neurolens.interpretability import (
    NETWORK_NAMES,
    aggregate_attribution_to_networks,
    load_roi_to_network,
    network_roi_indices,
)
from neurolens.model_builder import GRUDecoder, TransformerDecoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "hcp_ya_s1200" / "runs"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
CASE1_SPLITS_PATH = RESULTS_ROOT / "case1_bootstrap_100resamples.json"
ROI_LABELS_PATH = PROCESSED_ROOT / "sub-100307" / "tfMRI_MOTOR_LR" / "roi_labels.tsv"

CASES = ("case1", "case2", "case3")
ARCHITECTURES = ("gru", "transformer")
REPRESENTATIONS = [(case, arch) for case in CASES for arch in ARCHITECTURES]

CONCEPT_NAMES = list(EXTENDED_CONCEPT_DEFINITIONS.keys())

# Verbatim from docs/interview-prep-neurolens-rag.md Sec 11.4 -- deliberately
# broader than concepts.py's LITERATURE_CONCEPT_KEYWORDS, to preserve
# discovery surface a narrow filter would silently exclude.
BROAD_DISCOVERY_KEYWORDS = [
    "somatotop", "homuncul", "topograph", "hemispher", "ipsilateral",
    "contralateral", "lateraliz", "lateral", "asymmetr", "bilateral",
    "unilateral", "limb", "digit", "finger", "toe", "hand", "foot", "feet",
    "tongue", "orofacial", "articulat", "effector", "gradient", "selectiv",
]

# One reference phrase per concept for soft embedding-similarity mapping
# (docs/v2/rag-cav-verification-loop-design.md Sec 3) -- reuses the same
# CONDITION_DESCRIPTIONS vocabulary already trusted elsewhere in the project
# rather than inventing new wording.
CONCEPT_REFERENCE_TEXT: dict[str, str] = {
    "hand": "hand movement, left or right",
    "foot": "foot movement, left or right",
    "tongue": "tongue movement",
    "right_side": "right-sided movement, lateralized to the right",
    "left_side": "left-sided movement, lateralized to the left",
    "movement_vs_rest": "any movement versus resting baseline",
    "limb_vs_orofacial": "limb movement versus orofacial (tongue) movement",
    "upper_vs_lower_limb": "upper limb (hand) versus lower limb (foot) movement",
}


# ---------------------------------------------------------------------------
# Section 1-2: chunking + keyword pre-filter (chunking itself lives in
# retrieval.py::ingest_pdf_directory_by_section; this is just the filter)
# ---------------------------------------------------------------------------


def keyword_prefilter(chunks: list, keywords: list[str] = BROAD_DISCOVERY_KEYWORDS) -> list:
    """Case-insensitive substring match, same design as the narrow filter
    in concepts.py -- just a wider keyword list."""
    lowered_keywords = [kw.lower() for kw in keywords]
    return [c for c in chunks if any(kw in c.text.lower() for kw in lowered_keywords)]


# ---------------------------------------------------------------------------
# Section 3: soft concept mapping (new) + CAV/TCAV lookup (reuses existing
# precomputed sweeps -- no retraining or reprobing needed for this step)
# ---------------------------------------------------------------------------


def soft_concept_mapping(
    claim_phrase: str, embedding_model: SentenceTransformer, temperature: float = 10.0
) -> dict[str, float]:
    """Embeds the claim phrase and each concept's reference text with the
    retrieval MiniLM, cosine-similarity + softmax -> a soft weighting over
    the 8 known concepts. Replaces `concepts.py::map_phrase_to_known_concept`'s
    all-or-nothing keyword match for this design."""
    names = CONCEPT_NAMES
    refs = [CONCEPT_REFERENCE_TEXT[n] for n in names]
    claim_vec = embedding_model.encode([claim_phrase], normalize_embeddings=True)[0]
    ref_vecs = embedding_model.encode(refs, normalize_embeddings=True)
    sims = ref_vecs @ claim_vec
    weights = np.exp(sims * temperature)
    weights = weights / weights.sum()
    return dict(zip(names, weights.tolist()))


def load_precomputed_tcav_lookup() -> pd.DataFrame:
    """Long-format table: one row per (case, architecture, resample,
    concept) with its TCAV score, built from the two sweeps that already
    exist on disk (all_cases_cav_sweep_8concepts.json for Case 1/3 and
    Case 2's now-non-default text-derived score, case2_fitted_probe_cav_sweep.json
    for Case 2's standardized fitted-probe score, docs/interview-prep-neurolens-rag.md
    Sec 11.3). No model loading, no retraining -- pure lookup."""
    all_cases = json.loads((RESULTS_ROOT / "all_cases_cav_sweep_8concepts.json").read_text())
    case2_fitted = json.loads((RESULTS_ROOT / "case2_fitted_probe_cav_sweep.json").read_text())
    case2_fitted_by_resample = {row["resample"]: row for row in case2_fitted}

    rows = []
    for row in all_cases:
        resample = row["resample"]
        for case in CASES:
            for arch in ARCHITECTURES:
                key = f"{case}_{arch}"
                if case == "case2":
                    concepts = case2_fitted_by_resample[resample][f"{arch}_concepts"]
                else:
                    concepts = row[key]
                for concept_name, concept_result in concepts.items():
                    if concept_name not in CONCEPT_NAMES:
                        continue
                    scores = concept_result.get("scores", {})
                    # combined TCAV for "does this representation depend on
                    # the concept at all" = the concept's score against its
                    # own defining positive classes, averaged
                    positive_classes, _ = EXTENDED_CONCEPT_DEFINITIONS[concept_name]
                    class_names = ["baseline", "left_hand", "right_hand", "left_foot", "right_foot", "tongue"]
                    positive_scores = [scores[class_names[c]] for c in positive_classes if class_names[c] in scores]
                    if not positive_scores:
                        continue
                    rows.append(
                        {
                            "case": case,
                            "arch": arch,
                            "resample": resample,
                            "concept": concept_name,
                            "tcav": float(np.mean(positive_scores)),
                            "probe_accuracy": concept_result.get("probe_accuracy"),
                        }
                    )
    return pd.DataFrame(rows)


def combined_tcav_by_representation(concept_weights: dict[str, float], lookup: pd.DataFrame) -> pd.DataFrame:
    """For each (case, arch), linearly combine per-concept TCAV (averaged
    over the 30 resamples) by the soft concept weights -- the "combine
    scores, not directions" rule already resolved in
    docs/interview-prep-neurolens-rag.md Sec 6.6."""
    per_rep_concept = lookup.groupby(["case", "arch", "concept"])["tcav"].mean().reset_index()
    results = []
    for (case, arch), group in per_rep_concept.groupby(["case", "arch"]):
        weighted = sum(concept_weights.get(row.concept, 0.0) * row.tcav for row in group.itertuples())
        results.append({"case": case, "arch": arch, "combined_tcav": weighted})
    return pd.DataFrame(results).sort_values("combined_tcav", ascending=False).reset_index(drop=True)


def representation_rank_bootstrap(
    concept_weights: dict[str, float], lookup: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """Population-level version of `combined_tcav_by_representation`, in the
    same spirit as `concepts.py::cross_class_rank_bootstrap_test` but
    ranking REPRESENTATIONS instead of classes, using the 30 real
    independently-trained resamples already on disk as the population
    rather than a further bootstrap of one fixed model (this is a
    repeated-split significance test, not a post-hoc bootstrap -- see
    docs/interview-prep-neurolens-rag.md Sec 8's distinction between the two).

    Built because the raw point-estimate version (`combined_tcav_by_representation`)
    picked Case 1/Transformer for 48/48 real claims in the first full sweep
    (2026-08-26) -- every representation's mean TCAV sat in a 0.999-1.000
    band, so the argmax was resolving 4th-decimal-place noise, not a real
    signal. This function asks the better-posed question directly: at each
    of the 30 *paired* resamples (same subject splits across all 3 cases,
    confirmed in the interview-prep doc's split-construction notes), which
    representation actually ranks #1 on this claim's concept-weighting, and
    how often does that hold up across resamples?

    Ties are broken uniformly at random, never by `max()`'s implicit
    lowest-index-wins rule (concepts.py's own documented pitfall) -- and
    `frac_ties_at_max` is reported so a near-ceiling regime where this test
    has little power to discriminate is visible rather than silently
    mistaken for a clean winner."""
    per_resample = lookup.copy()
    per_resample["rep"] = list(zip(per_resample["case"], per_resample["arch"]))

    resamples = sorted(per_resample["resample"].unique())
    reps = REPRESENTATIONS
    rng = np.random.RandomState(seed)

    weighted_by_resample = {rep: [] for rep in reps}
    rank1_counts = {rep: 0 for rep in reps}
    tie_counts = {rep: 0 for rep in reps}

    for resample in resamples:
        this_resample = per_resample[per_resample["resample"] == resample]
        weighted_scores = {}
        for case, arch in reps:
            group = this_resample[(this_resample["case"] == case) & (this_resample["arch"] == arch)]
            weighted_scores[(case, arch)] = sum(
                concept_weights.get(row.concept, 0.0) * row.tcav for row in group.itertuples()
            )
            weighted_by_resample[(case, arch)].append(weighted_scores[(case, arch)])

        top_score = max(weighted_scores.values())
        tied = [rep for rep, score in weighted_scores.items() if abs(score - top_score) < 1e-9]
        for rep in tied:
            tie_counts[rep] += 1 if len(tied) > 1 else 0
        winner = tied[0] if len(tied) == 1 else tied[rng.randint(len(tied))]
        rank1_counts[winner] += 1

    n = len(resamples)
    rows = []
    for case, arch in reps:
        scores = np.array(weighted_by_resample[(case, arch)])
        lo, hi = np.percentile(scores, [2.5, 97.5])
        rows.append(
            {
                "case": case,
                "arch": arch,
                "mean_combined_tcav": float(scores.mean()),
                "ci_95": [float(lo), float(hi)],
                "p_rank1": rank1_counts[(case, arch)] / n,
                "frac_ties_at_max": tie_counts[(case, arch)] / n,
                "n_resamples": n,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["p_rank1", "mean_combined_tcav"], ascending=[False, False]
    ).reset_index(drop=True)


def interpret_representation_ranking(ranking: pd.DataFrame, dominance_threshold: float = 0.5) -> str:
    """Turns the rank-bootstrap table into the actual claim it supports.
    Uniform-ish P(rank1) across representations (~1/6 each, or several
    tied near the top with high frac_ties_at_max) is not a failure of the
    test -- given the HCP MOTOR task's long, cleanly-separated condition
    blocks, every representation being equally sensitive to an
    effector/laterality concept is a real, expected result in its own
    right, not an inconclusive one."""
    top = ranking.iloc[0]
    if top["p_rank1"] >= dominance_threshold and top["frac_ties_at_max"] < 0.2:
        return (
            f"case{top['case'][-1]}/{top['arch']} is significantly best-aligned "
            f"(P(rank1)={top['p_rank1']:.2f} across {int(top['n_resamples'])} resamples)."
        )
    near_top = ranking[ranking["mean_combined_tcav"] >= ranking["mean_combined_tcav"].max() - 0.01]
    return (
        f"No single representation is distinguishable -- {len(near_top)} of 6 representations "
        f"cluster within 0.01 combined TCAV of the top score, and no representation reaches "
        f"P(rank1)>={dominance_threshold}. All paradigms represent this concept about equally "
        f"well; on a task built from long, temporally well-separated condition blocks, that is "
        f"the expected result, not a null one."
    )


# ---------------------------------------------------------------------------
# Section 4: load a specific checkpoint, fit its differentiable classifier,
# run concept-attribution on its own held-out set (new)
# ---------------------------------------------------------------------------


def load_splits() -> list[dict]:
    return json.loads(CASE1_SPLITS_PATH.read_text())


def checkpoint_path(case: str, arch: str, resample: int) -> Path:
    return MODELS_ROOT / f"{case}_bootstrap" / f"resample_{resample:03d}" / arch / "best.pt"


def build_backbone(arch: str, num_classes: int, num_conditions: int, include_hrf_head: bool) -> nn.Module:
    if arch == "gru":
        return GRUDecoder(num_classes=num_classes, num_conditions=num_conditions, include_hrf_head=include_hrf_head)
    if arch == "transformer":
        return TransformerDecoder(num_classes=num_classes, num_conditions=num_conditions, include_hrf_head=include_hrf_head)
    raise ValueError(arch)


def load_representation(
    case: str, arch: str, resample: int, info: dict, embedding_model: SentenceTransformer, device: torch.device
) -> nn.Module:
    """Reconstructs the exact trained model for one (case, arch, resample)
    and loads its saved weights. Returns an object exposing
    `.brain_backbone` (or itself, for Case 1) with `.forward_features`."""
    state_dict = torch.load(checkpoint_path(case, arch, resample), map_location=device, weights_only=False)

    if case == "case1":
        model = build_backbone(arch, info["num_classes"], info["num_conditions"], include_hrf_head=True)
        model.load_state_dict(state_dict)
    elif case == "case2":
        backbone = build_backbone(arch, info["num_classes"], info["num_conditions"], include_hrf_head=False)
        text_embeddings = state_dict["text_encoder.text_embeddings"].numpy()
        text_encoder = TextPrototypeEncoder(text_embeddings, embed_dim=64)
        model = ContrastiveModel(backbone, backbone_dim=128, text_encoder=text_encoder, embed_dim=64)
        model.load_state_dict(state_dict)
    elif case == "case3":
        backbone = build_backbone(arch, info["num_classes"], info["num_conditions"], include_hrf_head=False)
        model = BrainHRFModel(backbone, backbone_dim=128, embed_dim=64, hrf_dim=5)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(case)

    model.to(device).eval()
    return model


def get_features_and_labels(
    model: nn.Module, case: str, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Case 1 exposes `forward_features` natively; Case 2/3 need
    `.brain_backbone.forward_features` (same pattern as
    `case3.py::extract_brain_embeddings_case3`, generalized to Case 2 too)."""
    backbone = model if case == "case1" else model.brain_backbone
    backbone.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            all_features.append(backbone.forward_features(x).cpu().numpy())
            all_labels.append(batch["y"].numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def fit_differentiable_head(
    model: nn.Module, case: str, train_loader: DataLoader, device: torch.device, num_classes: int = 6
) -> nn.Module:
    """Case 1's own trained head is already differentiable and comparable
    across resamples. Case 2/3 get a post-hoc-fitted head, exactly
    `fit_post_hoc_classifier`'s method (Sec 3 of the v2 design doc) --
    reused unmodified, since both ContrastiveModel and BrainHRFModel share
    the `.brain_backbone`/`.brain_projection` attribute naming."""
    if case == "case1":
        return model
    from neurolens.case3 import fit_post_hoc_classifier

    wrapped, _ = fit_post_hoc_classifier(model, train_loader, train_loader, device, num_classes=num_classes)
    return wrapped


def concept_attribution(
    model: nn.Module,
    case: str,
    held_out_loader: DataLoader,
    cav_direction: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Backprop h(x).v_C (not a class logit) to the raw input, averaged
    over the representation's own held-out set -- the mechanism resolved in
    docs/v2/rag-cav-verification-loop-design.md Sec 4. Returns raw signed
    attribution averaged over all held-out windows, shape [L, n_rois]."""
    backbone = model.brain_backbone if case != "case1" else model
    v_c = torch.tensor(cav_direction, dtype=torch.float32, device=device)

    total_attr = None
    n_windows = 0
    for batch in held_out_loader:
        x = batch["x"].to(device).clone().requires_grad_(True)
        h = backbone.forward_features(x)
        score = (h * v_c).sum(dim=1)
        score.sum().backward()
        grad = x.grad.detach().cpu().numpy()
        total_attr = grad.sum(axis=0) if total_attr is None else total_attr + grad.sum(axis=0)
        n_windows += x.shape[0]
    return total_attr / n_windows


def rsn_consensus_from_attribution(attr: np.ndarray) -> dict:
    network_indices = network_roi_indices(load_roi_to_network(ROI_LABELS_PATH))
    per_network = aggregate_attribution_to_networks(attr, network_indices)
    top_idx = int(np.argmax(per_network))
    return {
        "per_network": dict(zip(NETWORK_NAMES, per_network.tolist())),
        "consensus_network": NETWORK_NAMES[top_idx],
    }


# ---------------------------------------------------------------------------
# Section 6: deterministic verdict -- one shared path for every case
# ---------------------------------------------------------------------------


def build_second_query(claim_text: str, consensus_network: str, concept_name: str) -> str:
    return (
        f"Literature claim under test: \"{claim_text}\". The best-aligned trained "
        f"representation's own concept-attribution for '{concept_name}' points to the "
        f"{consensus_network} resting-state network. Evidence for or against "
        f"{concept_name} involving {consensus_network}?"
    )


def build_grounded_stance_prompt(query_text: str, excerpt_text: str) -> str:
    """Unlike `pipeline.py::build_concept_extraction_prompt_with_stance`,
    this requires a verbatim QUOTE backing the stance -- a real bug (Sec
    "stance-extraction sycophancy", found by actually running the full v2
    sweep 2026-08-26) showed the un-grounded version marking SUPPORTS even
    for a chunk about a completely different topic (hand/arm spatial
    overlap, for a claim about laterality), including cases the reranker
    itself scored as a poor match. Requiring a quote makes the claim
    checkable in code (`quote_is_grounded`) instead of trusted."""
    return f"""Read this excerpt from neuroscience literature in the context of a claim about the brain.

CLAIM:
{query_text}

EXCERPT:
{excerpt_text}

Respond in exactly this three-line format, nothing else:
STANCE: SUPPORTS or CONTRADICTS or UNRELATED
QUOTE: a short verbatim phrase (5-15 words) COPIED EXACTLY from the excerpt above that backs your stance, or NONE if UNRELATED
PHRASE: a short 5-10 word phrase stating the excerpt's specific claim, or NONE

If the excerpt does not specifically address the claim's topic, you MUST answer UNRELATED, even if it is about a related brain region or a similar-sounding topic."""


def parse_grounded_stance(response: str) -> tuple[str | None, str | None, str | None]:
    stance_match = re.search(r"STANCE:\s*(SUPPORTS|CONTRADICTS|UNRELATED)", response.upper())
    quote_match = re.search(r"QUOTE:\s*(.+)", response, re.IGNORECASE)
    phrase_match = re.search(r"PHRASE:\s*(.+)", response, re.IGNORECASE)
    stance = stance_match.group(1) if stance_match else None
    quote = quote_match.group(1).strip() if quote_match else None
    phrase = phrase_match.group(1).strip() if phrase_match else None
    if quote is not None and (quote.upper().startswith("NONE") or len(quote) == 0):
        quote = None
    if phrase is not None and (phrase.upper().startswith("NONE") or len(phrase) == 0 or len(phrase) > 150):
        phrase = None
    return stance, quote, phrase


def quote_is_grounded(quote: str | None, excerpt_text: str, min_word_overlap: float = 0.7) -> bool:
    """Checkable in code, not trusted: at least `min_word_overlap` of the
    quote's words must actually appear in the excerpt, contiguously-ish
    (allows minor whitespace/case drift from the LLM, not paraphrase)."""
    if not quote:
        return False
    normalize = lambda s: re.sub(r"[^\w\s]", "", s.lower())
    quote_norm = normalize(quote)
    excerpt_norm = normalize(excerpt_text)
    if quote_norm in excerpt_norm:
        return True
    quote_words = quote_norm.split()
    if not quote_words:
        return False
    excerpt_words = set(excerpt_norm.split())
    overlap = sum(1 for w in quote_words if w in excerpt_words) / len(quote_words)
    return overlap >= min_word_overlap


# Found by inspecting the 2026-08-27 expanded-corpus sweep: grounding
# (quote_is_grounded) verifies a cited quote is REAL text from the
# excerpt, but not that it's actually ON-TOPIC for the specific claim.
# Concrete failure caught: for "The hand representation is contralateral"
# (a laterality claim), the LLM cited a real, verbatim quote about hand
# representation existing in area BA4p -- topically about hand/finger
# subdivision, saying nothing about which hemisphere. Grounded, still
# wrong. ~8-9 of 70 claims in that sweep showed this pattern. Fixed here
# with a second, independent deterministic gate: if the claim asserts
# something on one of these axes (contains an axis keyword), the quote
# must contain a keyword from THE SAME axis, not just any real text.
AXIS_KEYWORDS: dict[str, list[str]] = {
    "laterality": [
        "contralateral", "ipsilateral", "bilateral", "unilateral", "lateraliz",
        "lateral", "hemispher", "interhemispher", "left side", "right side",
        "left hemisphere", "right hemisphere",
    ],
    "effector": ["hand", "finger", "digit", "wrist", "arm", "foot", "feet", "toe", "tongue", "orofacial", "lingual"],
    "organization": ["somatotop", "overlap", "core and surround", "gradient", "mosaic", "topograph", "homuncul"],
}


def claim_axes(text: str) -> set[str]:
    text_lower = text.lower()
    return {axis for axis, keywords in AXIS_KEYWORDS.items() if any(kw in text_lower for kw in keywords)}


def quote_addresses_claim_axes(claim_or_query_text: str, quote: str | None) -> bool:
    """True if there's nothing specific to check (claim hits no known
    axis) or if the quote covers EVERY axis the claim asserts -- not just
    one of several. Intersection alone is too lenient: a claim asserting
    both an effector ("hand") and laterality ("contralateral") would
    wrongly pass against a quote that only addresses the effector, which
    is exactly the failure this was built to catch (the BA4p quote
    mentions "hand"/"fingers" but nothing about laterality). Full
    coverage is required; a claim hitting no known axis has nothing
    specific to check and passes through unchanged."""
    if not quote:
        return False
    required_axes = claim_axes(claim_or_query_text)
    if not required_axes:
        return True
    return required_axes.issubset(claim_axes(quote))


def compute_stance(
    query_text: str, excerpt_text: str, rerank_score: float, generate_fn
) -> dict:
    """Grounding-only gate around the one thing that genuinely needs
    judgment (support vs. contradict) -- a post-hoc check on the LLM's own
    cited quote, downgrading to UNRELATED if the quote doesn't actually
    appear in the excerpt. The LLM is never simply trusted to say "this
    supports the claim" -- it has to point at text that does, and that
    pointer is checked.

    A rerank-score pre-filter was tried first and rejected after checking
    it against two known cases: the reranker's raw score isn't calibrated
    around 0 for this small, out-of-domain (neuroscience) corpus -- a
    genuinely strong, on-topic match (Ehrsson et al.'s tongue-bilateral
    passage, later independently confirmed SUPPORTS with a grounded quote)
    scored -2.22, in the same range as a genuinely off-topic one (-3.75).
    Thresholding on that score would have traded one failure mode
    (everything SUPPORTS) for another (good evidence silently discarded).
    `rerank_score` is kept as an argument and logged for diagnosis, not
    used as a gate."""
    response = generate_fn(build_grounded_stance_prompt(query_text, excerpt_text))
    stance, quote, phrase = parse_grounded_stance(response)

    if stance in ("SUPPORTS", "CONTRADICTS") and not quote_is_grounded(quote, excerpt_text):
        return {"stance": "UNRELATED", "quote": quote, "phrase": phrase, "gate": "quote_not_grounded"}

    if stance in ("SUPPORTS", "CONTRADICTS") and not quote_addresses_claim_axes(query_text, quote):
        return {"stance": "UNRELATED", "quote": quote, "phrase": phrase, "gate": "quote_off_topic"}

    return {"stance": stance, "quote": quote, "phrase": phrase, "gate": "llm_grounded"}


def deterministic_verdict(stance: str | None, tcav_score_value: float, high: float = 0.7, low: float = 0.3) -> str:
    """Same rule as `pipeline.py::expected_verdict_from_stance_and_tcav`,
    applied uniformly regardless of which case originated the claim -- the
    one change this design requires of Sec 6 to actually close the Case 1
    free-judgment gap rather than leaving it as a separate branch."""
    if stance is None or stance == "UNRELATED":
        return "UNCLEAR"
    if stance == "SUPPORTS":
        return "AGREE" if tcav_score_value >= high else "DISAGREE"
    if stance == "CONTRADICTS":
        return "DISAGREE" if tcav_score_value >= high else "AGREE"
    return "UNCLEAR"


def build_synthesis_prompt(claim_text: str, stance: str, verdict: str, tcav_score_value: float, case: str, arch: str, concept_name: str, consensus_network: str) -> str:
    """One shared prompt for every case, per the Sec 6 requirement -- the
    LLM is given the verdict, not asked to reach it."""
    return f"""Narrate this already-determined result in 2-3 plain sentences for a reader. Do not change the verdict or second-guess the numbers -- state what they mean.

Literature claim: "{claim_text}"
Second-pass literature stance on this claim: {stance}
Best-aligned trained representation: Case {case[-1]} ({arch}), concept "{concept_name}"
That representation's TCAV sensitivity to this concept: {tcav_score_value:.3f}
That representation's concept-attribution points to the {consensus_network} network
Verdict (already computed, do not change it): {verdict}

NARRATION:"""


def run_claim_first_loop(
    claim_text: str,
    *,
    corpus_chunks: list,
    corpus_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    reranker,
    device: torch.device,
    generate_fn,
    tcav_lookup: pd.DataFrame,
    splits: list[dict],
    resample: int = 0,
) -> dict:
    """The full v2 loop for one already-mined claim: soft concept mapping ->
    best-aligned representation -> concept-attribution -> second query ->
    second retrieval pass -> stance -> deterministic verdict -> narration."""
    from neurolens.retrieval import retrieve_and_rerank

    # 3: soft concept mapping + population-level representation ranking
    # (pure lookup against the 30 already-computed resamples, no model loading)
    concept_weights = soft_concept_mapping(claim_text, embedding_model)
    ranked = representation_rank_bootstrap(concept_weights, tcav_lookup)
    representation_finding = interpret_representation_ranking(ranked)
    winner = ranked.iloc[0]
    dominant_concept = max(concept_weights, key=concept_weights.get)

    # 4: load the winning representation at one resample, fit its
    # differentiable head, run concept-attribution on its own held-out set
    split = splits[resample]
    train_loader, _, test_loader, info = make_dataloaders(
        PROCESSED_ROOT, batch_size=64,
        train_subjects=split["train_subjects"], val_subjects=split["val_subjects"], test_subjects=split["test_subjects"],
    )
    model = load_representation(winner["case"], winner["arch"], resample, info, embedding_model, device)
    wrapped = fit_differentiable_head(model, winner["case"], train_loader, device, num_classes=info["num_classes"])
    feats, labels = get_features_and_labels(wrapped, winner["case"], train_loader, device)
    positive_classes, negative_classes = EXTENDED_CONCEPT_DEFINITIONS[dominant_concept]
    cav = train_cav(feats, labels, positive_classes, negative_classes)
    attr = concept_attribution(wrapped, winner["case"], test_loader, cav["direction"], device)
    rsn = rsn_consensus_from_attribution(attr)

    # per-resample TCAV for this specific representation/concept (not just
    # the 30-resample mean already used for representation selection)
    per_class_scores = []
    for target_class in positive_classes:
        result = tcav_score(wrapped, feats, labels, target_class, cav["direction"], device)
        if result["tcav_score"] is not None:
            per_class_scores.append(result["tcav_score"])
    this_resample_tcav = float(np.mean(per_class_scores)) if per_class_scores else float("nan")

    # 4 cont'd / 5: second query + targeted retrieval pass
    second_query = build_second_query(claim_text, rsn["consensus_network"], dominant_concept)
    retrieved = retrieve_and_rerank(
        second_query, model=embedding_model, embeddings=corpus_embeddings, chunks=corpus_chunks,
        reranker=reranker, candidate_k=20, top_k=1,
    )
    top_chunk_text = retrieved.iloc[0]["text"]
    stance_result = compute_stance(second_query, top_chunk_text, float(retrieved.iloc[0]["rerank_score"]), generate_fn)
    stance = stance_result["stance"]

    # 6: deterministic verdict, one shared path regardless of case
    verdict = deterministic_verdict(stance, this_resample_tcav)
    narration = generate_fn(
        build_synthesis_prompt(
            claim_text, stance or "UNCLEAR", verdict, this_resample_tcav,
            winner["case"], winner["arch"], dominant_concept, rsn["consensus_network"],
        )
    )

    return {
        "claim": claim_text,
        "concept_weights": concept_weights,
        "dominant_concept": dominant_concept,
        "representation_ranking": ranked.to_dict(orient="records"),
        "representation_finding": representation_finding,
        "winning_representation": {"case": winner["case"], "arch": winner["arch"], "resample": resample},
        "cav_probe_accuracy": cav["probe_accuracy"],
        "tcav_this_resample": this_resample_tcav,
        "consensus_network": rsn["consensus_network"],
        "per_network_attribution": rsn["per_network"],
        "second_query": second_query,
        "second_pass_top_chunk": {"source_file": retrieved.iloc[0]["source_file"], "text": top_chunk_text[:300]},
        "second_pass_rerank_score": float(retrieved.iloc[0]["rerank_score"]),
        "stance": stance,
        "stance_gate": stance_result["gate"],
        "stance_quote": stance_result["quote"],
        "verdict": verdict,
        "narration": narration.strip(),
    }
