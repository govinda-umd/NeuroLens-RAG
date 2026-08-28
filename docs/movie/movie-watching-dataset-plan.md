# Movie-Watching Dataset — Acquisition & Extension Plan

> Extends Misra, Surampudi, Venkatesh, Limbachia, Jaja & Pessoa (2021), *"Learning brain dynamics for decoding and predicting individual differences,"* PLoS Comput Biol 17(9):e1008943 — the GRU-decoding precursor to NeuroLens-RAG, co-authored by the project owner. That paper used HCP movie-watching fMRI for 15-way movie-clip classification; this plan brings the *current* system (Case 1/2/3 multi-paradigm representation learning, CAV/TCAV, claim-first RAG verification) to the same data, as the direct stress test of `docs/end-to-end-report.md` §7's hypothesis that the cross-representation convergence findings depend on MOTOR's clean block structure.

## 1. What the precursor paper actually did (verified by reading the paper directly, not recalled)

- **Data**: HCP movie-watching fMRI, Schaefer-300 parcellation (same atlas NeuroLens-RAG already uses) — `Nx = 300` ROI time series, z-scored per run.
- **Subjects**: 176 total — 100 for training, 76 held out as a completely untouched test set. 5-fold CV on the training set for hyperparameter selection, then retrained on all of it.
- **Task**: 15-way movie-clip classification at every timepoint (which of 15 clips is currently playing), evaluated as accuracy over time — sharp rise over the first 60s, stabilizing by ~90s at 89.46% mean accuracy (chance ≈ 8.40% by permutation test).
- **Model**: single-layer GRU, 32 hidden units (`Nh=32`), softmax classifier head. Same family as NeuroLens-RAG's `GRUDecoder`, much smaller.
- **The precursor paper's own baseline check — directly relevant to `docs/end-to-end-report.md` §3.2**: a parameter-matched feed-forward network (1 hidden layer, 103 units, 32,563 params vs. the GRU's 32,559) scored only **44.86%**, and temporal shuffling of the *GRU's own* input dropped it to 54.14%. **This is the opposite pattern from what the new MOTOR baseline check found** (§3.2: a flat MLP ties GRU on MOTOR, p=0.73). On movie-watching, a parameter-matched non-recurrent model loses by ~45 points — real, direct evidence that temporal structure matters far more for naturalistic, continuous stimuli than for MOTOR's clean, discrete blocks. This single comparison is close to a pre-registered confirmation of the block-structure hypothesis, using the project owner's own prior result.
- **No structural connectivity anywhere in the precursor paper** — SC/DTI (§2 below) is a genuinely new modality this project adds, not a re-run of prior work.

## 2. Data location, verified directly against S3 (not assumed)

Movie-watching runs live in the **same** `hcp-openaccess` S3 bucket and the **same** `HCP_1200/<subject>/` prefix as MOTOR — not a separate dataset or bucket. The difference is per-subject: only subjects who completed the 7T protocol have these result folders at all.

```
HCP_1200/<subject>/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/
HCP_1200/<subject>/MNINonLinear/Results/tfMRI_MOVIE2_7T_PA/
HCP_1200/<subject>/MNINonLinear/Results/tfMRI_MOVIE3_7T_PA/
HCP_1200/<subject>/MNINonLinear/Results/tfMRI_MOVIE4_7T_AP/
```

Confirmed by direct listing: subject `100307` (already in the current MOTOR pool) has **no** 7T folders at all; subject `100610` has the full 7T battery (4 movie runs + 7T resting-state + retinotopy). **7T movie-watching eligibility does not track MOTOR-task eligibility** — this is a real, separate subject-availability question, not an assumption.

## 3. Subject overlap with the current pool — checked directly, and it's small

The current 90-subject bootstrap pool (`results/case1_bootstrap_100resamples.json`) was selected for 3T MOTOR-task completeness, with no consideration of 7T eligibility. Real discovery scan against exactly these 90 subjects (`data/subject_discovery_dti_movie.json`, run 2026-08-27):

| | Count | % of the 90 |
|---|---|---|
| Have diffusion/DTI data | 85 | 94% |
| Have 7T movie-watching data | **11** | **12%** |
| Have both | 11 | 12% |

**7T eligibility is genuinely rare in this pool, confirming the concern in §2 rather than an assumption.** 11 subjects is too few to bootstrap the way MOTOR's 30-resample studies do. Two paths forward, not yet decided between:

1. **Search from the other direction.** HCP-YA's full 7T cohort is ~184 subjects total, out of 1,113 — rare in *any* random subset, so starting from "which of the current 90 have 7T" was close to the worst-case search order. Better: enumerate the ~184 subjects known to have 7T movie data first (a single, bounded scan, not 90 more guesses), then check *those* for MOTOR + DTI eligibility. Likely yields a larger, still-modest, joint-overlap set than 11 — not yet run.
2. **Accept two mostly-different subject cohorts.** MOTOR/DTI on the current 90, movie-watching on its own ~184-subject (or eligible subset) cohort. Still a valid, real stress test of the §7 convergence hypothesis, just not a within-subject three-modality comparison. Cheaper, available immediately, no further discovery work needed.

**Recommendation: try (1) first — it's one more bounded S3 scan, not a new data commitment — and fall back to (2) if the joint-overlap count still comes back small.** Not run yet; this document stops at the open question deliberately rather than guessing which path pays off.

## 4. Proposed preprocessing pipeline (mirrors `02_data_complete.ipynb`'s MOTOR pipeline, task-specific parts swapped)

```
HCP S3 (tfMRI_MOVIE{1,2,3,4}_7T_{AP,PA})
  ↓
download one run into a temporary directory
  ↓
validate NIfTI + movement regressors (no EV/condition files -- movie-watching has no discrete conditions)
  ↓
extract motion-cleaned Schaefer-300 ROI time series (same atlas, same masker as MOTOR)
  ↓
label each timepoint with its clip identity + within-clip timepoint (from HCP's published movie-segment timing files, not derived)
  ↓
save final processed bundle, delete raw source files
```

**Real differences from MOTOR's pipeline, not just a relabeling:**
- No `y_hrf` — there's no discrete-condition GLM regressor for continuous naturalistic viewing. The precursor paper's target is purely the clip-identity label; NeuroLens-RAG's Case 3 (self-supervised, currently aligned to `y_hrf`) needs a different alignment target for movie data — the natural candidate is inter-subject correlation structure or the clip's own visual/audio features, not HRF. Not yet designed; flagged here rather than guessed at.
- No `valid_mask` frame-scrubbing precedent to reuse as-is — motion scrubbing thresholds for 7T movie-watching runs (longer, ~15 min each) may need separate validation.
- Windowing convention (32 TRs, stride 2) is MOTOR-specific, chosen around HRF-scale timing for a ~12s condition block. Movie-watching has no block structure to size a window against — the precursor paper classified at *every* timepoint using the GRU's running hidden state, not fixed windows. Whether to keep NeuroLens-RAG's fixed-window convention or adopt the precursor paper's continuous-decoding convention is an open design choice, not resolved here.

## 5. What NeuroLens-RAG adds beyond the 2021 paper, concretely

1. **Three paradigms, not one.** The 2021 paper only ever trained a supervised GRU classifier. Case 2 (contrastive, against clip-identity or scene-content prototypes) and Case 3 (self-supervised, against a to-be-designed alignment target — §4) extend this to the same three-paradigm comparison already run on MOTOR.
2. **CAV/TCAV concept probing**, not just saliency/lesion analysis. The 2021 paper used saliency maps and lesion analysis (§1) for interpretability — real, but a strictly weaker claim than CAV/TCAV's "does the decision depend on a named concept" (`docs/end-to-end-report.md` §4). Concepts here would be scene-content categories (e.g. faces present, indoor/outdoor, dialogue vs. action) rather than motor effectors — needs its own concept-definition pass, not a reuse of `CONCEPT_DEFINITIONS`.
3. **Literature-grounded verification.** The 2021 paper made no contact with independent literature verification at all. A movie-watching literature corpus (naturalistic-viewing fMRI, inter-subject correlation studies, scene-segmentation literature) would need its own corpus build, separate from the current motor-cortex-organization corpus (`docs/hcp-dataset-extension-options.md` doesn't cover this — a literature corpus is dataset-specific, not just a data-download question).

## 6. Honest scope check

This is not a small extension. Items still undesigned, not just unbuilt: Case 3's alignment target for continuous data (§4), the concept taxonomy for CAV/TCAV on movie content (§5.2), and a movie-watching-specific literature corpus (§5.3). This document covers the *data acquisition and preprocessing* layer only — the representation-learning and verification layers need their own design passes once real data is in hand, the same way Case 2/3's design predated their implementation for MOTOR (`case2-3-design-plan.md`).
