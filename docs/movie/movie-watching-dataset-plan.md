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

**Resolved by searching from the other direction, per the recommendation this section originally made.** Checking the current 90-subject pool for rare 7T eligibility found only 11/90 (12%) — close to the worst-case search order, since 7T eligibility (184/1,113 subjects HCP-YA-wide) is rare in *any* arbitrary subset. Reversing the search — full HCP-YA discovery scan, all 1,113 subjects, 2 S3 calls each (`scripts/run_full_hcp_discovery_scan.py`, `data/hcp_full_discovery_scan.{json,csv}`, run 2026-08-27) — gives a completely different picture:

| | Count | % of 1,113 |
|---|---|---|
| Have DTI | 1,065 | 96% |
| Have MOTOR | 1,086 | 98% |
| Have 7T movie-watching | 184 | 17% |
| **DTI + MOTOR + 7T movie, all three** | **179** | **16%** |
| DTI + movie + all 7 standard tasks (full battery) | 174 | 16% |

**179 subjects qualify for a genuine within-subject three-modality comparison — 179/184 (97%) of every 7T-movie-eligible subject also has DTI and MOTOR.** 7T eligibility, not DTI or MOTOR, is the actual bottleneck; once a subject cleared the 7T protocol, the rest of the battery came essentially for free. Full ID list: `data/hcp_triple_modality_eligible_subjects.json`. Only 11 of these 179 are already in the current 90-subject pool — this is functionally a *different* subject cohort from the one MOTOR's existing results are built on, not an extension of it.

**Decision this sets up, not yet made**: build the movie-watching (and future SC) work on this new 179-subject cohort specifically (loses direct comparability with the exact subjects behind the existing Case 1/2/3 MOTOR numbers, gains a real within-subject three-modality design and a pool comparable in size to the current bootstrap's 90), or re-run MOTOR/DTI on this same 179-subject cohort too so every result going forward shares one subject pool. The second option is more work (a MOTOR re-run) but avoids ever having to caveat "these two results used different people."

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

**Extraction run, completed (2026-09-01 12:52 → 2026-09-02 11:30, ~22.6 hours):** `scripts/run_movie_extraction_batch.py` ran the pipeline above (`scripts/movie_roi_extraction.py`) over all 4 movie runs for the same 174-subject cohort the DTI/SC batch used, deliberately, so movie and structural data exist for the same subjects. **685/696 runs succeeded, 11 failed** — all real 404s, not crashes: 6 subjects are missing 1–2 of their 4 movie runs on S3 (5 missing MOVIE3+MOVIE4, 1 missing MOVIE2 only), meaning they only partially completed the 7T movie-watching protocol. The Aug 27 discovery scan's `has_movie` flag only checked for *any* movie-related folder under `Results/`, not all 4 specific runs — a real gap in that scan, not previously visible until this batch actually tried to download each run. **168/174 subjects have the complete 4-run set; 6 have 2–3 runs.** Every successful run's `X.npy` validated finite with the correct ROI count, and each run type has an exactly consistent timepoint count across every subject who has it (MOVIE1=921 TRs, MOVIE2=918, MOVIE3=915, MOVIE4=901 — no truncated downloads). Total processed footprint: 731MB, against ~0.95TB of raw functional volumes downloaded and deleted per-run along the way (confirmed via S3 `HeadObject` before committing to the batch: 4 runs × ~1.4GB × 174 subjects).

## 5. What NeuroLens-RAG adds beyond the 2021 paper, concretely

1. **Three paradigms, not one.** The 2021 paper only ever trained a supervised GRU classifier. Case 2 (contrastive, against clip-identity or scene-content prototypes) and Case 3 (self-supervised, against a to-be-designed alignment target — §4) extend this to the same three-paradigm comparison already run on MOTOR.
2. **CAV/TCAV concept probing**, not just saliency/lesion analysis. The 2021 paper used saliency maps and lesion analysis (§1) for interpretability — real, but a strictly weaker claim than CAV/TCAV's "does the decision depend on a named concept" (`docs/end-to-end-report.md` §4). Concepts here would be scene-content categories (e.g. faces present, indoor/outdoor, dialogue vs. action) rather than motor effectors — needs its own concept-definition pass, not a reuse of `CONCEPT_DEFINITIONS`.
3. **Literature-grounded verification.** The 2021 paper made no contact with independent literature verification at all. A movie-watching literature corpus (naturalistic-viewing fMRI, inter-subject correlation studies, scene-segmentation literature) would need its own corpus build, separate from the current motor-cortex-organization corpus (`docs/hcp-dataset-extension-options.md` doesn't cover this — a literature corpus is dataset-specific, not just a data-download question).

## 6. Honest scope check

This is not a small extension. Items still undesigned, not just unbuilt: Case 3's alignment target for continuous data (§4), the concept taxonomy for CAV/TCAV on movie content (§5.2), and a movie-watching-specific literature corpus (§5.3). This document covers the *data acquisition and preprocessing* layer only — the representation-learning and verification layers need their own design passes once real data is in hand, the same way Case 2/3's design predated their implementation for MOTOR (`case2-3-design-plan.md`).
