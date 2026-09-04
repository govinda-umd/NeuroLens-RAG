# Repo Restructuring Plan: Motor / Movie-Watching / Structural Split

**Superseded, 2026-09-04.** This plan proposed a subfolder split *within this repo*. The movie-watching and structural-connectivity work instead moved to its own separate repository, [naturalistic-brain-dynamics](../../naturalistic-brain-dynamics) — a cleaner boundary than any subfolder split could give, since that work had become a genuinely independent line of research rather than an extension of this repo's MOTOR-task + RAG-verification system. This document is kept for the historical record of the intermediate proposal; do not act on it.

---

> Proposal for reorganizing the repo so the motor-task work, the planned movie-watching extension (`docs/movie-watching-dataset-plan.md`), and the planned structural-connectivity extension (`docs/dti-sc-pipeline-plan.md`) each get their own space, instead of everything living in one flat, motor-assuming layout. Written up before executing anything invasive — the migration cost is real and worth seeing in full before committing to it.

## 1. Why this needs a real plan, not just "move the files"

Sizing the actual blast radius before proposing a structure, not after:

- **4,551 lines across `src/neurolens/`**, some genuinely dataset-agnostic (`engine.py`, `evaluation.py`, `model_builder.py`'s architecture classes, `concepts.py`'s CAV/TCAV mechanism, `retrieval.py`, `verification_v2.py`), some motor-specific (`data_setup.py`'s window/label construction, `contrastive.py`'s 6 fixed condition prototypes, `case3.py`'s HRF-specific alignment).
- **18 notebooks**, **35 result files**, **16 files with a hardcoded `PROCESSED_ROOT`** pointing at `data/processed/hcp_ya_s1200/runs` specifically.
- All of it currently working, cross-referenced by dozens of doc citations (`docs/*.md` linking to specific `results/*.json` and `notebooks/*.ipynb` paths).

Moving this wholesale means updating every one of those path references correctly, in one pass, with no partial-migration state where some code points at the old location and some at the new one. That's the real cost — not the `git mv` itself.

## 2. Proposed structure

```
data/
  motor/processed/hcp_ya_s1200/runs/    <- current data/processed/hcp_ya_s1200/runs/, moved
  motor/papers/                           <- current data/papers/*.pdf, moved (motor-cortex-organization corpus)
  movie/raw/, movie/processed/, movie/papers/   <- new, empty until docs/movie/movie-watching-dataset-plan.md executes (papers/ = naturalistic-viewing literature corpus, §5.3 there)
  structural/raw/, structural/processed/  <- new, DTI downloads + SC matrices (docs/structural/dti-sc-pipeline-plan.md)

models/
  motor/{case1,case2,case3}_bootstrap/, minilm_domain_finetuned/, rag_synthesis_lora_adapter/   <- moved
  movie/                                  <- new, empty

results/
  motor/*.json                            <- current 35 files, moved
  movie/                                  <- new, empty

notebooks/
  motor/01-17...ipynb                     <- current 18 notebooks, moved
  movie/                                  <- new, empty

scripts/
  motor/run_v2_*.py, run_baseline_mlp_sweep.py   <- moved
  movie/                                  <- new, empty
  structural/                             <- new, DTI/SC preprocessing scripts

docs/
  motor/  project-summary.md, case1-summary-report.md, end-to-end-report.md,
          interview-prep-*.md, v2/, case2-3-design-plan.md, etc.   <- moved
  movie/  movie-watching-dataset-plan.md   <- moved (already written, at top level for now)
  structural/  dti-sc-pipeline-plan.md     <- moved (already written, at top level for now)
  (top-level) hcp-dataset-extension-options.md, repo-restructuring-plan.md, README.md, NeuroLens-RAG-Report.md
              -- genuinely cross-cutting, apply to no single dataset

src/neurolens/
  (unchanged: engine.py, evaluation.py, model_builder.py, concepts.py, retrieval.py, verification_v2.py, interpretability.py)
  motor/  data_setup.py, contrastive.py, case3.py, pipeline.py, baseline_mlp.py   <- moved, motor-specific
  movie/  (new modules, once movie-watching-dataset-plan.md §4-5 gets built)
```

**Principle drawn from actually reading the code, not asserted**: `src/neurolens/` splits cleanly along a real seam — modules built around the *mechanism* (training loop, CAV/TCAV math, retrieval, the v2 verification loop) stay shared; modules built around *this specific dataset's* data shape (300-ROI motor windows, 6 fixed condition prototypes, HRF-vector alignment) move under `motor/`. This isn't a guess — it's the same seam `docs/movie-watching-dataset-plan.md` §4 already had to name explicitly when it flagged that Case 3's HRF alignment target has no movie-watching equivalent.

## 3. Sequencing — two genuinely different risk levels, not one migration

**Low-risk, do now, nothing to break:** create the `movie/` and `structural/` sides of every top-level directory as empty (or near-empty) skeletons. Nothing currently depends on these paths, so there's no reference-updating to get wrong. Move the two plan docs (`movie-watching-dataset-plan.md`, `dti-sc-pipeline-plan.md`) into `docs/movie/` and `docs/structural/` respectively at the same time — they're new enough that nothing else cites them yet either.

**High-risk, needs a decision before executing:** moving the *existing* motor content (`data/processed/`, all 18 notebooks, all 35 results files, `models/`, the motor-specific `src/neurolens/` modules) into `motor/` subfolders. Every one of the 16 hardcoded `PROCESSED_ROOT` references, every doc's relative links to `results/*.json` and `notebooks/*.ipynb`, and every script's `sys.path` assumption needs updating in the same pass — a real, mechanical, error-prone piece of work, not a `git mv -r`.

**Recommendation: do the low-risk half now, defer the high-risk half.** There's no movie-watching or structural data yet to justify disrupting a large, currently-working motor pipeline today. The parallel structure pays off once real movie-watching/SC work is actually running side-by-side with motor work — not before. Revisit once `docs/movie-watching-dataset-plan.md` §3's subject-overlap question is resolved and real preprocessing starts, at which point the migration is justified by actual new content, not anticipated future content.

## 4. What this document does not do

It does not move any existing file. §3's low-risk skeleton creation is a separate, explicit follow-up action, not silently bundled into writing this plan.
