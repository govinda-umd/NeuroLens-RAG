# HCP Movie-Watching dataset — reserved, not yet populated

Placeholder for a second HCP dataset (naturalistic movie-watching fMRI), planned to test whether the NeuroLens-RAG paradigm — three representation-learning cases, CAV/TCAV concept probing, and the claim-first RAG-CAV verification loop — generalizes beyond the MOTOR task's clean, discrete-block design to a more continuous, mixed-condition task.

## Why this matters for the existing findings

Two results from the MOTOR task (`docs/interview-prep-neurolens-rag.md`) are specifically conditioned on that task's structure — long, temporally well-separated condition blocks:

- All 6 representations (3 paradigms × 2 architectures) converge to near-ceiling TCAV on every tested concept.
- Concept-attribution is unanimous across all 6 representations for every concept (`results/v2_attribution_consistency.json`).

The working hypothesis is that both results follow from the MOTOR task's clean block structure making the real signal easy for any reasonable representation-learning approach to find — not from something special about these three training objectives. Movie-watching data (continuous, naturalistic, temporally overlapping cognitive demands) is a genuine stress test of that hypothesis: if representation-learning-objective convergence weakens on messier data, that confirms the block-structure explanation; if it holds up, the finding is more general than the MOTOR task alone can show.

## Expected layout once populated

Mirrors the existing MOTOR pipeline's convention:

- `data/raw/hcp_movie_watching/` (this directory) — raw downloaded HCP movie-watching fMRI + associated timing/annotation files.
- `data/processed/hcp_ya_s1200_movie/runs/sub-<id>/tfMRI_MOVIE*/` — preprocessed ROI time series, following the same Schaefer-300 parcellation and preprocessing steps as `data/processed/hcp_ya_s1200/runs/` (the existing MOTOR data), so `src/neurolens/data_setup.py`'s loaders need a new `processed_root` and task name, not new parsing logic.

Nothing under `data/raw/` or `data/processed/` is tracked by git (see `.gitignore`) — this README is the one exception, so the plan survives even though the data itself doesn't.
