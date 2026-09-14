# NeuroLens-RAG

[![tests](https://github.com/govinda-umd/NeuroLens-RAG/actions/workflows/tests.yml/badge.svg)](https://github.com/govinda-umd/NeuroLens-RAG/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Does a brain-decoding model's accuracy come from something real, or a shortcut?** NeuroLens-RAG trains the same fMRI decoding task three structurally different ways (supervised, supervised-contrastive, self-supervised), tests whether each resulting representation depends on concepts a human would recognize — which body part moved, which side, movement vs. rest — using Concept Activation Vectors, validates every claim across 30 paired subject-level bootstrap resamples, and cross-checks the result against retrieved neuroscience literature through a verification pipeline that is itself held to the same evidentiary standard. Built and validated on HCP Young Adult MOTOR-task fMRI (100 subjects).

![System overview](docs/figures/fig_system_overview.svg)

## Headline results

1. **Architecture ranking flips by training paradigm, not a fixed winner.** Transformer significantly beats GRU under both label-driven objectives (supervised, contrastive; paired Wilcoxon p<0.0001), but GRU edges ahead under the one objective that never sees a label (self-supervised; p=0.0004) — across 30 matched resamples per paradigm.
2. **A flat MLP statistically ties a recurrent model under two of three paradigms — but not the third.** FlattenMLP (no learned temporal structure at all) ties GRU under supervised and supervised-contrastive training (p=0.73, p=0.887), yet *loses* significantly under self-supervised training (p<0.0001) — recurrence appears to matter specifically when no label drives training directly, a result only visible once the baseline check was run across all three paradigms.
3. **All three paradigms converge to near-ceiling concept-interpretability — but only once a diagnosed methodological artifact was fixed.** A convenient, zero-fit CAV-derivation shortcut looked like a real representational weakness in one paradigm (TCAV ≈0.33 on laterality, even sign-inverted for one backbone); a controlled experiment showed it was purely a limitation of that derivation method, not the representation (0.997–0.999 probe accuracy, 0.92–1.00 TCAV once measured the same way as the other two paradigms).
4. **Reranking, not embedding fine-tuning, drives most of the retrieval quality.** A component-level ablation ladder (BM25 → frozen embeddings → domain-adapted embeddings → ± cross-encoder reranking → oracle) shows frozen embeddings + reranker already matches the fully fine-tuned pipeline — domain adaptation alone buys a modest +0.04 recall@1 next to reranking's +0.28.
5. **A measured LLM sycophancy failure, fixed and formally audited.** Asked to freely judge whether retrieved literature supports a model's behavior, the verification LLM defaulted to agreement in 10 of 12 real cases regardless of the actual evidence. Fixed by computing the verdict deterministically from (stance, concept-sensitivity score) — the LLM only narrates a decision it doesn't make — and validated with a formal precision/recall/confusion-matrix audit against a labeled example set, which surfaces the real tradeoff the fix makes (catches 4/4 of the hardest "topically related but non-specific" failures, at a quantified cost to raw recall on clear cases).

Full detail, every number, every experiment that produced it: **[docs/end-to-end-report.md](docs/end-to-end-report.md)**.

## Connection to connectomics and scalable scientific computing

The interpretability layer operates at the level of the 7 Yeo canonical resting-state networks over a Schaefer-300 cortical parcellation — the same network-level anatomical grouping used throughout functional connectomics — and the concept-attribution mechanism (backpropagating a concept's alignment score to the raw ROI input) is designed to generalize directly to a structural-connectome graph input (an SC-GNN front-end, with edge-level attribution via GNNExplainer/PGExplainer, is a scoped, not-yet-built extension). On the computing side: every result in this repo is backed by an automated, checkpointed, resumable bootstrap pipeline (30 independently-retrained models per comparison, one shared subject-split protocol reused across every paradigm so comparisons stay paired rather than independently noisy), the same discipline used at larger scale elsewhere in this author's work (10,000+ automated model runs on Slurm/HPC, described in the author's CV) — this repo is a smaller, fully public demonstration of that same automated-pipeline, validate-don't-assume approach.

## Contributions and ownership

Sole author and developer: architecture, training pipelines, interpretability mechanism, retrieval/verification system, statistical validation framework, and all documentation in this repository. Built as an independent research project, not a lab-assigned or team deliverable — every design decision, bug found, and methodological correction documented in `docs/end-to-end-report.md` reflects work done by one person, including the self-correction record (derivation-method artifacts found and fixed, LLM failure modes measured and designed around) that the report deliberately keeps visible rather than smoothing over.

## Installation and a minimal reproducible check

```bash
conda env create -f environment.yml
conda activate neurolens
```

PyTorch installation may vary by operating system and accelerator; verify with `notebooks/00_environment_check.ipynb` after creating the environment.

**Verify the environment and core logic** (no data download required — runs against synthetic data, ~5 seconds):

```bash
pytest tests/ -v
```

**Reproduce a headline result** (requires the processed HCP data — see Data access below): `notebooks/05_train_eval_compare.ipynb` trains and compares all four Case-1 architectures (GRU, Transformer, FlattenMLP, MeanPoolMLP) on one subject split in under a minute; `notebooks/12_population_level_evaluation.ipynb` and `notebooks/13_architecture_comparison_bootstrap.ipynb` run the full 30-resample population-level version behind headline result #1 above.

## What's here

- **Case 1 — supervised multi-task decoding**: a GRU/Transformer encoder predicts movement class and hemodynamic response jointly.
- **Case 2 — supervised-contrastive representation learning**: a symmetric multi-positive contrastive objective aligns a brain encoder and a text encoder of the six condition descriptions in a shared embedding space, with no classification head.
- **Case 3 — self-supervised representation learning**: a brain encoder is aligned to its own window's hemodynamic-response vector via a symmetric InfoNCE loss — no class label ever enters training; a linear probe fit after the fact provides the readout needed for accuracy and interpretability testing.
- **Baseline and control battery**: flat and mean-pooled MLP baselines (does learned temporal structure matter at all?), temporal perturbation controls (shuffle/reverse/circular-shift/mean-pool at test time), and ROI-network ablation controls (does a suspected confound survive input-level lesioning?) — every headline claim is checked against the simplest alternative explanation before being reported.
- **Mechanistic interpretability**: four attribution methods (Saliency, Integrated Gradients, exact Shapley, LIME) identifying which resting-state network drives a decode, plus Concept Activation Vectors (CAV/TCAV) testing whether a model's decision is causally sensitive to a human-specified concept, standardized across all three paradigms.
- **Literature-grounded verification (v1 and v2)**: a RAG system retrieves relevant neuroscience literature and converts extracted claims into concept tests against each model's own representation, with the evidentiary verdict computed deterministically rather than left to an LLM's free judgment — a two-bug-found-by-running-it design history documented in full.
- **Population-level statistics**: every comparative claim is backed by repeated subject-level resampling and paired non-parametric tests, following Misra & Pessoa (2025, *eLife*).

## Model architectures

**Case 1 — supervised multi-task decoder.** A shared GRU or Transformer backbone (compared directly, see Results) pools a 32-TR window into a 128-dim representation, feeding a classification head and an auxiliary HRF-regression head.

![Case 1 architecture](docs/figures/fig_case1_architecture.png)

**Case 2 — contrastive brain–text representation learning.** The same backbone (heads removed) projects into a 64-dim shared space; a frozen sentence-embedding model plus a small trainable projection does the same for the six condition descriptions. Training aligns each brain window with its condition's text prototype via a temperature-scaled cosine-similarity loss.

![Case 2 architecture](docs/figures/fig_case2_architecture.png)

## Verification architecture

**Concept verification mechanism (CAV/TCAV).** A concept direction is derived by fitting a linear probe on frozen pooled features (standardized across all three cases, including a post-hoc-fitted classifier head for Cases 2 and 3, which never train one directly). Tested identically across paradigms via a directional derivative.

![CAV/TCAV verification mechanism](docs/figures/fig_cav_verification_mechanism.png)

**Literature-grounded verification loop.** Retrieved literature is converted into a stance and a testable concept phrase; the concept is tested against the model via CAV/TCAV; the agree/disagree/unclear verdict is computed deterministically from the stance and the TCAV score. The LLM's only free-form output is the final narration of a verdict it did not decide.

![Verification loop architecture](docs/figures/fig_verification_loop_architecture.png)

A worked example tracing one real decoded window through every stage of both diagrams above is in [`docs/NeuroLens-RAG-Report.md`](docs/NeuroLens-RAG-Report.md) (Figure 1b) — a paper-style narrative write-up; see the note at its top for which numbers it as superseded by the current report.

## Repository layout

```
src/neurolens/       persistent logic (data, models, training, interpretability, retrieval, pipeline)
tests/               pytest suite: model shape contracts, CAV/TCAV mechanism, loss sanity checks, metrics
notebooks/           numbered, executed drivers with real outputs (03-16)
docs/                reports and design docs -- start with end-to-end-report.md for current results
results/             saved experiment outputs (metrics, examples, figures source data)
models/              trained checkpoints (gitignored)
data/                raw and processed data, and the literature corpus (gitignored)
```

## Tests, CI, license, and data access

- **Tests**: `pytest tests/` — 17 tests covering model interface contracts across all four architectures, the CAV/TCAV mechanism against synthetic ground truth, contrastive-loss alignment sensitivity, and classification metrics against hand-computed values. Runs in seconds, no data download needed.
- **CI**: every push/PR to `main` runs the full suite on GitHub Actions (badge above).
- **License**: [MIT](LICENSE).
- **Data access**: HCP Young Adult data requires a Data Use Agreement through [ConnectomeDB](https://db.humanconnectome.org/) (no direct redistribution here — `data/` is gitignored). The literature corpus (`data/papers/`) is built from openly-available published PDFs and preprints, also gitignored to keep the repository lean; `artifacts/paper_index/` (also gitignored) is rebuilt from it on demand via `retrieval.py::build_index`.

## Honest limitations

Stated in full in [`docs/end-to-end-report.md`](docs/end-to-end-report.md) §8 rather than summarized here — includes an unresolved verdict-design gap (Case 1's literature-verification loop still uses an older, unfixed design), the small size of the literature corpus and its direct effect on retrieval/grounding results, and an explicit note that the population-level interpretability convergence finding is conditioned on this task's clean block structure and not yet stress-tested on a messier one.
