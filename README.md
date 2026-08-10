# NeuroLens-RAG

NeuroLens-RAG decodes movement conditions from human fMRI (HCP Young Adult, MOTOR task, 100 subjects) using two complementary representation-learning objectives, then validates what each model actually learned using mechanistic interpretability and a retrieval-augmented literature-grounding pipeline — rather than trusting classification accuracy alone.

For the full write-up (abstract, methods, results, discussion, figures), see **[docs/NeuroLens-RAG-Report.md](docs/NeuroLens-RAG-Report.md)**. For a running engineering log of every experiment and its numbers, see [docs/project-summary.md](docs/project-summary.md).

## What's here

- **Case 1 — supervised multi-task decoding**: a GRU/Transformer encoder predicts movement class and hemodynamic response jointly.
- **Case 2 — contrastive brain–text representation learning**: a CLIP-style objective aligns a brain encoder and a text encoder of the six condition descriptions in a shared embedding space, with no classification head.
- **Mechanistic interpretability**: four attribution methods (Saliency, Integrated Gradients, exact Shapley, LIME) identifying which resting-state network drives a decode, plus Concept Activation Vectors (CAV/TCAV) testing whether a model's decision is causally sensitive to a human-specified concept.
- **Literature-grounded verification**: a RAG system retrieves relevant neuroscience literature and converts extracted claims into concept tests against each model's own representation, with the evidentiary verdict computed deterministically rather than left to an LLM's free judgment.
- **Population-level statistics**: every comparative claim is backed by repeated subject-level resampling and paired non-parametric tests, following Misra & Pessoa (2025, *eLife*).

## Model architectures

**Case 1 — supervised multi-task decoder.** A shared GRU or Transformer backbone (compared directly, see Results) pools a 32-TR window into a 128-dim representation, feeding a classification head and an auxiliary HRF-regression head.

![Case 1 architecture](docs/figures/fig_case1_architecture.png)

**Case 2 — contrastive brain–text representation learning.** The same backbone (heads removed) projects into a 64-dim shared space; a frozen sentence-embedding model plus a small trainable projection does the same for the six condition descriptions. Training aligns each brain window with its condition's text prototype via a temperature-scaled cosine-similarity cross-entropy loss.

![Case 2 architecture](docs/figures/fig_case2_architecture.png)

## Verification architecture

**Concept verification mechanism (CAV/TCAV).** A concept direction can be derived from labeled brain examples (Case 1) or, uniquely for the contrastive model, from arithmetic on text-prototype embeddings alone, with no brain examples needed. Both routes land in the same representational space and are tested identically via a directional derivative.

![CAV/TCAV verification mechanism](docs/figures/fig_cav_verification_mechanism.png)

**Literature-grounded verification loop.** Retrieved literature is converted into a stance and a testable concept phrase in one LLM call; the concept is tested against the model via CAV/TCAV; and the agree/disagree/unclear verdict is computed deterministically from the stance and the TCAV score. The LLM's only free-form output is the final narration of a verdict it did not decide.

![Verification loop architecture](docs/figures/fig_verification_loop_architecture.png)

A worked example tracing one real decoded window through every stage of both diagrams above is in the full report ([Figure 1b](docs/NeuroLens-RAG-Report.md)).

## Repository layout

```
src/neurolens/       persistent logic (data, models, training, interpretability, retrieval, pipeline)
notebooks/           numbered, executed drivers with real outputs (03-16)
docs/                reports and design docs, including the full project report and figures
results/             saved experiment outputs (metrics, examples, figures source data)
models/              trained checkpoints (gitignored)
data/                raw and processed data, and the literature corpus (gitignored)
```

## Environment

```bash
conda env create -f environment.yml
conda activate neurolens
```

PyTorch installation may vary by operating system and accelerator; verify with `notebooks/00_environment_check.ipynb` after creating the environment.
