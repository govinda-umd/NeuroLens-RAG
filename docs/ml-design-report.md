# NeuroLens-RAG — ML Design Report: Brain Decoding

> **Status: design specification, not a results report.** No model has been trained yet. The data pipeline (`02_data.ipynb`) is complete and processed HCP MOTOR-task bundles exist under `data/processed/hcp_ya_s1200/runs/`. `03_dataset_dataloaders.ipynb` (Dataset/DataLoader construction) has not been implemented yet, and none of the four planned experiments (GRU / Transformer × classification-only / +HRF) have been run. This document specifies the design so implementation and results can be filled in incrementally. Expands on [§8 Learning Setup](project-handoff-summary.md#8-learning-setup) of the project handoff summary.

## 1. Problem Statement

**Case 1 — Brain decoding.** Given a causal window of ROI activity, predict:

- **Primary task**: `X → y` — discrete movement condition classification (6 classes for MOTOR: baseline + 5 movements)
- **Auxiliary task**: `X → y_hrf` — continuous canonical-HRF-convolved regression target (5 channels for MOTOR)

Framed as **multi-task learning** with a shared temporal encoder and two heads:

```
                           ┌── classification head → y      (primary)
X window → shared encoder ─┤
                           └── regression head → y_hrf       (auxiliary)
```

`y_hrf` must never be fed as a model **input** when predicting `y` — it is derived from the event labels and would leak the target. It only appears as a training signal (auxiliary loss), never as a feature.

**Why auxiliary HRF regression at all?** The hypothesis is that forcing the shared encoder to also reconstruct graded, continuous temporal structure (onset/rise/peak/decay, condition overlap) regularizes the representation and prevents it from collapsing onto features that are only useful for the discrete label — potentially improving generalization to held-out subjects.

## 2. Input / Output Specification

| Quantity | Shape | Notes |
|---|---|---|
| Raw run signal `X` | `[time, 300]` | motion-cleaned Schaefer-300 ROI time series |
| Window sample `x` | `[32, 300]` | window length 32, stride 2, causal |
| Batch `x` | `[B, 32, 300]` | |
| Batch `y` | `[B]` | int class label, target = last volume in window |
| Batch `y_hrf` | `[B, 5]` | MOTOR only; general case `[B, n_conditions]` |

**"Tokenization" for the Transformer path**: one token = one fMRI volume (one 300-dim ROI vector at one TR). No NLP-style vocabulary or subword tokenizer is used.

```
[32, 300] --Linear(300 → d_model)--> [32, d_model] --+ positional encoding--> Transformer encoder
```

This is the simplest possible tokenization scheme (whole-brain vector per timestep, evenly spaced, no learned discretization). See [literature-notes-tokenization.md](literature-notes-tokenization.md) for alternative schemes worth evaluating later (patch tokenization, spatial position embeddings from atlas coordinates, event/spike-style tokenization, subject-specific calibration).

The GRU path consumes `[B, 32, 300]` directly with no projection step.

## 3. Model Architectures (planned, not yet implemented)

### GRU (Experiments 1 & 2)

```
input_size = 300
hidden_size = 128
num_layers = 1

X [B, 32, 300] → GRU → final hidden state [B, 128]
                          ├── classifier: Linear(128 → 6)
                          └── HRF regressor: Linear(128 → 5)
```

Rough parameter count: GRU(300→128, 1 layer) ≈ 3 × 128 × (300 + 128 + 1) ≈ 165K params — small enough to train many epochs quickly even on CPU/MPS with 16GB unified memory.

### Transformer (Experiments 3 & 4)

```
input projection: Linear(300 → 128)
d_model = 128, n_heads = 4, n_layers = 2, feedforward = 256, dropout = 0.1

X [B, 32, 300] → input projection → + positional encoding → Transformer encoder
                                          → final temporal token [B, 128]
                                                ├── classifier: Linear(128 → 6)
                                                └── HRF regressor: Linear(128 → 5)
```

Uses the final sequence token as the pooled representation (no `[CLS]` token in v1). Worth revisiting — mean-pooling or attention-pooling over all 32 tokens is a cheap ablation once the baseline works.

## 4. Loss and Optimization Design

```python
classification_loss = cross_entropy(pred_logits, true_y, weight=class_weights)
hrf_loss = mse_loss(pred_hrf, true_y_hrf)
total_loss = classification_loss + lambda_hrf * hrf_loss   # lambda_hrf = 0.1 initial
```

- `lambda_hrf` is a starting guess — must be re-tuned once the two losses' numerical magnitudes are measured empirically (they are not naturally on the same scale: cross-entropy over 6 classes vs. MSE over normalized continuous targets).
- **Class weights** computed from training subjects only: `weight[c] = num_training_samples / (num_classes × samples_in_class_c)`.

**Not yet decided (open):**
- Optimizer (AdamW is the default assumption, unconfirmed)
- Learning rate / schedule
- Batch size (bounded by 16GB unified memory + MPS backend behavior, not yet profiled)
- Number of epochs / early-stopping criterion (proposed: early stop on validation macro F1 plateau)
- Weight decay, gradient clipping

## 5. Training Protocol (planned)

- **Split**: subject-level (never window-level) — 3 train / 1 val / 1 test subjects initially, to avoid near-duplicate overlapping windows leaking across splits.
- **Checkpointing**: save best checkpoint by validation macro F1; convention TBD (likely `models/<experiment_name>/best.pt` + a metrics JSON, consistent with the modular PyTorch project structure already used elsewhere in the repo).
- **Experiment tracking**: none wired up yet. Given portfolio-quality documentation goals, a lightweight tracker (e.g. a plain CSV/JSON run log, or a local MLflow/W&B-offline instance) should be decided before Experiment 1 runs, not after.

## 6. Evaluation Protocol

**Primary (classification) metrics**: loss, accuracy, balanced accuracy, macro F1 (**model-selection metric**), per-class precision/recall, confusion matrix.

**Auxiliary (HRF regression) metrics**: MSE, MAE, per-condition correlation, possibly R².

Report both per-experiment and side-by-side across the 4-experiment ladder so the effect of (a) architecture and (b) the auxiliary HRF objective can each be isolated.

## 7. Experimental Ladder (planned runs)

| # | Encoder | Objective | Status |
|---|---|---|---|
| 1 | GRU | classification-only | not started |
| 2 | GRU | classification + HRF | not started |
| 3 | Transformer | classification-only | not started |
| 4 | Transformer | classification + HRF | not started |

## 8. Current Implementation Status

| Component | Status |
|---|---|
| `01_pdf_ingestion.ipynb` | done |
| `02_data.ipynb` (HCP → ROI bundles) | done — processed MOTOR runs exist under `data/processed/hcp_ya_s1200/runs/` |
| `03_dataset_dataloaders.ipynb` | not started (next notebook) |
| `04_gru_baseline.ipynb` | not started |
| `05_transformer_baseline.ipynb` | not started |
| `src/` training/eval loop modules | not started |
| Any trained model / results | **none yet** |

## 9. Results

_No experiments have been run yet. This section is a placeholder to fill in once Experiments 1–4 are trained._

| Experiment | Val macro F1 | Test macro F1 | Val balanced acc | HRF val MSE | Notes |
|---|---|---|---|---|---|
| 1 — GRU, cls-only | — | — | — | n/a | |
| 2 — GRU, cls+HRF | — | — | — | — | |
| 3 — Transformer, cls-only | — | — | — | n/a | |
| 4 — Transformer, cls+HRF | — | — | — | — | |

## 10. Hardware/Compute Considerations (Apple M3, 16GB unified memory)

- Model sizes were deliberately kept small (hidden/`d_model` 64–128, 1–4 layers) so that both training and inference for the temporal models are cheap relative to the ROI-extraction preprocessing step, which is the actual bottleneck.
- Apple MPS backend should be used where supported; not yet validated for the exact GRU/Transformer configs above (some ops, e.g. certain attention masks, have historically had partial MPS support — worth a smoke test before Experiment 3).
- Batch size and dataloader worker count need empirical tuning against the 16GB ceiling — not yet profiled.
- This machine is also expected to eventually host **local LLM inference for the RAG interpretation step** (see open question below) — memory budget across the training pipeline and any local LLM needs to be planned jointly rather than assuming both run at full size simultaneously.

## 11. Open Design Questions (for discussion)

1. **Tokenization strategy** — is "one volume = one token" (current plan) the right scheme, or should tokens be patches of multiple ROIs/timepoints, use learned/spatial position embeddings (e.g. atlas coordinates instead of arbitrary sequence index), or move toward an event-style tokenization inspired by Eva Dyer's lab's spike-tokenization work? See [literature-notes-tokenization.md](literature-notes-tokenization.md).
2. **Subject-specific calibration** — beyond subject-level train/val/test splitting, should the model include subject-specific embeddings/calibration heads to explicitly account for inter-subject variability (borrowing from cross-subject intracranial-recording work)?
3. **`lambda_hrf` tuning strategy** — grid search vs. uncertainty-weighted multi-task loss (e.g. Kendall et al. homoscedastic uncertainty weighting) instead of a fixed scalar.
4. **Pooling strategy for the Transformer** — final-token vs. mean-pooling vs. attention-pooling vs. a `[CLS]` token.
5. **Local LLM/RAG interface** — how does a trained brain/task representation eventually connect to a RAG pipeline running entirely on this iMac? Candidate directions: (a) generate a structured text description of decoded state + active networks and feed it as a retrieval query against the existing `01_pdf_ingestion` paper index; (b) project `z_brain` into the embedding space of a local small LLM via a learned adapter; (c) run a quantized local LLM (e.g. via Ollama, llama.cpp, or Apple MLX) purely for the generation/interpretation step, keeping retrieval separate and classical (dense + optional BM25/rerank, per §1 of the handoff summary). Needs a hardware/memory budget decision given the 16GB ceiling shared with model training.
6. **Optimizer/LR/batch-size sweep** — none of these have been chosen yet; needs a short calibration pass before Experiment 1.

## References

- [project-handoff-summary.md](project-handoff-summary.md) — full project context
- [literature-notes-tokenization.md](literature-notes-tokenization.md) — Eva Dyer lab tokenization survey
