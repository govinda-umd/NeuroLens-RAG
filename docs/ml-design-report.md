# NeuroLens-RAG — ML Design Report: Brain Decoding

> **Status: v1 implemented and trained.** All four planned experiments (GRU / Transformer × classification-only / +HRF) have been implemented in `src/neurolens/` and run end-to-end via `notebooks/03_dataset_dataloaders.ipynb` → `04_models.ipynb` → `05_train_eval_compare.ipynb`. This is a first pass (5 subjects, 5 epochs) — see §9 for results and caveats, and §11 for what's still open. Expands on [§8 Learning Setup](project-handoff-summary.md#8-learning-setup) of the project handoff summary.

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

- `lambda_hrf = 0.1` was used as-is for v1 (untuned). §9 shows evidence it may be too small — the HRF head's per-condition R² is mostly negative despite positive correlation, suggesting the regression loss isn't dominant enough to fit amplitude/scale well.
- **Class weights** computed from training subjects only: `weight[c] = num_training_samples / (num_classes × samples_in_class_c)`.

**v1 choices (untuned, first pass):**
- Optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`
- Schedule: `CosineAnnealingLR` over the 5 training epochs
- Batch size: 64 (worked fine on 16GB unified memory / MPS; not stress-tested at larger sizes)
- Epochs: fixed at 5 (chosen for fast iteration, not early-stopped) — training curves in `05_train_eval_compare.ipynb` suggest val macro F1 was still trending upward for most runs at epoch 5
- Weight decay: 1e-4; no gradient clipping

**Still open**: a real LR/batch-size/optimizer sweep, early stopping instead of a fixed epoch count, and `lambda_hrf` tuning (see §11).

## 5. Training Protocol

- **Split**: subject-level (never window-level) — train = `100307, 100408, 101006`, val = `101107`, test = `101309`.
- **Checkpointing**: best checkpoint by validation macro F1, saved to `models/<experiment_name>/best.pt` (gitignored — local artifacts only).
- **Experiment tracking**: none wired up yet beyond the consolidated `results/motor_v1_results.json` (full per-epoch history + final metrics for all 4 experiments). A lightweight tracker (CSV/JSON run log, or local MLflow/W&B-offline) is still worth adding once the number of experiments grows beyond what fits in one notebook run.

## 6. Evaluation Protocol

**Primary (classification) metrics**: loss, accuracy, balanced accuracy, macro F1 (**model-selection metric**), per-class precision/recall, confusion matrix.

**Auxiliary (HRF regression) metrics**: MSE, MAE, per-condition correlation, possibly R².

Report both per-experiment and side-by-side across the 4-experiment ladder so the effect of (a) architecture and (b) the auxiliary HRF objective can each be isolated.

## 7. Experimental Ladder

| # | Encoder | Objective | Status |
|---|---|---|---|
| 1 | GRU | classification-only | **done** |
| 2 | GRU | classification + HRF | **done** |
| 3 | Transformer | classification-only | **done** |
| 4 | Transformer | classification + HRF | **done** |

Run in [`05_train_eval_compare.ipynb`](../notebooks/05_train_eval_compare.ipynb), 5 epochs each, AdamW + cosine LR annealing, `lambda_hrf = 0.1`, seed 42, batch size 64. Full per-epoch history: [`results/motor_v1_results.json`](../results/motor_v1_results.json).

## 8. Current Implementation Status

| Component | Status |
|---|---|
| `01_pdf_ingestion.ipynb` | done |
| `02_data.ipynb` (HCP → ROI bundles) | done — processed MOTOR runs exist under `data/processed/hcp_ya_s1200/runs/` |
| `03_dataset_dataloaders.ipynb` | done — `src/neurolens/data_setup.py` |
| `04_models.ipynb` | done — `src/neurolens/model_builder.py` |
| `05_train_eval_compare.ipynb` | done — `src/neurolens/engine.py`, `src/neurolens/evaluation.py` |
| Trained models / results | **done for v1** — checkpoints under `models/` (local, gitignored), metrics in `results/motor_v1_results.json` |

## 9. Results

First pass, 5 subjects (3 train / 1 val / 1 test), 5 epochs. **Caveat: a single test subject (101309) means these numbers carry high variance — not yet a reliable estimate of generalization.** Treat as a first checkpoint to compare future changes against, not a final number.

| Experiment | Params | Best val macro F1 | Test macro F1 | Test balanced acc | Test HRF MSE | Test HRF MAE |
|---|---|---|---|---|---|---|
| 1 — GRU, cls-only | 165,894 | 0.682 | 0.558 | 0.572 | n/a | n/a |
| 2 — GRU, cls+HRF | 166,539 | 0.681 | 0.586 | 0.596 | 0.118 | 0.249 |
| 3 — Transformer, cls-only | 304,262 | 0.709 | 0.621 | 0.629 | n/a | n/a |
| 4 — Transformer, cls+HRF | 304,907 | 0.685 | 0.632 | 0.638 | 0.150 | 0.303 |

**Observations:**

- **Transformer > GRU** on this split for both objectives (test macro F1 0.62–0.63 vs. 0.56–0.59), consistent with the extra capacity (~305K vs ~166K params) and full-window self-attention vs. a single recurrent hidden state.
- **The auxiliary HRF objective modestly helped classification** for both encoders (GRU: 0.558→0.586 test macro F1; Transformer: 0.621→0.632), supporting the hypothesis in §1 that the auxiliary task regularizes the shared representation — though with n=1 test subject this could also be noise, not a real effect.
- **HRF regression is only partially working**: per-condition correlations between predicted and true HRF regressors are positive and moderate (GRU: 0.32–0.44; Transformer: 0.17–0.51 across conditions), but **R² is mostly negative** (e.g. Transformer: -0.07 to -0.53). This means the model tracks the general shape/timing but not the amplitude/scale — predictions correlate with the target but have subpar absolute fit. Plausible causes worth investigating before trusting the HRF head: `lambda_hrf = 0.1` may be too small to actually fit the regression well (classification loss dominates), only 5 epochs may be too few for the regression head to converge, or the HRF targets' scale/normalization may need revisiting. Not investigated further in this pass, per the "5 epochs to run fast" scope for v1.
- **Per-class F1 is uneven** (e.g. GRU exp1: tongue F1=0.71 vs. right_foot F1=0.39), plausibly reflecting real motor-cortex signal differences (e.g. tongue movements activate a larger, less spatially overlapping cortical region than foot movements) but also plausibly an artifact of the tiny 5-subject sample — worth re-examining once more subjects are added.
- Best-validation-epoch selection landed at epoch 5/5 for GRU experiments and epoch 4/5 for Transformer experiments — training curves in the notebook show val macro F1 still trending upward at 5 epochs for most runs, suggesting more epochs (not just LR scheduling) may still help; not explored in this pass to keep training fast.

**Not yet done** (explicitly out of scope for this first pass, candidates for the next iteration): more than 5 subjects, more than 5 epochs, `lambda_hrf` tuning, hyperparameter sweep (LR/batch size/optimizer), alternative Transformer pooling, and any of the tokenization/architecture alternatives in [literature-notes-tokenization.md](literature-notes-tokenization.md).

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
6. **Optimizer/LR/batch-size/epoch-count sweep** — v1 used untuned defaults (AdamW, lr=1e-3, batch=64, 5 fixed epochs); val macro F1 was still rising at epoch 5 for most runs, so more epochs and/or a real sweep is the most likely easy win before anything more elaborate.
7. **Why is HRF R² negative despite positive correlation?** — v1's multi-task experiments show the HRF head tracks direction/timing (correlation 0.17–0.51) but fits amplitude poorly (R² mostly negative). Worth checking whether this is `lambda_hrf` being too small, too few epochs, or a target-scaling issue, before reading too much into the auxiliary-task hypothesis from §1.
8. **More subjects** — v1's 1-subject test set makes every number here high-variance; the ladder should be re-run on more subjects before drawing real conclusions about GRU vs. Transformer or classification-only vs. multi-task.
9. **Causal attention masking inside the Transformer window** — v1's `TransformerDecoder` (`src/neurolens/model_builder.py`) applies no attention mask, so within a 32-token window attention is fully bidirectional: position 5 can attend to position 31 and vice versa. "Causal" in v1 only holds at the sample-construction level (a window never includes volumes after its own target time `t`), not inside the encoder's self-attention. Worth investigating whether adding a true causal (triangular) attention mask — so position `i` can only attend to positions `<= i` — changes performance or better supports an eventual streaming/real-time use case, versus the current full-window bidirectional attention.

## References

- [project-handoff-summary.md](project-handoff-summary.md) — full project context
- [literature-notes-tokenization.md](literature-notes-tokenization.md) — Eva Dyer lab tokenization survey
