# NeuroLens-RAG — Project Handoff Summary

High-level summary of the project scope, decisions, and roadmap, consolidated from planning discussions. Use this as an orientation doc before diving into the notebooks.

## 1. Project Goal

**NeuroLens-RAG** combines:

- Scientific-paper ingestion and retrieval (RAG)
- fMRI temporal modeling using GRUs and Transformers
- Eventually, brain-state interpretation connected to scientific literature retrieval

Long-term concept:

```
brain dynamics
+ experimental/task context
+ model attribution or active networks
        ↓
structured neurocognitive representation
        ↓
scientific literature retrieval
        ↓
RAG-based interpretation
```

Inspired by the paper *"Learning brain dynamics for decoding and predicting individual differences."*

## 2. Hardware and Environment

- Apple M3 iMac, 16 GB unified memory
- VS Code, Miniconda, Python 3.11, PyTorch, Apple MPS
- Conda environment: `neurolens`

Feasible because voxelwise NIfTI data is processed one run at a time; resulting ROI arrays are small. The expensive step is NIfTI-to-ROI extraction, not training the compact temporal models.

Reasonable model sizes:

- **GRU**: `input_size=300`, `hidden_size=64–128`, 1–2 layers
- **Transformer**: `d_model=128`, `n_heads=4`, `n_layers=2–4`, `feedforward=256–512`, window length 32–100

## 3. Notebook Roadmap

```
01_pdf_ingestion.ipynb            ✅ done
02_data.ipynb                     ✅ done (consolidates 02_hcp_data_acquisition,
                                      03_hcp_roi_timeseries_extraction,
                                      04_hcp_events_and_targets)
03_dataset_dataloaders.ipynb      ⏭ next
04_gru_baseline.ipynb
05_transformer_baseline.ipynb
...
```

## 4. Notebook 01 — PDF Ingestion & Retrieval (done)

Pipeline: PDF → page-level Markdown/text extraction → cleaning → overlapping chunks → embedding → query embedding → cosine-similarity retrieval → saved paper index.

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval: normalized query embedding · normalized chunk embeddings (dot product = cosine similarity)
- Chunking defaults: ~220 words, 50-word overlap
- Artifacts: `chunks.jsonl`, `embeddings.npy`, metadata

Future retrieval experiments: chunk size/overlap sweeps, section-aware (Markdown heading) chunking, alternate embedding models, BM25, hybrid dense+BM25, Reciprocal Rank Fusion, cross-encoder reranking, better benchmark queries/metrics.

> **Reranking note**: reusing the same embedding cosine similarity to "rerank" won't change chunk order. A true reranker jointly scores `(query, chunk)` — e.g., a MiniLM cross-encoder.

## 5. HCP Data Source & AWS Setup

- Source: **HCP Young Adult S1200**, AWS bucket `hcp-openaccess`, prefix `HCP_1200`
- Distinct from the newer HCP-YA 2025 release — **do not mix S1200 and 2025 processed data**
- AWS credentials live outside the repo (`~/.aws/credentials`, `~/.aws/config`) via `aws configure --profile hcp` — never committed or placed in notebooks
- boto3: `boto3.Session(profile_name="hcp", region_name="us-east-1")`

## 6. Notebook 02 — `02_data.ipynb` (done)

Produces compact processed modeling bundles while retaining no downloaded NIfTI files.

**Flow**: HCP S3 → inspect files → download one run to temp dir → validate NIfTI + movement regressors → extract motion-cleaned ROI time series → parse modeled EV files → construct HRF targets → save bundle → delete temp dir and raw source files.

Only one subject-task-run is processed at a time (bounded memory/disk).

**Output layout**:
```
data/processed/hcp_ya_s1200/runs/
└── sub-<subject_id>/
    └── tfMRI_<TASK>_<RUN>/
        ├── X.npy
        ├── y.npy
        ├── y_hrf.npy
        ├── valid_mask.npy
        ├── task_mask.npy
        ├── frame_times.npy
        ├── roi_labels.tsv
        ├── events_long.tsv
        ├── source_manifest.tsv
        └── metadata.json
```

Notebook verifies `retained_nifti_count == 0` and `raw_files_retained == false`. Older raw downloads may still exist under `data/raw/hcp_ya_s1200/` — not auto-deleted, remove manually after verifying processed bundles.

### ROI extraction choices

Source NIfTI is already minimally preprocessed; the notebook does **not** rerun motion correction, distortion correction, MNI registration, or spatial normalization.

Analysis-level processing: `Movement_Regressors.txt` regression, detrending, within-run ROI standardization, **Schaefer 2018 atlas** (300 cortical ROIs, 7-network solution, 2mm).

```python
NiftiLabelsMasker(
    labels_img=atlas,
    detrend=True,
    standardize="zscore_sample",
    standardize_confounds=True,
    low_pass=None,
    high_pass=None,
    t_r=tr,
    resampling_target="data",
)
```

Motion-regressed version preferred over no-confound-regression version. **Not yet implemented** (future ablations): volume scrubbing, frame censoring, spatial smoothing, subcortical ROI extraction, aggressive temporal filtering, derivative motion-regressor concatenation.

### Event mappings

Explicit mappings prevent cue/instruction files from being treated as classes:

- **MOTOR**: `lh→left_hand`, `rh→right_hand`, `lf→left_foot`, `rf→right_foot`, `t→tongue`
- **EMOTION**: `fear→faces`, `neut→shapes`
- **WM**: 0-back/2-back × body/faces/places/tools (8 conditions)

Initial modeling uses **MOTOR only**.

## 7. Saved Array Semantics

Per subject-task-run:

```
X.shape      == [time, 300]     # motion-cleaned ROI signals per volume
y.shape      == [time]          # hard class label per volume
y_hrf.shape  == [time, n_conditions]   # HRF-convolved event regressors
```

For MOTOR: `y_hrf.shape == [time, 5]`, `y[t] ∈ {0..5}` (0 = baseline, 1–5 = movement conditions).

`y_hrf[t, condition]` is a continuous canonical-HRF-convolved regressor capturing onset, rise, peak, decay, and cross-condition overlap.

## 8. Learning Setup

**Case 1 — brain decoding**: `X → y` and `X → y_hrf`.

⚠️ `y_hrf` must **never** be an input to the classifier predicting `y` — it's derived from event labels and would leak the target.

**Multi-task architecture**:
```
                           ┌── classification head → y
X window → shared encoder ─┤
                           └── regression head → y_hrf
```

```python
total_loss = cross_entropy(pred_logits, true_y) + lambda_hrf * mse_loss(pred_hrf, true_y_hrf)
```
Suggested initial `lambda_hrf = 0.1` (tune after comparing loss magnitudes). Classification is the primary task; HRF regression is auxiliary, encouraging the shared representation to retain graded temporal structure.

## 9. fMRI "Tokenization"

No NLP tokenizer needed — one token = one fMRI volume (a 300-dim ROI vector).

```
X_window.shape == [window_length, 300]
→ nn.Linear(300, d_model)
→ e.g. [32, 300] → [32, 128] → positional encoding → Transformer encoder
```

GRU consumes `[batch, time, ROI]` directly.

## 10. Dataset Sample Definition

Do **not** use a whole run as one sample — create overlapping temporal windows.

Recommended first setup: window length = 32 volumes, stride = 2, **causal** sequence-to-one prediction.

```python
# For target time t:
X[t-31 : t+1] → y[t]
X[t-31 : t+1] → y_hrf[t]

sample = {
    "x": X[start:end],          # [32, 300]
    "y": y[end - 1],            # scalar
    "y_hrf": y_hrf[end - 1],    # [5]
    "subject_id": subject_id,
    "task": task,
    "run": run,
    "target_volume": end - 1,
}
# batched: x=[B,32,300], y=[B], y_hrf=[B,5]
```

## 11. Subject-Level Splitting

⚠️ Do **not** randomly split overlapping windows — near-identical windows from the same subject/run would leak across train/val.

Use subject-level splits (all runs/windows from a subject stay together). Example for 5 subjects: 3 train / 1 val / 1 test. Scale up later with leave-one-subject-out or grouped CV.

## 12. Initial Dataset Scale

- Subjects: 5
- Task: MOTOR (then possibly EMOTION, WM — kept separate due to differing label spaces / HRF channel counts)
- Runs: LR, optionally RL
- ROIs: 300

| Task | Hard classes (incl. baseline) | HRF channels |
|---|---|---|
| MOTOR | 6 | 5 |
| EMOTION | 3 | 2 |
| WM | 9 | 8 |

Future multi-task-across-tasks design: shared brain encoder + task-specific heads (MOTOR / EMOTION / WM).

## 13. Initial Model Architectures

**GRU**: `input_size=300`, `hidden_size=128`, `num_layers=1`
```
X [B,32,300] → GRU → final hidden [B,128] → classifier [B,6] + HRF regressor [B,5]
```

**Transformer**: input projection 300→128, `d_model=128`, 4 heads, 2 encoder layers, feedforward=256, dropout=0.1
```
X [B,32,300] → input projection → positional encoding → Transformer encoder
             → final temporal token [B,128] → classifier [B,6] + HRF regressor [B,5]
```
Initially use the final sequence token rather than a `[CLS]` token.

## 14. Experimental Ladder

| Encoder | Classification only | Classification + HRF |
|---|---|---|
| GRU | Experiment 1 | Experiment 2 |
| Transformer | Experiment 3 | Experiment 4 |

## 15. Class Imbalance & Metrics

Baseline class 0 may be overrepresented — retain baseline volumes initially, compute class weights from training subjects only:

```
weight[c] = num_training_samples / (num_classes × samples_in_class_c)
```

**Primary classification metrics**: loss, accuracy, balanced accuracy, macro F1, per-class precision/recall, confusion matrix. **Model-selection metric**: validation macro F1.

**Auxiliary HRF regression metrics**: MSE, MAE, per-condition correlation, possibly R².

## 16. Future Multimodal & RAG Direction

- `y_hrf` could later serve as an input stream for a different task: *past brain activity + known task context → predict future brain activity*
- Future multimodal formulation: ROI sequence → brain encoder → `z_brain`; HRF/event sequence → event encoder → `z_event`; combined via contrastive alignment, cosine similarity, gated fusion, or cross-attention
- Future NeuroLens-RAG bridge: brain representation + task representation + model attribution → structured description of active networks/cognitive context → scientific-text embedding → retrieve relevant paper chunks
- More ambitious future objective: directly align brain representation ↔ scientific text representation using paired brain windows and neurocognitive descriptions

## 17. Immediate Next Step

Create **`03_dataset_dataloaders.ipynb`**:

- Task: MOTOR, 5 subjects, runs LR (optionally RL)
- Window length 32, stride 2, causal sequence-to-one samples
- Subject-level train/val/test split
- Class weights from training data only
- PyTorch `Dataset` and `DataLoader`
- Batch shape: `[B, 32, 300]`

Notebook should first inspect all processed bundles and class distributions before building the final `Dataset`/`DataLoader` objects.

---
*Source: consolidated from ChatGPT project-planning conversation.*
