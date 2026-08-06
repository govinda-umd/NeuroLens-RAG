# Literature Notes — Eva Dyer Lab: Tokenization Strategies for Neural Time-Series

> **Research notes only — nothing here has been implemented.** Collected to inform tomorrow's discussion on tokenization design for NeuroLens-RAG (currently using the naive "one fMRI volume = one token" scheme; see [ml-design-report.md §2](ml-design-report.md#2-input--output-specification)). Eva Dyer leads the NerDS Lab at Georgia Tech (Biomedical Engineering), focused on data-centric AI, representation learning, and AI for neuroscience. Most of the lab's tokenization work targets spiking/electrophysiology data rather than fMRI directly, but the underlying design principles transfer.

## Core idea across the lab's work

Rather than binning continuous/point-process neural activity into fixed time windows and treating each bin as a "frame" (which is what our current `X[t] = 300-dim ROI vector` scheme effectively does), the lab's line of work tokenizes **individual events** (spikes) and lets the model attend across an irregular, variable-length set of tokens. Each token carries **identity** (which unit/channel it came from) and **timing** (when it happened), encoded separately and combined via attention — rather than baking position into a fixed grid index.

## Key papers

### 1. POYO — "A Unified, Scalable Framework for Neural Population Decoding" (Azabou, Dyer et al., NeurIPS 2023)
- Tokenizes **individual spikes**, not binned counts. Each spike token = (unit identity embedding) + (timestamp via rotary position embedding for relative timing).
- Uses a Perceiver-IO-style cross-attention "pooling" step to compress a variable, potentially huge number of spike tokens (and variable number of units across sessions/animals) into a fixed set of latent tokens before the main transformer — this is what lets the same architecture handle recordings with wildly different numbers of channels/units across subjects and sessions.
- Site: https://poyo-brain.github.io/

### 2. POSSM — "Generalizable, real-time neural decoding with hybrid state-space models" (NeurIPS 2025)
- Extends POYO-style spike tokenization (cross-attention module) but swaps the transformer backbone for a recurrent state-space model (SSM), trading some accuracy for ~9x faster, causal, millisecond-resolution online decoding.
- Notably: the spike tokenization was **adapted to handle binned counts via "value embeddings"** for applications like speech decoding — i.e., when the signal isn't naturally point-process (spikes), a continuous/count value at a given channel+time is embedded much like a spike token would be (identity + timing + magnitude), rather than reverting to a plain fixed-grid vector.
- This is the closest analogue in the lab's public work to "tokenizing a continuous signal" rather than a literal spike train.
- arXiv: https://arxiv.org/pdf/2506.05320

### 3. Self-supervised modeling of human intracranial recordings during natural behavior (NeurIPS 2025)
- Directly relevant to continuous multi-channel signals (SEEG/ECoG — continuous voltage traces, much closer in spirit to fMRI ROI time series than spike trains are).
- Tokenizes **per-electrode activity** (i.e., a patch of the continuous signal at one electrode over some time window becomes a token), and **injects each electrode's 3D anatomical coordinates** as positional/identity information.
- Attention operates jointly across **time and electrodes**, with **subject-specific heads** to absorb inter-subject variability in electrode placement and count, while sharing the backbone across all subjects.
- Authors: Mahato, Xiao, Andre, Chau, Ma, Knight, Nguyen, Hu, Brunton, Beauchamp, Pesaran, Shuvaev, Dyer.

### 4. Multi-session, multi-task neural decoding from distinct cell-types (ICLR 2025, Spotlight)
- Data: two-photon calcium imaging (Allen Brain Observatory) — a genuinely continuous fluorescence signal, unlike spikes.
- Uses a transformer trained jointly across sessions/cell-types/tasks; public description doesn't detail a novel tokenizer beyond the shared multi-session transformer framework, but it's evidence the same general architecture family is being pushed onto continuous imaging signals, not just spikes.

### 5. Neural Encoding and Decoding at Scale — NEDS (ICML 2025, Spotlight)
- Neuropixels electrophysiology across 83 animals; multimodal transformer using a **multi-task masking strategy** that alternates masking neural vs. behavioral tokens (a joint-token-space masked-modeling scheme, conceptually similar to BERT-style pretraining but across modalities).

## Relevance to NeuroLens-RAG's fMRI tokenization decision

Our current setup treats each fMRI volume (300-dim ROI vector, evenly spaced in time by the TR) as one token, projected linearly to `d_model`. This is closest in spirit to the **grid/patch-tokenization** end of the spectrum (like the intracranial-recordings paper's per-electrode patches), not the event/spike end — which makes sense, since fMRI BOLD signal is slow, smooth, and evenly sampled, unlike spikes. A few ideas worth weighing against our current scheme:

- **Spatial position embeddings from atlas coordinates.** Instead of (or in addition to) a purely temporal positional encoding, embed each of the 300 ROI channels using their Schaefer-atlas spatial coordinates or network label (7-network solution), analogous to how the intracranial paper injects 3D electrode coordinates. Could help the model generalize across subjects with a principled notion of "where in the brain."
- **Subject-specific calibration heads.** Rather than relying purely on subject-level data splitting to avoid leakage, add a small subject-conditioned calibration layer (as in the intracranial paper) so the shared encoder can absorb known inter-subject variability explicitly instead of only being evaluated against it.
- **Patch tokenization over ROI groups or short time spans**, rather than one token per single volume — e.g., token = small block of `(few ROIs) × (few TRs)`, cutting sequence length and possibly capturing local spatiotemporal structure more directly, at the cost of losing the clean "one token = one timepoint" interpretation used in the auxiliary HRF-regression framing.
- **Value-embedding style tokens** (per POSSM's adaptation for non-spike signals) — could inform how to fold ROI magnitude into the token embedding jointly with position, instead of a flat linear projection of the whole 300-dim vector.
- **Latent bottleneck pooling (Perceiver-IO/POYO-style cross-attention)** — potentially useful later if the project scales beyond 300 ROIs (e.g. subcortical ROIs, voxelwise, or multi-atlas fusion) where the token count/channel count grows large and variable across configurations.

None of the above should be implemented before tomorrow's discussion — flagging them here as candidate directions to weigh against the current "one volume = one token" baseline once Experiments 1–4 establish a reference point.

## Sources

- [POYO-1 project page](https://poyo-brain.github.io/)
- [POYO: A Unified, Scalable Framework for Neural Population Decoding (alphaXiv)](https://www.alphaxiv.org/overview/2310.16046v1)
- [Generalizable, real-time neural decoding with hybrid state-space models (POSSM), arXiv](https://arxiv.org/pdf/2506.05320)
- [POSSM — NeurIPS 2025 poster page](https://neurips.cc/virtual/2025/poster/120204)
- [Multi-session, multi-task neural decoding from distinct cell-types, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/953390c834451505703c9da45de634d8-Paper-Conference.pdf)
- [NeurIPS 2025: A scalable self-supervised method for modeling human intracranial recordings during natural behavior](https://neurips.cc/virtual/2025/loc/san-diego/132700)
- [Mehdi Azabou — publication list](https://www.mehai.dev/publications)
- [Eva Dyer — Google Scholar](https://scholar.google.com/citations?user=Sb_jcHcAAAAJ&hl=en)
- [Eva Dyer — ResearchGate profile](https://www.researchgate.net/profile/Eva-Dyer)
