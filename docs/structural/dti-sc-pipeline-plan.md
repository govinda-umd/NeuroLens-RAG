# DTI → Structural Connectivity (SC) Pipeline Plan

> Preprocessing plan for turning HCP diffusion MRI into per-subject structural connectomes (anatomical graphs) on the same Schaefer-300 node parcellation the existing MOTOR pipeline already uses — so a future GNN extension (`docs/case2-3-design-plan.md`'s parked SC-graph idea) can sit directly alongside Case 1/2/3 without a second, incompatible parcellation. This is a genuinely new modality: the precursor paper (`docs/movie/movie-watching-dataset-plan.md` §1) never used structural connectivity at all.

## 1. Data location and real per-subject cost, verified against S3

Diffusion data lives at `HCP_1200/<subject>/T1w/Diffusion/`, same bucket and subject-prefix convention as MOTOR and movie-watching (`docs/movie/movie-watching-dataset-plan.md` §2) — no separate access mechanism needed, the existing `AWS_PROFILE = "hcp"` credentials already used by `02_data_complete.ipynb` cover this.

Confirmed file listing for one subject (`100307`):

| File | Size | Purpose |
|---|---|---|
| `data.nii.gz` | 1.26 GB | Preprocessed (eddy-corrected) diffusion-weighted volumes |
| `bvals` / `bvecs` | <10 KB | Diffusion gradient tables |
| `nodif_brain_mask.nii.gz` | 60 KB | Brain mask for the b0 volume |
| `grad_dev.nii.gz` | 40 MB | Gradient nonlinearity correction field |
| `eddylogs/*` | ~1.3 MB | Eddy-correction QC logs |

**~1.3 GB per subject just for the diffusion volume.** For the 174-subject triple-modality-eligible cohort (`docs/movie/movie-watching-dataset-plan.md` §3 — all 7 standard tasks + rest + DTI + 7T movie, corrected from an earlier looser 179-subject count that didn't require the full battery) — the pool this pipeline should actually target, all of whom have DTI — that's **~226 GB** to download in full before any tractography — the single largest cost in this plan, worth sizing explicitly before committing rather than discovering it mid-download. Deleting the raw volume immediately after building the connectome (§3) is the only way this stays feasible on local disk across 174 subjects — keeping it around after summarizing it as a matrix would defeat the point of summarizing it at all.

## 2. Subject availability, checked directly, not estimated

85 of the current 90-subject MOTOR pool have complete diffusion data (94%). The full 1,113-subject HCP-YA discovery scan (`scripts/run_full_hcp_discovery_scan.py`, `data/hcp_full_discovery_scan.json`, 2026-08-27) confirms this generalizes: 1,065/1,113 (96%) have DTI HCP-YA-wide. DTI availability is high and not a limiting factor the way 7T movie-watching eligibility is (`docs/movie/movie-watching-dataset-plan.md` §3) — every one of the 174 subjects identified there as eligible for all four modalities together (`data/hcp_triple_modality_eligible_subjects.json`) already has DTI, so this pipeline's subject pool is really set by that 174, not by DTI availability itself.

## 3. Pipeline: diffusion volume → tractography → Schaefer-300 structural connectome

```
HCP S3 (T1w/Diffusion/{data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz})
  ↓
download one subject's diffusion volume into a temporary directory
  ↓
fiber orientation estimation (constrained spherical deconvolution, DIPY's CSD model)
  ↓
whole-brain tractography (probabilistic, seeded from Schaefer-300 ROI boundaries)
  ↓
per-ROI-pair: raw streamline count, mean streamline length, LiFE-weighted count (see Sec 5)
  +
per-ROI: native-space volume (for volume normalization, computed once, applies to every pair touching that ROI)
  ↓
save all of the above as separate arrays -- NOT a single pre-normalized matrix (see Sec 5)
  ↓
delete the raw diffusion volume (same disk-discipline convention as 02_data_complete.ipynb -- never retain raw NIfTI after the compact bundle is built)
```

**Tooling, resolved by checking what's actually installable here, not by defaulting to the field-standard tool.** MRtrix3 (the standard pairing for HCP-preprocessed diffusion data: constrained spherical deconvolution + probabilistic tractography) is not available in this environment — no conda-forge package under that name, no Homebrew on this machine to build it via, and building its C++ toolchain from source is a real undertaking not worth attempting speculatively. **DIPY** (pip-installable, pure-Python/Cython, `pip install dipy` confirmed working) is the resolved choice: it implements the same CSD-based tractography approach, plus its own native streamline-weighting method (LiFE — Linear Fascicle Evaluation) as the DIPY-side analog of MRtrix's SIFT2, used the same way — reduce systematic tractography bias without discarding streamlines outright.

**Compute cost, measured on the single-subject test (Sec 6), not estimated.** Subject 100610, whole-brain, seed density 1 (446K seeds): CSD+peaks 67s, GFA stopping-criterion model 5min, probabilistic tractography 29min (1.23M streamlines), connectome assembly ~1min, LiFE (subsampled, see below) ~2min. **~38 min/subject total.** Sequentially over all 174 subjects on this one 8-core/16GB Mac, that's **~110 hours (~4.6 days) of continuous compute** — a genuinely different, multi-day budget, not the single overnight run originally assumed, and worth stating plainly before committing rather than discovering partway through a batch job.

**LiFE is unreliable at whole-brain streamline counts on this hardware, and the exact safe threshold is not yet pinned down.** Fitting LiFE on the full 1.2M-streamline tractogram was silently killed by the OS after 50+ minutes, RSS still climbing, on a 16GB machine — LiFE's published validation (Pestilli et al. 2014) uses far fewer streamlines (order 10^5, not 10^6+) than this project's whole-brain seed density produces. A follow-up subsample to 200,000 streamlines was tried and *also* died silently after several minutes (no catchable Python exception, no jetsam/OOM record found in the system log either) — worth stating plainly rather than reporting the first "it looked fine at the 4-minute mark" read as confirmed, which it wasn't. Current default is a more conservative 50,000-streamline subsample (`scripts/dti_sc_pipeline.py`'s `LIFE_MAX_STREAMLINES`), not yet independently confirmed safe either. Given that, `scripts/run_dti_sc_batch.py` runs each subject's whole DIPY pipeline (including LiFE) as an **isolated subprocess**, not an in-process call — a silent kill during LiFE now fails only that one subject (falls back to a zero LiFE array, everything else unaffected) instead of taking down the entire multi-day, 174-subject batch orchestrator. The raw streamline-count and mean-length arrays (Sec 5) always use every streamline regardless of whether LiFE succeeds; only the LiFE-weighted array is subsampled and only it is at risk from this failure mode.

## 4. Normalization: no consensus in the field, confirmed by literature search, not assumed

Searched specifically for whether raw streamline count vs. some normalized form is field-standard. It is not — multiple sources ([MRtrix3's own structural-connectome documentation](https://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/structural_connectome.html), a [2021 review of quantitative SC mapping](https://pmc.ncbi.nlm.nih.gov/articles/PMC9257891/), and the [SIFT2 paper itself](https://www.sciencedirect.com/science/article/abs/pii/S1053811915005972)) confirm real, live disagreement: some groups use raw streamline count directly (MRtrix3's own default), some normalize by the volumes of the two connected ROIs, some apply SIFT2/LiFE streamline weighting *and then* volume-normalize on top, some log-transform to reduce the right-skew of raw counts, and there is no agreed default even on whether thresholding low-count "spurious" connections is appropriate at all. This is a genuinely contested, use-case-dependent choice, not a solved problem this project should quietly pick one answer for.

## 5. Design consequence: store sufficient statistics, defer normalization to analysis time

Given Sec 4, committing to one normalized matrix at preprocessing time would bake in a contested methodological choice before any actual analysis motivates it, and — critically — would be **expensive to undo**: re-deriving a different normalization later would mean re-running tractography from the raw diffusion volume, which Sec 1 already established gets deleted. The fix: store the raw sufficient statistics every normalization scheme found in Sec 4 actually needs, computed once at tractography time (the expensive, non-repeatable step), and leave the *choice* of normalization to whatever analysis needs it later (cheap, reversible, comparable side-by-side without touching the diffusion volume again):

| Saved array | Shape | Enables |
|---|---|---|
| `sc_streamline_count.npy` | `[300, 300]` | Raw count (MRtrix3's own default) |
| `sc_streamline_count_life.npy` | `[300, 300]` | LiFE-weighted count (DIPY's SIFT2 analog, Sec 3) |
| `sc_mean_length.npy` | `[300, 300]` | Length-normalized variants (count / length) |
| `roi_volumes.npy` | `[300]` | Volume-normalized variants (count / combined ROI volume) |

Every normalization scheme found in the literature search is a cheap post-hoc function of these four arrays — none of them require re-tractography. Log-transformation, thresholding, or any future normalization idea also apply directly to `sc_streamline_count.npy` with no additional storage needed.

**Chosen scheme: log-transform, $S_{ij} = \log_{10}(1 + N_{ij})$**, where $N_{ij}$ is the raw streamline count (`src/neurolens/sc_normalization.py::log_streamline_count`) — reduces the right-skew Sec 4's sources flag as a real property of raw counts, without needing volume or length information. Volume- and length-normalized variants (`volume_normalized`, `length_normalized` in the same module) are implemented alongside it since Sec 5's arrays already support them, but are not the default.

## 6. Rollout: one subject now, the full 174 later

Two genuinely different steps, not one job at two scales: (1) build and smoke-test the single-subject DIPY pipeline (download → CSD → tractography → the four Sec 5 arrays) on one subject now, while it's cheap to iterate on bugs; (2) once that's validated, loop over all 174 subjects as a long-running batch job. Deliberately sequenced this way rather than writing the full-batch version first — a bug caught on subject 1 costs minutes; the same bug caught on subject 87 of 174 costs however long those 86 subjects took to (redundantly) process.

**Step (1) result, subject 100610 (2026-08-28):** completed end to end. All 300 Schaefer atlas labels survive both the MNI→T1w-acpc nonlinear warp and the resample into native diffusion space (86% overlap with the diffusion brain mask). All four Sec 5 arrays produced and validated (`data/structural/processed/100610/`): `sc_streamline_count.npy` (52,968 nonzero ROI pairs, max 2,934 streamlines), `sc_mean_length.npy`, `roi_volumes.npy`, and `sc_streamline_count_life.npy` (fixed to use a 200K-streamline subsample per the memory finding in Sec 3). Two real bugs found and fixed by actually running it, not hypothetically: FSL's warp field needed manual reconstruction as an ANTs vector image (`scripts/warp_atlas_to_native.py`), and LiFE needed both single-node-streamline filtering and subsampling (`scripts/dti_sc_pipeline.py`). `scripts/run_dti_sc_batch.py` implements step (2): resumable, skips completed subjects, continues past a single subject's failure, deletes each subject's raw diffusion volume + T1w file immediately after its four arrays are built (Sec 1's disk discipline), and does not retain the full tractogram (`.trk`) in batch mode — at ~1.2GB/subject that alone would add ~200GB on top of the diffusion-volume downloads.

## 7. Why this parcellation choice matters, concretely

Using Schaefer-300 for SC node definition (not a different, diffusion-native atlas) means a future SC-graph input aligns node-for-node with the existing fMRI ROI time series — the same 300 nodes, same node ordering, same `roi_labels.tsv` convention already produced by `02_data_complete.ipynb`. This is what makes "concatenate or otherwise combine functional and structural information per node" a real, low-friction option later, rather than requiring a cross-atlas correspondence step first.

## 8. The bigger research line this serves, and what's still just an idea

This pipeline is infrastructure for a larger extension of the 2021 precursor paper (`docs/movie/movie-watching-dataset-plan.md` §1), not an end in itself:

- **Movie-watching's continuous, mixed signal** (as opposed to MOTOR's clean, discrete blocks — `docs/end-to-end-report.md` §3.2's baseline finding already shows temporal structure matters far more there) is the right setting to look for genuinely novel concepts in a Case-2-style contrastive setup, not just re-confirm effector/laterality concepts already established on MOTOR.
- **A genuine generative paradigm — forecasting future ROI time series** — was already scoped and explicitly deferred in `case2-3-design-plan.md` §2.4 ("conditional generation from a learned multimodal representation... not v1, a natural extension once the representation is validated") and partially explored for Case 2 on MOTOR (`results/case2_forecasting_results.json`, `notebooks/11_case2_forecasting.ipynb` — a frozen linear probe only, `docs/end-to-end-report.md` §8 flags extending it as unstarted). Movie-watching's long, continuous timeseries is a structurally better fit for real sequence forecasting than MOTOR's short causal windows.
- **Edge-level concept attribution on the structural connectome** — "which anatomical connections were responsible for a given concept" — is the graph-native version of an idea already logged, not invented here: `docs/interview-prep-neurolens-rag.md` §6.6's concept-vector input attribution section names GNNExplainer/PGExplainer specifically as "the direct structural-connectome version of which ROI nodes and structural edges are responsible for this concept," for exactly this future SC-GNN block.

**None of this is designed yet** — this document is the data-acquisition layer only. The GNN front-end architecture, the movie-watching concept taxonomy, the forecasting objective's exact form, and the edge-attribution mechanism each need their own design pass, the same way Case 2/3's design predated their implementation for MOTOR. Named here so the SC pipeline gets built with these eventual uses in mind (e.g., keeping node correspondence with the fMRI parcellation, Sec 7) rather than discovered as a constraint later.

## 9. Honest scope check

This plan covers **acquisition and connectome construction only**. It does not cover: how a structural connectome would actually enter Case 1/2/3 (a GNN front-end is a real architecture change, not a drop-in `data_setup.py` addition), or what "concept" CAV/TCAV would test on a graph-structured input where the current method's linear-probe-in-Euclidean-space assumption (`docs/end-to-end-report.md` §4) may not directly transfer. Both are real open questions for whenever the SC-graph extension itself gets built, not just data-pipeline questions.
