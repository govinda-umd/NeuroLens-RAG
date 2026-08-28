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

**~1.3 GB per subject just for the diffusion volume.** For the 85 of the current 90 subjects confirmed to have DTI data (`data/subject_discovery_dti_movie.json`), that's **~110 GB** to download in full before any tractography — the single largest cost in this plan, worth sizing explicitly before committing rather than discovering it mid-download.

## 2. Subject availability, checked directly, not estimated

85 of the current 90-subject MOTOR pool have complete diffusion data (94%) — confirmed 2026-08-27 alongside the movie-watching discovery scan (`data/subject_discovery_dti_movie.json`), consistent with the earlier, broader scan reported in project memory (958 of 1,013 candidates, 95%). DTI availability is high and not a limiting factor the way 7T movie-watching eligibility is (`docs/movie/movie-watching-dataset-plan.md` §3) — this pipeline can essentially reuse the entire existing MOTOR subject pool.

## 3. Proposed pipeline: diffusion volume → tractography → Schaefer-300 structural connectome

```
HCP S3 (T1w/Diffusion/{data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz})
  ↓
download one subject's diffusion volume into a temporary directory
  ↓
fiber orientation estimation (constrained spherical deconvolution or DTI tensor fit)
  ↓
whole-brain tractography (probabilistic, seeded from Schaefer-300 ROI boundaries)
  ↓
streamline count / density between every ROI pair -> [300, 300] structural connectivity matrix
  ↓
save SC matrix + QC metrics (streamline count, tractography convergence diagnostics)
  ↓
delete the raw diffusion volume (same disk-discipline convention as 02_data_complete.ipynb -- never retain raw NIfTI after the compact bundle is built)
```

**Real design decisions this plan doesn't resolve yet, flagged rather than guessed at:**

- **Tractography algorithm and tool.** HCP's own minimally-preprocessed diffusion data is compatible with MRtrix3 (constrained spherical deconvolution + probabilistic tractography, the field-standard combination for this exact data) or DSI Studio. Neither is currently a project dependency (`environment.yml` has no diffusion-MRI tooling) — this is a new external dependency to add, not a pure-Python addition like everything built so far in `src/neurolens/`.
- **Streamline count vs. normalized connectivity.** Raw streamline counts between ROI pairs are biased by ROI size and tract length; whether to use raw counts, length-normalized counts, or a fractional anisotropy-weighted variant is an open methodological choice with real consequences for what a downstream GNN would learn from — not decided here.
- **Compute cost.** Whole-brain probabilistic tractography is CPU/GPU-intensive per subject (typically minutes to tens of minutes depending on streamline count and tool), unlike the existing fMRI pipeline's fast ROI-timeseries extraction. 85 subjects at even 10-15 min each is a genuinely different compute budget than anything run so far in this project — needs sizing with a single-subject timing test before committing to the full batch, not assumed.

## 4. Why this parcellation choice matters, concretely

Using Schaefer-300 for SC node definition (not a different, diffusion-native atlas) means a future SC-graph input aligns node-for-node with the existing fMRI ROI time series — the same 300 nodes, same node ordering, same `roi_labels.tsv` convention already produced by `02_data_complete.ipynb`. This is what makes "concatenate or otherwise combine functional and structural information per node" a real, low-friction option later, rather than requiring a cross-atlas correspondence step first.

## 5. Honest scope check

This plan covers **acquisition and connectome construction only**. It does not cover: how a structural connectome would actually enter Case 1/2/3 (a GNN front-end is a real architecture change, not a drop-in `data_setup.py` addition — flagged as future work in `case2-3-design-plan.md`, not resolved here), or what "concept" CAV/TCAV would test on a graph-structured input where the current method's linear-probe-in-Euclidean-space assumption (`docs/end-to-end-report.md` §4) may not directly transfer. Both are real open questions for whenever the SC-graph extension itself gets built, not just data-pipeline questions.
