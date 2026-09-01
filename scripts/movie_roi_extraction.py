"""Single-run movie-watching ROI-timeseries extraction.

Mirrors notebooks/02_data_complete.ipynb's MOTOR extraction exactly (same
Schaefer-300 atlas, same NiftiLabelsMasker settings, same Movement_Regressors
confound regression) so the two modalities are directly comparable -- no new
methodological choice introduced here. The one real difference: movie-
watching has no discrete task conditions, so there is no y/y_hrf/task_mask/
events output (docs/movie/movie-watching-dataset-plan.md Sec 4) -- this
script produces X only. Clip-identity labels, windowing convention, and
Case 3's alignment target are deliberately deferred design questions, not
solved here.

Usage: python scripts/movie_roi_extraction.py <work_dir> <out_dir> <run_name>
  work_dir: directory containing <run_name>.nii.gz and Movement_Regressors.txt
  out_dir:  where to write X.npy, frame_times.npy, roi_labels.tsv, metadata.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

STANDARDIZE = "zscore_sample"
STANDARDIZE_CONFOUNDS = True
DETREND = True
LOW_PASS_HZ = None
HIGH_PASS_HZ = None
ATLAS_N_ROIS = 300
ATLAS_NETWORKS = 7
ATLAS_RESOLUTION_MM = 2


def fetch_schaefer_atlas(cache_dir: Path):
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas = fetch_atlas_schaefer_2018(
        n_rois=ATLAS_N_ROIS,
        yeo_networks=ATLAS_NETWORKS,
        resolution_mm=ATLAS_RESOLUTION_MM,
        data_dir=str(cache_dir / "atlases"),
    )
    labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in atlas.labels]
    if len(labels) == ATLAS_N_ROIS + 1:
        labels = labels[1:]
    if len(labels) != ATLAS_N_ROIS:
        raise ValueError(f"Atlas label count {len(labels)} does not match {ATLAS_N_ROIS}.")
    return Path(atlas.maps), labels


def main(work_dir: Path, out_dir: Path, run_name: str) -> None:
    from nilearn.maskers import NiftiLabelsMasker

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path.home() / ".cache" / "neurolens_nilearn"

    func_path = work_dir / f"{run_name}.nii.gz"
    movement_path = work_dir / "Movement_Regressors.txt"

    print(f"Loading {func_path}...", flush=True)
    img = nib.load(func_path)
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D fMRI, got shape {img.shape}.")
    n_timepoints = int(img.shape[-1])
    tr_seconds = float(img.header.get_zooms()[3])
    if not np.isfinite(img.affine).all():
        raise ValueError("NIfTI affine contains non-finite values.")
    print(f"  shape={img.shape}, TR={tr_seconds}s", flush=True)

    movement = np.loadtxt(movement_path)
    if movement.ndim == 1:
        movement = movement[:, None]
    if movement.shape[0] != n_timepoints:
        raise ValueError(
            f"Motion regressors and NIfTI have different time dimensions: "
            f"{movement.shape[0]} vs {n_timepoints}."
        )
    if not np.isfinite(movement).all():
        raise ValueError("Movement regressors contain non-finite values.")

    print("Fetching Schaefer-300 atlas...", flush=True)
    atlas_path, roi_labels = fetch_schaefer_atlas(cache_dir)

    print("Extracting ROI time series...", flush=True)
    masker = NiftiLabelsMasker(
        labels_img=str(atlas_path),
        labels=roi_labels,
        background_label=0,
        standardize=STANDARDIZE,
        standardize_confounds=STANDARDIZE_CONFOUNDS,
        detrend=DETREND,
        low_pass=LOW_PASS_HZ,
        high_pass=HIGH_PASS_HZ,
        t_r=tr_seconds,
        resampling_target="data",
        memory=str(cache_dir),
        memory_level=1,
        verbose=0,
    )
    X = masker.fit_transform(str(func_path), confounds=movement)
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2 or X.shape[1] != ATLAS_N_ROIS:
        raise ValueError(f"Expected [time, {ATLAS_N_ROIS}], got {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("ROI matrix contains non-finite values.")

    frame_times = np.arange(n_timepoints, dtype=np.float64) * tr_seconds

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "frame_times.npy", frame_times)
    with open(out_dir / "roi_labels.tsv", "w") as f:
        f.write("roi_label\n")
        for label in roi_labels:
            f.write(f"{label}\n")

    metadata = {
        "run_name": run_name,
        "n_timepoints": n_timepoints,
        "n_rois": ATLAS_N_ROIS,
        "tr_seconds": tr_seconds,
        "standardize": STANDARDIZE,
        "detrend": DETREND,
        "atlas": f"Schaefer2018_{ATLAS_N_ROIS}Parcels{ATLAS_NETWORKS}Networks_{ATLAS_RESOLUTION_MM}mm",
        "confounds": "Movement_Regressors.txt (12-column: 6 motion params + 6 derivatives)",
        "has_task_labels": False,
        "note": "No y/y_hrf -- movie-watching has no discrete task conditions (docs/movie/movie-watching-dataset-plan.md Sec 4). Clip-identity labels deferred.",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved X={X.shape} to {out_dir}", flush=True)


if __name__ == "__main__":
    work_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    run_name = sys.argv[3]
    main(work_dir, out_dir, run_name)
