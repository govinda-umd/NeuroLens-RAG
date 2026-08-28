"""Single-subject DTI -> structural connectome pipeline (DIPY-based smoke test).

Produces the four sufficient-statistic arrays designed in
docs/structural/dti-sc-pipeline-plan.md Sec 5, on the Schaefer-300 parcellation
already warped into this subject's native diffusion space
(scripts/warp_atlas_to_native.py, run separately -- this script assumes
<work_dir>/schaefer300_in_dwi_space.nii.gz already exists).

Usage: python scripts/dti_sc_pipeline.py <work_dir> <out_dir>
  work_dir: directory containing data.nii.gz, bvals, bvecs,
            nodif_brain_mask.nii.gz, schaefer300_in_dwi_space.nii.gz
  out_dir:  where to write the four .npy arrays
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


def main(work_dir: Path, out_dir: Path, save_streamlines: bool = True) -> None:
    from dipy.core.gradients import gradient_table
    from dipy.io.image import load_nifti
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.reconst.shm import CsaOdfModel
    from dipy.direction import ProbabilisticDirectionGetter, peaks_from_model
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking import utils
    from dipy.data import get_sphere
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading diffusion data...", flush=True)
    data, affine = load_nifti(str(work_dir / "data.nii.gz"))
    mask, _ = load_nifti(str(work_dir / "nodif_brain_mask.nii.gz"))
    mask = mask.astype(bool)
    atlas_img = nib.load(str(work_dir / "schaefer300_in_dwi_space.nii.gz"))
    atlas = np.asarray(atlas_img.dataobj).astype(int)
    print(f"  data shape={data.shape}, mask voxels={mask.sum()}, atlas labels={len(np.unique(atlas)) - 1}", flush=True)

    gtab = gradient_table(str(work_dir / "bvals"), bvecs=str(work_dir / "bvecs"))
    sphere = get_sphere("repulsion724")

    print(f"[{time.time()-t0:.0f}s] Estimating CSD response function...", flush=True)
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    print(f"  response={response}, ratio={ratio:.3f}", flush=True)

    print(f"[{time.time()-t0:.0f}s] Fitting CSD model + computing peaks (this is the slow step)...", flush=True)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=6)
    csd_peaks = peaks_from_model(
        model=csd_model,
        data=data,
        sphere=sphere,
        mask=mask,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        parallel=True,
        num_processes=-1,
    )
    print(f"[{time.time()-t0:.0f}s] Peaks done.", flush=True)

    print(f"[{time.time()-t0:.0f}s] Fitting CSA/GFA model for stopping criterion...", flush=True)
    csa_model = CsaOdfModel(gtab, sh_order_max=6)
    gfa = csa_model.fit(data, mask=mask).gfa
    stopping_criterion = BinaryStoppingCriterion(mask & (np.nan_to_num(gfa) > 0.1))

    print(f"[{time.time()-t0:.0f}s] Seeding + probabilistic tracking...", flush=True)
    seed_mask = mask & (atlas > 0)
    seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
    print(f"  n_seeds={len(seeds)}", flush=True)

    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(
        csd_peaks.shm_coeff, max_angle=30.0, sphere=sphere
    )
    streamlines_gen = LocalTracking(prob_dg, stopping_criterion, seeds, affine, step_size=1.0)
    streamlines = list(streamlines_gen)
    print(f"[{time.time()-t0:.0f}s] Tracking done. n_streamlines={len(streamlines)}", flush=True)

    if save_streamlines:
        sft = StatefulTractogram(streamlines, atlas_img, Space.RASMM)
        save_tractogram(sft, str(out_dir / "streamlines.trk"))

    print(f"[{time.time()-t0:.0f}s] Building streamline-count + mean-length connectomes...", flush=True)
    n_rois = int(atlas.max())
    count_matrix, grouping = utils.connectivity_matrix(
        streamlines, affine, atlas, return_mapping=True, mapping_as_streamlines=False
    )
    # connectivity_matrix labels 0..n_rois inclusive (0 = background); drop it
    count_matrix = count_matrix[1 : n_rois + 1, 1 : n_rois + 1]

    length_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
    for (i, j), idxs in grouping.items():
        if i == 0 or j == 0 or not idxs:
            continue
        lengths = [len(streamlines[k]) for k in idxs]
        length_matrix[i - 1, j - 1] = float(np.mean(lengths))

    print(f"[{time.time()-t0:.0f}s] Computing ROI volumes...", flush=True)
    voxel_vol_mm3 = float(np.abs(np.linalg.det(affine[:3, :3])))
    roi_volumes = np.array(
        [(atlas == r).sum() * voxel_vol_mm3 for r in range(1, n_rois + 1)], dtype=np.float64
    )

    print(f"[{time.time()-t0:.0f}s] Computing LiFE weights (streamline evaluation)...", flush=True)
    # LiFE can't fit single-node streamlines (immediate-termination artifacts
    # from tracking near mask boundaries) -- filter them first.
    #
    # On subject 100610, fitting LiFE on the full ~1.2M-streamline tractogram
    # was silently OOM-killed after 50+ min on this 16GB machine (LiFE's
    # published use cases top out around 500K streamlines; this project's
    # whole-brain seed density produces far more). A random subsample to
    # LIFE_MAX_STREAMLINES fits comfortably (~2GB RSS, <2 min) and LiFE's
    # own validation (Pestilli et al. 2014) already relies on random
    # streamline subsets for cross-validation, so subsampling before fitting
    # is consistent with how the method is normally used, not a shortcut
    # invented here.
    LIFE_MAX_STREAMLINES = 200_000
    lengths = np.array([len(s) for s in streamlines])
    keep_idx = np.where(lengths >= 2)[0]
    if len(keep_idx) > LIFE_MAX_STREAMLINES:
        rng = np.random.default_rng(0)
        keep_idx = np.sort(rng.choice(keep_idx, size=LIFE_MAX_STREAMLINES, replace=False))
    keep_set = set(keep_idx.tolist())
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    streamlines_life = [streamlines[i] for i in keep_idx]
    print(f"  using {len(streamlines_life)}/{len(streamlines)} streamlines for LiFE (filtered + subsampled)", flush=True)
    try:
        from dipy.tracking.life import FiberModel

        fiber_model = FiberModel(gtab)
        fiber_fit = fiber_model.fit(data, streamlines_life, affine=np.eye(4))
        life_weights = fiber_fit.beta
        life_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)
        for (i, j), idxs in grouping.items():
            if i == 0 or j == 0 or not idxs:
                continue
            remapped = [old_to_new[k] for k in idxs if k in keep_set]
            if remapped:
                life_matrix[i - 1, j - 1] = float(np.sum(life_weights[remapped]))
    except Exception as e:
        print(f"  LiFE fitting failed ({e}); saving zeros as placeholder", flush=True)
        life_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)

    np.save(out_dir / "sc_streamline_count.npy", count_matrix)
    np.save(out_dir / "sc_streamline_count_life.npy", life_matrix)
    np.save(out_dir / "sc_mean_length.npy", length_matrix)
    np.save(out_dir / "roi_volumes.npy", roi_volumes)

    print(f"[{time.time()-t0:.0f}s] Saved 4 arrays to {out_dir}", flush=True)
    print(f"  sc_streamline_count: shape={count_matrix.shape}, nonzero={np.count_nonzero(count_matrix)}, max={count_matrix.max()}", flush=True)
    print(f"  Total wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    work_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    main(work_dir, out_dir)
