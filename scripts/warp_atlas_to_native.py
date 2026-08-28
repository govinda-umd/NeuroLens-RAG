"""Warp the Schaefer-300 MNI atlas into one subject's native diffusion space.

Two steps, confirmed working on subject 100610:
1. MNI (atlas, 1mm) -> T1w acpc-dc (subject native, 0.7mm) via HCP's own
   nonlinear warp field (standard2acpc_dc.nii.gz). FSL stores this warp as a
   4D scalar nifti (X,Y,Z,3) of mm displacements; antspyx needs it as a 3D
   vector-component ANTsImage, so it's reconstructed manually from the raw
   array + affine rather than read directly.
2. T1w acpc-dc (0.7mm) -> diffusion native grid (1.25mm) via resampling only,
   not a second registration -- HCP's T1w/Diffusion/ folder means diffusion
   data is already registered to the subject's T1w acpc space, just at a
   different voxel size.

Usage: python scripts/warp_atlas_to_native.py <work_dir>
  work_dir must contain: T1w_acpc_dc_restore.nii.gz, standard2acpc_dc.nii.gz,
  nodif_brain_mask.nii.gz, schaefer300_MNI.nii.gz
  Writes: schaefer300_in_T1w_acpc.nii.gz, schaefer300_in_dwi_space.nii.gz
"""

from __future__ import annotations

import sys
from pathlib import Path

import ants
import nibabel as nib
import numpy as np


def warp_atlas_to_native(work_dir: Path) -> Path:
    t1w_ref = ants.image_read(str(work_dir / "T1w_acpc_dc_restore.nii.gz"))
    atlas_mni = ants.image_read(str(work_dir / "schaefer300_MNI.nii.gz"))

    img = nib.load(str(work_dir / "standard2acpc_dc.nii.gz"))
    data = np.asarray(img.dataobj).astype("float32")
    affine = img.affine
    spacing = nib.affines.voxel_sizes(affine)
    origin = affine[:3, 3]
    direction = affine[:3, :3] / spacing

    warp_field = ants.from_numpy(
        data,
        origin=tuple(origin.astype(float)),
        spacing=tuple(spacing.astype(float)),
        direction=direction,
        has_components=True,
    )
    transform = ants.transform_from_displacement_field(warp_field)
    warped_atlas = ants.apply_ants_transform_to_image(
        transform, atlas_mni, t1w_ref, interpolation="nearestneighbor"
    )
    t1w_path = work_dir / "schaefer300_in_T1w_acpc.nii.gz"
    ants.image_write(warped_atlas, str(t1w_path))

    arr = warped_atlas.numpy()
    n_labels = len(np.unique(arr)) - 1
    print(f"MNI->T1w_acpc: shape={arr.shape}, labels={n_labels}, nonzero_frac={(arr > 0).mean():.3f}")
    if n_labels != 300:
        raise RuntimeError(f"Expected 300 labels after MNI->T1w warp, got {n_labels}")

    dwi_mask = ants.image_read(str(work_dir / "nodif_brain_mask.nii.gz"))
    atlas_dwi = ants.resample_image_to_target(warped_atlas, dwi_mask, interp_type="nearestNeighbor")
    dwi_path = work_dir / "schaefer300_in_dwi_space.nii.gz"
    ants.image_write(atlas_dwi, str(dwi_path))

    arr2 = atlas_dwi.numpy()
    mask_arr = dwi_mask.numpy()
    n_labels2 = len(np.unique(arr2)) - 1
    overlap = mask_arr[arr2 > 0].mean()
    print(f"T1w_acpc->dwi: shape={arr2.shape}, labels={n_labels2}, mask_overlap={overlap:.3f}")
    if n_labels2 != 300:
        raise RuntimeError(f"Expected 300 labels after resample to dwi space, got {n_labels2}")

    return dwi_path


if __name__ == "__main__":
    work_dir = Path(sys.argv[1])
    warp_atlas_to_native(work_dir)
