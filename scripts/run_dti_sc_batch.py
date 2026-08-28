"""Full-cohort DTI -> SC batch runner, subject-by-subject, over the 174
DTI+movie+all-7-task subjects (data/hcp_triple_modality_eligible_subjects.json
-> full_battery_subject_ids).

Per subject: download diffusion + T1w + warp files from S3 -> warp Schaefer-300
into native diffusion space (scripts/warp_atlas_to_native.py) -> DIPY CSD +
probabilistic tractography -> the 4 sufficient-statistic arrays
(scripts/dti_sc_pipeline.py) -> delete the raw diffusion volume and T1w file
(only the compact arrays are kept; the .trk tractogram is NOT saved in batch
mode -- at ~1.2GB/subject x 174 that alone would be ~200GB, on top of the
~226GB of downloads already budgeted in docs/structural/dti-sc-pipeline-plan.md
Sec 1).

Real per-subject cost, measured on subject 100610 (this same pipeline,
single-subject smoke test, 2026-08-28): ~30 min tractography + ~2 min
LiFE (subsampled to 200K streamlines, Sec 3 note in dti_sc_pipeline.py) +
download time. That's a genuinely multi-day job run sequentially on one
8-core/16GB Mac, not an overnight one -- flagged explicitly here rather than
silently run against the user's original "loop over them tonight" framing.

Resumable: skips any subject whose 4 output arrays already exist. Continues
past a single subject's failure (logs it, moves on) rather than aborting the
whole batch.

Usage: python scripts/run_dti_sc_batch.py [--limit N] [--start-from SUBJECT_ID]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from warp_atlas_to_native import warp_atlas_to_native  # noqa: E402
import dti_sc_pipeline  # noqa: E402

BUCKET = "hcp-openaccess"
PROFILE = "hcp"
REGION = "us-east-1"

WORK_ROOT = Path("/tmp/dti_batch_work")
OUT_ROOT = PROJECT_ROOT / "data" / "structural" / "processed"
SHARED_ATLAS = PROJECT_ROOT / "data" / "structural" / "schaefer300_MNI.nii.gz"
LOG_PATH = PROJECT_ROOT / "logs" / "dti_sc_batch.log"

REQUIRED_S3_FILES = {
    "T1w/Diffusion/data.nii.gz": "data.nii.gz",
    "T1w/Diffusion/bvals": "bvals",
    "T1w/Diffusion/bvecs": "bvecs",
    "T1w/Diffusion/nodif_brain_mask.nii.gz": "nodif_brain_mask.nii.gz",
    "T1w/T1w_acpc_dc_restore.nii.gz": "T1w_acpc_dc_restore.nii.gz",
    "MNINonLinear/xfms/standard2acpc_dc.nii.gz": "standard2acpc_dc.nii.gz",
}

OUTPUT_ARRAYS = [
    "sc_streamline_count.npy",
    "sc_streamline_count_life.npy",
    "sc_mean_length.npy",
    "roi_volumes.npy",
]


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def already_done(subject_id: str) -> bool:
    out_dir = OUT_ROOT / subject_id
    return all((out_dir / f).exists() for f in OUTPUT_ARRAYS)


def download_subject(s3, subject_id: str, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for s3_suffix, local_name in REQUIRED_S3_FILES.items():
        key = f"HCP_1200/{subject_id}/{s3_suffix}"
        dest = work_dir / local_name
        if dest.exists():
            continue
        s3.download_file(BUCKET, key, str(dest))


def process_subject(s3, subject_id: str) -> None:
    work_dir = WORK_ROOT / subject_id
    out_dir = OUT_ROOT / subject_id
    t0 = time.time()

    log(f"{subject_id}: downloading...")
    download_subject(s3, subject_id, work_dir)
    shutil.copy(SHARED_ATLAS, work_dir / "schaefer300_MNI.nii.gz")
    log(f"{subject_id}: downloaded in {time.time()-t0:.0f}s, warping atlas...")

    warp_atlas_to_native(work_dir)
    log(f"{subject_id}: atlas warped, running DIPY pipeline...")

    dti_sc_pipeline.main(work_dir, out_dir, save_streamlines=False)

    # disk discipline: never retain the raw diffusion volume or T1w after
    # the compact arrays are built (docs/structural/dti-sc-pipeline-plan.md Sec 1)
    shutil.rmtree(work_dir, ignore_errors=True)
    log(f"{subject_id}: done in {(time.time()-t0)/60:.1f} min, work dir cleaned up.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    args = parser.parse_args()

    eligible = json.loads(
        (PROJECT_ROOT / "data" / "hcp_triple_modality_eligible_subjects.json").read_text()
    )
    subjects = eligible["full_battery_subject_ids"]
    if args.start_from:
        idx = subjects.index(args.start_from)
        subjects = subjects[idx:]
    if args.limit:
        subjects = subjects[: args.limit]

    log(f"Batch starting: {len(subjects)} subjects queued.")
    n_skipped = n_done = n_failed = 0

    for i, subject_id in enumerate(subjects):
        if already_done(subject_id):
            n_skipped += 1
            continue
        log(f"[{i+1}/{len(subjects)}] Processing {subject_id}...")
        try:
            process_subject(s3=boto3.Session(profile_name=PROFILE).client("s3", region_name=REGION), subject_id=subject_id)
            n_done += 1
        except Exception as e:
            n_failed += 1
            log(f"{subject_id}: FAILED -- {e}")
            log(traceback.format_exc())
            shutil.rmtree(WORK_ROOT / subject_id, ignore_errors=True)

    log(f"Batch complete. done={n_done} skipped={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
