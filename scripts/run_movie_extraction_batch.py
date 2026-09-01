"""Full-cohort movie-watching ROI-timeseries extraction, run-by-run, over the
174 DTI+movie+full-task-battery subjects
(data/hcp_triple_modality_eligible_subjects.json -> full_battery_subject_ids)
x 4 movie runs each (tfMRI_MOVIE{1,2,3,4}_7T_{AP,PA}).

Subject pool matches the just-completed DTI/SC batch (scripts/run_dti_sc_batch.py)
deliberately, not the full 184-subject movie-eligible pool -- so movie and
structural data exist for the same subjects, sidestepping (for these two
modalities at least) the pool-mismatch question flagged in
docs/movie/movie-watching-dataset-plan.md Sec 3 / docs/end-to-end-report.md
Sec 9.6. Not a decision about MOTOR re-use, just about not needlessly
introducing a third, non-overlapping cohort.

Per run: download the plain functional NIfTI + Movement_Regressors.txt from
S3 -> extract Schaefer-300 ROI time series via NiftiLabelsMasker, identical
settings to the existing MOTOR pipeline (scripts/movie_roi_extraction.py) ->
delete the raw functional NIfTI immediately after (~1.4GB/run; peak local
disk stays low even though the cumulative download across all 696 runs is
~0.95TB, confirmed via S3 HeadObject before committing to this batch).

No y/y_hrf/task labels are produced -- movie-watching has no discrete
conditions (docs/movie/movie-watching-dataset-plan.md Sec 4). This batch
produces X (the fMRI time series) only, deliberately, per the explicit
instruction to prepare the data first and decide what to do with it once
all four runs are in hand for every subject.

Resumable: skips any run whose output already exists. Continues past a
single run's failure. Runs each run's extraction as an isolated subprocess
(scripts/movie_roi_extraction.py), same defensive architecture as the DTI
batch, even though NiftiLabelsMasker is far less failure-prone than DIPY's
tractography/LiFE -- consistent, not costly.

Usage: python scripts/run_movie_extraction_batch.py [--limit N] [--start-from SUBJECT_ID]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUCKET = "hcp-openaccess"
PROFILE = "hcp"
REGION = "us-east-1"

MOVIE_RUNS = [
    "tfMRI_MOVIE1_7T_AP",
    "tfMRI_MOVIE2_7T_PA",
    "tfMRI_MOVIE3_7T_PA",
    "tfMRI_MOVIE4_7T_AP",
]

WORK_ROOT = Path("/tmp/movie_batch_work")
OUT_ROOT = PROJECT_ROOT / "data" / "movie" / "processed"
LOG_PATH = PROJECT_ROOT / "logs" / "movie_extraction_batch.log"

PYTHON = sys.executable
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "movie_roi_extraction.py"
RUN_TIMEOUT_S = 1800  # 30 min ceiling/run -- measured cost is a few minutes; guards against a hang

OUTPUT_FILES = ["X.npy", "frame_times.npy", "roi_labels.tsv", "metadata.json"]


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def already_done(subject_id: str, run_name: str) -> bool:
    out_dir = OUT_ROOT / subject_id / run_name
    return all((out_dir / f).exists() for f in OUTPUT_FILES)


def download_run(s3, subject_id: str, run_name: str, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for suffix, local_name in [
        (f"{run_name}.nii.gz", f"{run_name}.nii.gz"),
        ("Movement_Regressors.txt", "Movement_Regressors.txt"),
    ]:
        key = f"HCP_1200/{subject_id}/MNINonLinear/Results/{run_name}/{suffix}"
        dest = work_dir / local_name
        if dest.exists():
            continue
        s3.download_file(BUCKET, key, str(dest))


def process_run(s3, subject_id: str, run_name: str) -> None:
    work_dir = WORK_ROOT / subject_id / run_name
    out_dir = OUT_ROOT / subject_id / run_name
    t0 = time.time()

    log(f"{subject_id}/{run_name}: downloading...")
    download_run(s3, subject_id, run_name, work_dir)
    log(f"{subject_id}/{run_name}: downloaded in {time.time()-t0:.0f}s, extracting...")

    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [PYTHON, str(EXTRACT_SCRIPT), str(work_dir), str(out_dir), run_name],
        timeout=RUN_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"movie_roi_extraction.py exited with code {result.returncode}")

    shutil.rmtree(work_dir, ignore_errors=True)
    log(f"{subject_id}/{run_name}: done in {(time.time()-t0)/60:.1f} min, work dir cleaned up.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of (subject, run) pairs")
    parser.add_argument("--start-from", type=str, default=None, help="Subject ID to resume from")
    args = parser.parse_args()

    eligible = json.loads(
        (PROJECT_ROOT / "data" / "hcp_triple_modality_eligible_subjects.json").read_text()
    )
    subjects = eligible["full_battery_subject_ids"]
    if args.start_from:
        idx = subjects.index(args.start_from)
        subjects = subjects[idx:]

    jobs = [(sid, run) for sid in subjects for run in MOVIE_RUNS]
    if args.limit:
        jobs = jobs[: args.limit]

    log(f"Batch starting: {len(jobs)} (subject, run) pairs queued ({len(subjects)} subjects x {len(MOVIE_RUNS)} runs).")
    n_skipped = n_done = n_failed = 0
    s3 = boto3.Session(profile_name=PROFILE).client("s3", region_name=REGION)

    for i, (subject_id, run_name) in enumerate(jobs):
        if already_done(subject_id, run_name):
            n_skipped += 1
            continue
        log(f"[{i+1}/{len(jobs)}] Processing {subject_id}/{run_name}...")
        try:
            process_run(s3, subject_id, run_name)
            n_done += 1
        except Exception as e:
            n_failed += 1
            log(f"{subject_id}/{run_name}: FAILED -- {e}")
            shutil.rmtree(WORK_ROOT / subject_id / run_name, ignore_errors=True)

    log(f"Batch complete. done={n_done} skipped={n_skipped} failed={n_failed}")


if __name__ == "__main__":
    main()
