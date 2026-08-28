"""Full HCP-YA S1200 discovery scan (user-proposed 2026-08-27): for every
subject in the accessible S3 bucket, which task-fMRI datasets and DTI data
do they actually have -- one table, not a series of one-off checks.

Restricted to the 'hcp-openaccess' bucket via the existing 'hcp' AWS
profile (the credentials already used by 02_data_complete.ipynb) and to
MNINonLinear/ (preprocessed) data specifically, so anything this table
marks available is immediately downloadable with no further preprocessing
needed -- consistent with how MOTOR data was originally acquired.

2 S3 calls per subject, not ~9: one Results/ prefix listing (Delimiter='/')
covers every task-fMRI run in one shot; one existence check on
T1w/Diffusion/data.nii.gz covers DTI. Threaded for speed -- S3 handles
concurrency fine (unlike the bioRxiv/Cloudflare rate-limiting encountered
earlier in this project).
"""

from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = PROJECT_ROOT / "data" / "hcp_full_discovery_scan.json"
OUT_CSV = PROJECT_ROOT / "data" / "hcp_full_discovery_scan.csv"

BUCKET = "hcp-openaccess"
PROFILE = "hcp"
REGION = "us-east-1"

# Base task name a raw folder name reduces to, stripping run/phase-encoding
# suffixes (_LR, _RL, _AP, _PA, _7T, run numbers) -- e.g.
# "tfMRI_MOTOR_LR" -> "MOTOR", "tfMRI_MOVIE1_7T_AP" -> "MOVIE".
TASK_PATTERN = re.compile(r"^tfMRI_([A-Za-z]+?)\d*(?:_7T)?(?:_(?:LR|RL|AP|PA))?$")
REST_PATTERN = re.compile(r"^rfMRI_REST\d*(?:_7T)?(?:_(?:LR|RL|AP|PA))?$")

TASK_COLUMNS = ["MOTOR", "WM", "GAMBLING", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION", "MOVIE", "RETBAR", "RETCCW", "RETCW", "RETCON", "RETEXP"]


def get_client():
    session = boto3.Session(profile_name=PROFILE)
    return session.client("s3", region_name=REGION)


def scan_one_subject(sid: str) -> dict:
    s3 = get_client()
    row = {"subject_id": sid, "has_dti": False, "has_rest": False}
    for col in TASK_COLUMNS:
        row[f"has_{col.lower()}"] = False

    dti = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"HCP_1200/{sid}/T1w/Diffusion/data.nii.gz", MaxKeys=1)
    row["has_dti"] = dti.get("KeyCount", 0) > 0

    results = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"HCP_1200/{sid}/MNINonLinear/Results/", Delimiter="/", MaxKeys=200)
    found_tasks = set()
    found_rest = False
    for p in results.get("CommonPrefixes", []):
        name = p["Prefix"].rstrip("/").split("/")[-1]
        m = TASK_PATTERN.match(name)
        if m:
            found_tasks.add(m.group(1).upper())
        elif REST_PATTERN.match(name):
            found_rest = True

    for col in TASK_COLUMNS:
        row[f"has_{col.lower()}"] = col in found_tasks
    row["has_rest"] = found_rest
    row["n_task_types"] = len(found_tasks)
    return row


def main() -> None:
    subjects = json.loads(Path("/tmp/all_hcp_subjects.json").read_text())
    print(f"Scanning {len(subjects)} subjects (2 S3 calls each, threaded)...", flush=True)

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(scan_one_subject, sid): sid for sid in subjects}
        for i, future in enumerate(as_completed(futures)):
            rows.append(future.result())
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(subjects)} scanned, elapsed={time.time() - t0:.0f}s", flush=True)

    rows.sort(key=lambda r: r["subject_id"])
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    fieldnames = ["subject_id", "has_dti", "has_rest", "n_task_types"] + [f"has_{c.lower()}" for c in TASK_COLUMNS]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {n} subjects scanned.", flush=True)
    print(f"has_dti:  {sum(r['has_dti'] for r in rows)}", flush=True)
    print(f"has_rest: {sum(r['has_rest'] for r in rows)}", flush=True)
    for col in TASK_COLUMNS:
        print(f"has_{col.lower():12s}: {sum(r[f'has_{col.lower()}'] for r in rows)}", flush=True)
    print(f"\nWrote {OUT_JSON} and {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
