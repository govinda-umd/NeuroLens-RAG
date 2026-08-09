"""Windowed Dataset/DataLoader construction for MOTOR-task brain decoding.

Consumes the processed per-run bundles produced by ``02_data.ipynb``
(``data/processed/hcp_ya_s1200/runs/sub-<id>/tfMRI_<TASK>_<RUN>/``) and turns
them into causal, overlapping temporal windows for sequence-to-one decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Subject-level split over the 100 subjects with completed MOTOR bundles on
# disk (v3 scale-up: 20 -> 100). Splits must never mix windows from the
# same subject across train/val/test/hyperparam (see
# docs/project-handoff-summary.md §14). Fixed seed-42 shuffle over the
# sorted 100-subject list, then sliced 10/65/13/12 (hyperparam/train/val/
# test) - see docs/population-level-evaluation-plan.md §3.1 for why the
# hyperparam subset exists and must never be used for headline results.
MOTOR_HYPERPARAM_SUBJECTS = [
    "108323", "108222", "118225", "102008", "113619",
    "110613", "100307", "114318", "102816", "115724",
]
MOTOR_TRAIN_SUBJECTS = [
    "114823", "102109", "111514", "111716", "114621", "109123", "110007",
    "118528", "115219", "107422", "106016", "103818", "106521", "119025",
    "116221", "110411", "116726", "105115", "117728", "106824", "101915",
    "109830", "112516", "113316", "114924", "108828", "119732", "111211",
    "117122", "102513", "107321", "104416", "108020", "108121", "103212",
    "113821", "112920", "112819", "101410", "107018", "119833", "109325",
    "100408", "111009", "103010", "107725", "112314", "114116", "104012",
    "113215", "104820", "101107", "101309", "113922", "116524", "103414",
    "115825", "108525", "118124", "103515", "100206", "118932", "112112",
    "118730", "111312",
]
MOTOR_VAL_SUBJECTS = [
    "118023", "105014", "114419", "117021", "115320", "113417", "105923",
    "105216", "117930", "119126", "101006", "111413", "115017",
]
MOTOR_TEST_SUBJECTS = [
    "102311", "114217", "117324", "102614", "103111", "105620", "106319",
    "107220", "118831", "100610", "102715", "116423",
]

# Superseded splits, kept for reference/reproducibility of earlier results:
# v2 (20 subjects, 14/3/3): see docs/ml-design-report.md and git history
# prior to the v3 (100-subject) scale-up commit.
MOTOR_RUNS = ["LR", "RL"]

DEFAULT_WINDOW_LENGTH = 32
DEFAULT_STRIDE = 2
DEFAULT_BATCH_SIZE = 64


@dataclass
class RunArrays:
    subject_id: str
    run: str
    X: np.ndarray  # [T, n_rois] float32
    y: np.ndarray  # [T] int64
    y_hrf: np.ndarray  # [T, n_conditions] float32
    valid_mask: np.ndarray  # [T] bool
    metadata: dict


@dataclass(frozen=True)
class WindowRef:
    subject_id: str
    run: str
    start: int
    end: int  # exclusive; target volume index is end - 1


def find_run_dir(processed_root: Path, subject_id: str, task: str, run: str) -> Path:
    return processed_root / f"sub-{subject_id}" / f"tfMRI_{task}_{run}"


def load_run_arrays(processed_root: Path, subject_id: str, task: str, run: str) -> RunArrays:
    run_dir = find_run_dir(processed_root, subject_id, task, run)
    X = np.load(run_dir / "X.npy")
    y = np.load(run_dir / "y.npy")
    y_hrf = np.load(run_dir / "y_hrf.npy")
    valid_mask = np.load(run_dir / "valid_mask.npy")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    return RunArrays(subject_id, run, X, y, y_hrf, valid_mask, metadata)


def load_all_runs(
    processed_root: Path,
    subjects: Sequence[str],
    task: str = "MOTOR",
    runs: Sequence[str] = MOTOR_RUNS,
) -> dict[tuple[str, str], RunArrays]:
    out: dict[tuple[str, str], RunArrays] = {}
    for subject_id in subjects:
        for run in runs:
            run_dir = find_run_dir(processed_root, subject_id, task, run)
            if not run_dir.exists():
                continue
            out[(subject_id, run)] = load_run_arrays(processed_root, subject_id, task, run)
    return out


def build_window_index(
    runs: dict[tuple[str, str], RunArrays],
    window_length: int = DEFAULT_WINDOW_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> list[WindowRef]:
    """Causal sequence-to-one windows: X[start:end] -> y[end - 1].

    A window is dropped if any frame it covers is flagged invalid.
    """
    windows: list[WindowRef] = []
    for (subject_id, run), arr in runs.items():
        n_timepoints = arr.X.shape[0]
        for end in range(window_length, n_timepoints + 1, stride):
            start = end - window_length
            if not bool(arr.valid_mask[start:end].all()):
                continue
            windows.append(WindowRef(subject_id, run, start, end))
    return windows


@dataclass(frozen=True)
class ForecastWindowRef:
    subject_id: str
    run: str
    start: int
    end: int  # input window is X[start:end]; "current" volume is end - 1
    target_idx: int  # forecast target volume: (end - 1) + horizon
    horizon: int


def build_forecast_window_index(
    runs: dict[tuple[str, str], RunArrays],
    window_length: int = DEFAULT_WINDOW_LENGTH,
    stride: int = DEFAULT_STRIDE,
    horizon: int = 1,
) -> list[ForecastWindowRef]:
    """Causal forecasting windows: X[start:end] -> X[target_idx], where
    target_idx = (end - 1) + horizon.

    Leak-safety (Case 2 v3, see docs/case2-3-design-plan.md): the MOTOR
    task is block-designed (~12s condition blocks). A window is only kept
    if the forecast target's condition label matches the window's own
    "current" label — otherwise a large horizon could silently forecast
    into an *adjacent* movement condition's block rather than a genuine
    continuation of the same condition, which would make the forecasting
    task secretly leak label-boundary information rather than testing
    pure signal continuation.
    """
    windows: list[ForecastWindowRef] = []
    for (subject_id, run), arr in runs.items():
        n_timepoints = arr.X.shape[0]
        for end in range(window_length, n_timepoints + 1, stride):
            start = end - window_length
            current_idx = end - 1
            target_idx = current_idx + horizon
            if target_idx >= n_timepoints:
                continue
            if not bool(arr.valid_mask[start : target_idx + 1].all()):
                continue
            if arr.y[target_idx] != arr.y[current_idx]:
                continue  # would leak into an adjacent condition block
            windows.append(ForecastWindowRef(subject_id, run, start, end, target_idx, horizon))
    return windows


class ForecastWindowDataset(Dataset):
    def __init__(self, runs: dict[tuple[str, str], RunArrays], windows: list[ForecastWindowRef]):
        self.runs = runs
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        arr = self.runs[(w.subject_id, w.run)]
        x = arr.X[w.start : w.end].astype(np.float32)
        target = arr.X[w.target_idx].astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "target": torch.from_numpy(target),
            "y": torch.tensor(int(arr.y[w.end - 1]), dtype=torch.long),  # "current" label, for analysis only
            "subject_id": w.subject_id,
            "run": w.run,
            "horizon": w.horizon,
        }


def make_forecast_dataloaders(
    processed_root: Path,
    task: str = "MOTOR",
    window_length: int = DEFAULT_WINDOW_LENGTH,
    stride: int = DEFAULT_STRIDE,
    horizon: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_subjects: Sequence[str] = MOTOR_TRAIN_SUBJECTS,
    val_subjects: Sequence[str] = MOTOR_VAL_SUBJECTS,
    test_subjects: Sequence[str] = MOTOR_TEST_SUBJECTS,
    runs: Sequence[str] = MOTOR_RUNS,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    all_subjects = [*train_subjects, *val_subjects, *test_subjects]
    run_arrays = load_all_runs(processed_root, all_subjects, task=task, runs=runs)
    all_windows = build_forecast_window_index(
        run_arrays, window_length=window_length, stride=stride, horizon=horizon
    )

    train_windows = [w for w in all_windows if w.subject_id in set(train_subjects)]
    val_windows = [w for w in all_windows if w.subject_id in set(val_subjects)]
    test_windows = [w for w in all_windows if w.subject_id in set(test_subjects)]

    # shuffle=True on an empty dataset crashes RandomSampler - at large
    # horizons the leak-safety filter can legitimately leave a split empty
    # (see build_forecast_window_index), so guard rather than assume.
    train_loader = DataLoader(
        ForecastWindowDataset(run_arrays, train_windows),
        batch_size=batch_size,
        shuffle=len(train_windows) > 0,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        ForecastWindowDataset(run_arrays, val_windows), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        ForecastWindowDataset(run_arrays, test_windows), batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    info = {
        "horizon": horizon,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "n_test_windows": len(test_windows),
    }
    return train_loader, val_loader, test_loader, info


def filter_windows_by_subjects(
    windows: list[WindowRef], subjects: Sequence[str]
) -> list[WindowRef]:
    subject_set = set(subjects)
    return [w for w in windows if w.subject_id in subject_set]


def compute_class_weights(
    runs: dict[tuple[str, str], RunArrays],
    windows: list[WindowRef],
    num_classes: int,
) -> torch.Tensor:
    """weight[c] = n_samples / (num_classes * samples_in_class_c), from `windows` only."""
    labels = np.array([runs[(w.subject_id, w.run)].y[w.end - 1] for w in windows])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # guard against unseen classes in a split
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class MotorWindowDataset(Dataset):
    def __init__(
        self,
        runs: dict[tuple[str, str], RunArrays],
        windows: list[WindowRef],
        task: str = "MOTOR",
    ):
        self.runs = runs
        self.windows = windows
        self.task = task

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        arr = self.runs[(w.subject_id, w.run)]
        target = w.end - 1
        x = arr.X[w.start : w.end].astype(np.float32)
        y_hrf = arr.y_hrf[target].astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(int(arr.y[target]), dtype=torch.long),
            "y_hrf": torch.from_numpy(y_hrf),
            "subject_id": w.subject_id,
            "task": self.task,
            "run": w.run,
            "target_volume": target,
        }


def make_dataloaders(
    processed_root: Path,
    task: str = "MOTOR",
    window_length: int = DEFAULT_WINDOW_LENGTH,
    stride: int = DEFAULT_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_subjects: Sequence[str] = MOTOR_TRAIN_SUBJECTS,
    val_subjects: Sequence[str] = MOTOR_VAL_SUBJECTS,
    test_subjects: Sequence[str] = MOTOR_TEST_SUBJECTS,
    runs: Sequence[str] = MOTOR_RUNS,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    all_subjects = [*train_subjects, *val_subjects, *test_subjects]
    run_arrays = load_all_runs(processed_root, all_subjects, task=task, runs=runs)
    all_windows = build_window_index(run_arrays, window_length=window_length, stride=stride)

    train_windows = filter_windows_by_subjects(all_windows, train_subjects)
    val_windows = filter_windows_by_subjects(all_windows, val_subjects)
    test_windows = filter_windows_by_subjects(all_windows, test_subjects)

    sample_meta = next(iter(run_arrays.values())).metadata["target_construction"]
    num_classes = len(sample_meta["class_to_condition"])
    num_conditions = len(sample_meta["condition_names"])

    class_weights = compute_class_weights(run_arrays, train_windows, num_classes)

    train_loader = DataLoader(
        MotorWindowDataset(run_arrays, train_windows, task=task),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        MotorWindowDataset(run_arrays, val_windows, task=task),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        MotorWindowDataset(run_arrays, test_windows, task=task),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    info = {
        "num_classes": num_classes,
        "num_conditions": num_conditions,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "n_test_windows": len(test_windows),
        "class_weights": class_weights,
        "class_to_condition": sample_meta["class_to_condition"],
        "condition_names": sample_meta["condition_names"],
    }
    return train_loader, val_loader, test_loader, info
