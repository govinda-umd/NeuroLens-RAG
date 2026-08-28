"""Baseline MLP bootstrap for Case 1 (user-proposed 2026-08-27): does a
non-recurrent, non-attention model on the raw window already separate the
6 motor classes, informing whether Case 1/2/3's sequence-modeled
representations are actually needed for motor classification at all.

Reuses Case 1's exact 30 subject splits (results/case1_bootstrap_100resamples.json)
so the comparison against GRU/Transformer is paired, not confounded by a
different subject composition -- same convention as the Case 2/3 bootstraps.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from neurolens.baseline_mlp import FlattenMLP, MeanPoolMLP
from neurolens.data_setup import make_dataloaders
from neurolens.engine import TrainConfig, train_experiment
from neurolens.model_builder import count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "hcp_ya_s1200" / "runs"
SPLITS_PATH = PROJECT_ROOT / "results" / "case1_bootstrap_100resamples.json"
OUT_PATH = PROJECT_ROOT / "results" / "baseline_mlp_bootstrap_30resamples.json"


def main() -> None:
    splits = json.loads(SPLITS_PATH.read_text())
    results: list[dict] = []
    t_start = time.time()

    for i, split in enumerate(splits):
        train_loader, val_loader, test_loader, info = make_dataloaders(
            PROCESSED_ROOT,
            batch_size=64,
            train_subjects=split["train_subjects"],
            val_subjects=split["val_subjects"],
            test_subjects=split["test_subjects"],
        )

        row: dict = {"resample": i}
        for name, model_fn in [
            ("flatten_mlp", lambda: FlattenMLP(num_classes=info["num_classes"], num_conditions=info["num_conditions"], include_hrf_head=True)),
            ("meanpool_mlp", lambda: MeanPoolMLP(num_classes=info["num_classes"], num_conditions=info["num_conditions"], include_hrf_head=True)),
        ]:
            torch.manual_seed(42)
            model = model_fn()
            if i == 0:
                row[f"{name}_n_params"] = count_parameters(model)
            cfg = TrainConfig(experiment_name=f"{name}_{i}", num_epochs=5, lambda_hrf=0.1)
            result = train_experiment(model, train_loader, val_loader, test_loader, info["num_classes"], info["class_weights"], cfg)
            row[f"{name}_test_macro_f1"] = result["test_metrics"]["macro_f1"]

        results.append(row)
        elapsed = time.time() - t_start
        print(
            f"[{i + 1}/{len(splits)}] flatten={row['flatten_mlp_test_macro_f1']:.4f} "
            f"meanpool={row['meanpool_mlp_test_macro_f1']:.4f} elapsed={elapsed / 60:.1f}min",
            flush=True,
        )
        OUT_PATH.write_text(json.dumps(results, indent=2))

    flatten_scores = [r["flatten_mlp_test_macro_f1"] for r in results]
    meanpool_scores = [r["meanpool_mlp_test_macro_f1"] for r in results]
    print(f"flatten_mlp mean={sum(flatten_scores)/len(flatten_scores):.4f}", flush=True)
    print(f"meanpool_mlp mean={sum(meanpool_scores)/len(meanpool_scores):.4f}", flush=True)
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
