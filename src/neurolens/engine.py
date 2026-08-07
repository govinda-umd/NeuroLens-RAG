"""Training/evaluation loop shared across the four decoding experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import classification_metrics, empty_hrf_metrics, hrf_regression_metrics


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor | None = None,
    lambda_hrf: float = 0.1,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates.

    Whether the auxiliary HRF loss/metrics apply is driven entirely by
    whether `model` returns a non-None hrf_pred, not by a separate flag —
    a classification-only model architecturally cannot produce one.
    """
    is_train = optimizer is not None
    model.train(is_train)
    ce_weight = class_weights.to(device) if class_weights is not None else None

    total_loss = total_cls_loss = total_hrf_loss = 0.0
    n_batches = 0
    all_y_true, all_y_pred = [], []
    all_hrf_true, all_hrf_pred = [], []

    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            y_hrf = batch["y_hrf"].to(device)

            logits, hrf_pred = model(x)
            cls_loss = nn.functional.cross_entropy(logits, y, weight=ce_weight)

            if hrf_pred is not None:
                hrf_loss = nn.functional.mse_loss(hrf_pred, y_hrf)
                loss = cls_loss + lambda_hrf * hrf_loss
                total_hrf_loss += hrf_loss.item()
                all_hrf_true.append(y_hrf.detach().cpu().numpy())
                all_hrf_pred.append(hrf_pred.detach().cpu().numpy())
            else:
                loss = cls_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            n_batches += 1
            all_y_true.append(y.detach().cpu().numpy())
            all_y_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    metrics = classification_metrics(y_true, y_pred, num_classes)

    has_hrf = len(all_hrf_true) > 0
    if has_hrf:
        hrf_true = np.concatenate(all_hrf_true)
        hrf_pred_all = np.concatenate(all_hrf_pred)
        metrics.update({f"hrf_{k}": v for k, v in hrf_regression_metrics(hrf_true, hrf_pred_all).items()})
    else:
        metrics.update({f"hrf_{k}": v for k, v in empty_hrf_metrics().items()})

    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["cls_loss"] = total_cls_loss / max(n_batches, 1)
    metrics["hrf_loss"] = (total_hrf_loss / max(n_batches, 1)) if has_hrf else None
    return metrics


@dataclass
class TrainConfig:
    experiment_name: str
    num_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_hrf: float = 0.1
    checkpoint_dir: Path | None = None


def train_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    class_weights: torch.Tensor,
    config: TrainConfig,
) -> dict:
    """AdamW + cosine LR annealing over `config.num_epochs`; checkpoints the
    epoch with the best validation macro F1 and reports its test metrics."""
    device = get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    history = []
    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, num_classes, class_weights, config.lambda_hrf, optimizer=optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, device, num_classes, class_weights, config.lambda_hrf, optimizer=None
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history.append({"epoch": epoch, "lr": current_lr, "train": train_metrics, "val": val_metrics})

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{config.experiment_name}] epoch {epoch}/{config.num_epochs} "
            f"lr={current_lr:.2e} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = run_epoch(
        model, test_loader, device, num_classes, class_weights, config.lambda_hrf, optimizer=None
    )

    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.checkpoint_dir / "best.pt")

    return {
        "experiment_name": config.experiment_name,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
    }


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """One row per window: decoded state at that timepoint, not just epoch-aggregate metrics.

    This is what makes per-timepoint RSN interpretation and RAG querying
    possible — each row is enough on its own to run an interpretability
    method and build a decoded-state-to-text query for that timepoint.
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits, hrf_pred = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_y = probs.argmax(dim=1)

            batch_size = x.shape[0]
            probs_np = probs.cpu().numpy()
            hrf_true_np = batch["y_hrf"].numpy()
            hrf_pred_np = hrf_pred.cpu().numpy() if hrf_pred is not None else None

            for i in range(batch_size):
                rows.append(
                    {
                        "subject_id": batch["subject_id"][i],
                        "task": batch["task"][i],
                        "run": batch["run"][i],
                        "target_volume": int(batch["target_volume"][i]),
                        "true_y": int(batch["y"][i]),
                        "pred_y": int(pred_y[i]),
                        "pred_proba": probs_np[i].tolist(),
                        "confidence": float(probs_np[i, pred_y[i]]),
                        "true_y_hrf": hrf_true_np[i].tolist(),
                        "pred_y_hrf": None if hrf_pred_np is None else hrf_pred_np[i].tolist(),
                    }
                )
    return pd.DataFrame(rows)
