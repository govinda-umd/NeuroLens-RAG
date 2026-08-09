"""Case 2: contrastive multimodal representation learning (ROI timeseries <-> condition).

CLIP-style (Radford et al. 2021) alignment between a brain encoder and a
text encoder of the condition description, EXCEPT the text side is a small,
fixed, closed vocabulary (6 MOTOR conditions) rather than an open per-example
caption set — so the loss compares against all 6 known prototypes every
step (supervised contrastive learning against semantically-initialized
class prototypes), not literal in-batch-negative InfoNCE. See
docs/case2-3-design-plan.md §2 for the full design rationale.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import classification_metrics

CONDITION_DESCRIPTIONS: dict[str, str] = {
    "baseline": "baseline resting state, no movement",
    "left_hand": "left hand movement",
    "right_hand": "right hand movement",
    "left_foot": "left foot movement",
    "right_foot": "right foot movement",
    "tongue": "tongue movement",
}


def encode_condition_prototypes(
    embedding_model: SentenceTransformer, class_to_condition: dict[str, str]
) -> np.ndarray:
    """[num_classes, text_dim], ordered by class index. Only ever computed
    once per embedding model — there are 6 fixed strings, not one per
    example."""
    num_classes = len(class_to_condition)
    descriptions = [CONDITION_DESCRIPTIONS[class_to_condition[str(c)]] for c in range(num_classes)]
    return embedding_model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True).astype(
        np.float32
    )


class TextPrototypeEncoder(nn.Module):
    """Frozen precomputed text embeddings + a small trainable projection
    into the shared embedding space. Sample-efficient by construction:
    with only 6 examples there's no signal to learn a text encoder from,
    so only the projection is trained."""

    def __init__(self, text_embeddings: np.ndarray, embed_dim: int):
        super().__init__()
        self.register_buffer("text_embeddings", torch.tensor(text_embeddings, dtype=torch.float32))
        self.projection = nn.Linear(text_embeddings.shape[1], embed_dim)

    def forward(self) -> torch.Tensor:
        z = self.projection(self.text_embeddings)
        return nn.functional.normalize(z, dim=-1)


class ContrastiveModel(nn.Module):
    """brain_backbone: a GRUDecoder/TransformerDecoder instance (its
    classifier/HRF heads are unused — only forward_features() is called)."""

    def __init__(self, brain_backbone: nn.Module, backbone_dim: int, text_encoder: TextPrototypeEncoder, embed_dim: int):
        super().__init__()
        self.brain_backbone = brain_backbone
        self.brain_projection = nn.Linear(backbone_dim, embed_dim)
        self.text_encoder = text_encoder
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))  # CLIP init

    def encode_brain(self, x: torch.Tensor) -> torch.Tensor:
        features = self.brain_backbone.forward_features(x)
        z = self.brain_projection(features)
        return nn.functional.normalize(z, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_brain = self.encode_brain(x)
        z_text = self.text_encoder()
        logits = z_brain @ z_text.T * self.log_temperature.exp().clamp(max=100)
        return z_brain, z_text, logits


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: ContrastiveModel, loader: DataLoader, device: torch.device, optimizer=None
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    all_y_true, all_y_pred = [], []

    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            _, _, logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_y_true.append(y.detach().cpu().numpy())
            all_y_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    num_classes = int(max(y_true.max(), y_pred.max())) + 1
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_contrastive(
    model: ContrastiveModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    experiment_name: str = "contrastive",
    checkpoint_dir=None,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{experiment_name}] epoch {epoch}/{num_epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, device, optimizer=None)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    return {
        "experiment_name": experiment_name,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
    }


def extract_brain_embeddings(
    model: ContrastiveModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """[N, embed_dim] brain embeddings + [N] true labels, for retrieval/visualization."""
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            z = model.encode_brain(x)
            all_z.append(z.cpu().numpy())
            all_labels.append(batch["y"].numpy())
    return np.concatenate(all_z), np.concatenate(all_labels)


def text_to_brain_retrieval_precision(
    brain_embeddings: np.ndarray,
    brain_labels: np.ndarray,
    text_prototypes: np.ndarray,
    top_k_values: list[int] = (5, 10, 20, 50),
) -> dict[int, dict[int, float]]:
    """For each condition's text prototype, retrieve the top-k most similar
    brain windows and compute precision@k (fraction whose true label
    matches) at each k in `top_k_values` — a single k hides where precision
    actually starts to degrade, so report the curve, not one point. The
    non-degenerate retrieval direction, since brain->text is just the
    classification argmax with only 6 possible targets.

    Returns {class_idx: {k: precision}}."""
    precisions: dict[int, dict[int, float]] = {}
    for class_idx in range(text_prototypes.shape[0]):
        scores = brain_embeddings @ text_prototypes[class_idx]
        ranked = np.argsort(scores)[::-1]
        precisions[class_idx] = {}
        for k in top_k_values:
            top_indices = ranked[:k]
            precisions[class_idx][k] = float((brain_labels[top_indices] == class_idx).mean())
    return precisions
