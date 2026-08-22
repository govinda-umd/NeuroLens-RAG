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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def text_to_brain_retrieval_metrics(
    brain_embeddings: np.ndarray,
    brain_labels: np.ndarray,
    text_prototypes: np.ndarray,
    top_k_values: list[int] = (5, 10, 20, 50),
) -> dict[int, dict[int, dict[str, float]]]:
    """For each condition's text prototype, retrieve the top-k most similar
    brain windows and compute precision@k, recall@k, and F1@k at each k in
    `top_k_values`.

    Precision-only is misleading here: classes have very different window
    counts (baseline alone is a large fraction of the test set), so
    precision@k can sit at 1.00 while recall@k is tiny simply because k is
    small relative to the true class size — precision never reveals that on
    its own. Recall@k = (# correct-class windows in top k) / (total
    correct-class windows in the test set); F1@k is their harmonic mean.

    Returns {class_idx: {k: {"precision": ..., "recall": ..., "f1": ...}}}."""
    metrics: dict[int, dict[int, dict[str, float]]] = {}
    for class_idx in range(text_prototypes.shape[0]):
        scores = brain_embeddings @ text_prototypes[class_idx]
        ranked = np.argsort(scores)[::-1]
        n_relevant = int((brain_labels == class_idx).sum())
        metrics[class_idx] = {}
        for k in top_k_values:
            top_indices = ranked[:k]
            n_correct = int((brain_labels[top_indices] == class_idx).sum())
            precision = n_correct / k
            recall = (n_correct / n_relevant) if n_relevant > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            metrics[class_idx][k] = {"precision": precision, "recall": recall, "f1": f1, "n_relevant": n_relevant}
    return metrics


def text_to_brain_retrieval_precision(
    brain_embeddings: np.ndarray,
    brain_labels: np.ndarray,
    text_prototypes: np.ndarray,
    top_k_values: list[int] = (5, 10, 20, 50),
) -> dict[int, dict[int, float]]:
    """Precision-only view, kept for backward compatibility with existing
    notebooks/results — prefer `text_to_brain_retrieval_metrics` for new
    code, since precision alone can't reveal a recall problem."""
    full = text_to_brain_retrieval_metrics(brain_embeddings, brain_labels, text_prototypes, top_k_values)
    return {class_idx: {k: v["precision"] for k, v in per_k.items()} for class_idx, per_k in full.items()}


# --- Case 2 v2: symmetric SupCon-style multi-positive loss (Khosla et al. 2020) ---
#
# The original ContrastiveModel/train_contrastive above run brain->text only,
# since a fixed 6-item text vocabulary has no unique per-example target for
# a text->brain direction the way CLIP's naturally-paired data does. But a
# text->brain direction IS still well-posed if you don't insist on a unique
# target per prototype: for text prototype c, every brain window in the
# batch with true label c is a valid positive (Khosla et al.'s multi-positive
# SupCon "L_out" formulation), everything else a negative. This function
# adds that second direction on top of the *same*, still-frozen-MiniLM
# ContrastiveModel — no architecture change, only a richer loss. Kept
# alongside (not replacing) the original asymmetric loss so both can be
# trained and compared directly.


def supcon_text_to_brain_loss(logits_text_to_brain: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """logits_text_to_brain: [num_classes, B] = z_text @ z_brain.T * tau.
    For each text prototype row c, every brain example with true label c
    is a positive. Log-softmax is taken over the full row (all B brain
    examples as candidates), then averaged only over that row's positive
    columns, and finally averaged across classes present in the batch."""
    log_probs = nn.functional.log_softmax(logits_text_to_brain, dim=1)  # softmax over brain examples
    per_class_losses = []
    for c in range(num_classes):
        mask = y == c
        if mask.sum() == 0:
            continue  # class absent from this batch - nothing to average
        per_class_losses.append(-log_probs[c, mask].mean())
    return torch.stack(per_class_losses).mean()


def symmetric_supcon_loss(
    z_brain: torch.Tensor, z_text: torch.Tensor, y: torch.Tensor, log_temperature: torch.Tensor, num_classes: int
) -> torch.Tensor:
    tau = log_temperature.exp().clamp(max=100)
    logits_b2t = z_brain @ z_text.T * tau  # [B, num_classes]
    loss_b2t = nn.functional.cross_entropy(logits_b2t, y)
    loss_t2b = supcon_text_to_brain_loss(logits_b2t.T, y, num_classes)
    return (loss_b2t + loss_t2b) / 2


def run_epoch_supcon(
    model: ContrastiveModel, loader: DataLoader, device: torch.device, num_classes: int, optimizer=None
) -> dict:
    """Same bookkeeping as run_epoch, but trains/evaluates with
    symmetric_supcon_loss instead of the one-directional cross-entropy.
    Classification metrics still come from the brain->text direction
    (argmax over z_brain @ z_text.T), since that's the direction with an
    actual per-example prediction to score."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_batches = 0.0, 0
    all_y_true, all_y_pred = [], []

    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            z_brain, z_text, logits = model(x)
            loss = symmetric_supcon_loss(z_brain, z_text, y, model.log_temperature, num_classes)

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
    metrics = classification_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_contrastive_supcon(
    model: ContrastiveModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    experiment_name: str = "contrastive_supcon",
    checkpoint_dir=None,
) -> dict:
    """Mirrors train_contrastive exactly, swapping in the symmetric
    multi-positive SupCon loss. Kept as a fully separate function (not a
    flag on train_contrastive) so both the original asymmetric model and
    this symmetric one can be trained independently and compared."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch_supcon(model, train_loader, device, num_classes, optimizer=optimizer)
        val_metrics = run_epoch_supcon(model, val_loader, device, num_classes, optimizer=None)
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
    test_metrics = run_epoch_supcon(model, test_loader, device, num_classes, optimizer=None)

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    return {
        "experiment_name": experiment_name,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics,
    }


# --- Case 2 v3: forecasting block on top of the frozen contrastive backbone ---


def extract_frozen_features_for_forecasting(
    contrastive_model: ContrastiveModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the already-trained brain_backbone.forward_features() (frozen,
    no_grad) over every forecasting window. Returns (features [N, backbone_dim],
    targets [N, n_rois])."""
    contrastive_model.eval()
    all_features, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            features = contrastive_model.brain_backbone.forward_features(x)
            all_features.append(features.cpu().numpy())
            all_targets.append(batch["target"].numpy())
    return np.concatenate(all_features), np.concatenate(all_targets)


def train_forecast_head(
    contrastive_model: ContrastiveModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
) -> dict:
    """A linear probe (ridge regression) on the frozen contrastive brain
    encoder's pooled features, predicting future ROI activity — the
    "forecasting lego-block on top of" the representation, per the design
    plan. Framed explicitly as a linear probe (like the CAV work), not a
    trained neural head, since the point is to test what the *already
    learned* representation contains, not to fit a new deep model."""
    train_features, train_targets = extract_frozen_features_for_forecasting(contrastive_model, train_loader, device)
    val_features, val_targets = extract_frozen_features_for_forecasting(contrastive_model, val_loader, device)
    test_features, test_targets = extract_frozen_features_for_forecasting(contrastive_model, test_loader, device)

    if len(train_targets) == 0 or len(test_targets) == 0:
        return {
            "n_train": len(train_targets),
            "n_val": len(val_targets),
            "n_test": len(test_targets),
            "val_mse": None,
            "test_mse": None,
            "test_mae": None,
            "test_r2": None,
        }

    probe = Ridge(alpha=alpha)
    probe.fit(train_features, train_targets)

    val_pred = probe.predict(val_features)
    test_pred = probe.predict(test_features)

    return {
        "n_train": len(train_targets),
        "n_val": len(val_targets),
        "n_test": len(test_targets),
        "val_mse": float(mean_squared_error(val_targets, val_pred)),
        "test_mse": float(mean_squared_error(test_targets, test_pred)),
        "test_mae": float(mean_absolute_error(test_targets, test_pred)),
        "test_r2": float(r2_score(test_targets, test_pred, multioutput="uniform_average")),
    }
