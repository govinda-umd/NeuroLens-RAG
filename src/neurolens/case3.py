"""Case 3: self-supervised brain-HRF co-embedding.

Structurally close to Case 2 (`contrastive.py`), with the same window's
HRF regressor vector in place of the condition's text description. The
critical difference is the supervision signal, not the architecture: Case
2's positive pairing comes from the true class label (a brain window is
paired with *its class's* text prototype) — genuinely supervised
contrastive learning. Case 3's positive pairing comes from *temporal
co-occurrence* (a brain window is paired with the HRF value computed from
that exact window) — no class label enters the loss anywhere. This makes
Case 3 self-supervised, not a third instance of Case 1/2's supervision
paradigm, which is a deliberate diversification of the project's
representation-learning coverage, not an oversight.

Because HRF is continuous and per-window (unlike Case 2's six fixed text
prototypes), both sides of the loss have genuine in-batch negatives, so
this uses a literal symmetric contrastive loss (brain->HRF and HRF->brain,
averaged) — the thing Case 2's closed vocabulary structurally couldn't
support (see docs/case2-3-design-plan.md §2.1's note on this).

Case 3 has no text branch, so it cannot support open-vocabulary CAV the
way Case 2 does (docs/project-summary.md §3.8) — there is nothing to embed
a literature phrase into. Concept testing here reuses Case 1's exact
labeled-example-probe machinery (`concepts.py`), applied to a small linear
classifier fit post-hoc on Case 3's frozen self-supervised features — the
readout Case 3 doesn't have by construction, needed both to report a
macro-F1 comparable to Case 1/2 and to give CAV/TCAV a target-class logit
to differentiate at all.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import classification_metrics


class HRFEncoder(nn.Module):
    """Small MLP encoding the 5-dim HRF regressor vector into the shared
    embedding space. Unlike Case 2's frozen-MiniLM text encoder, there is
    no pretrained starting point for HRF — the whole encoder trains
    end-to-end."""

    def __init__(self, hrf_dim: int = 5, hidden_dim: int = 32, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hrf_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, y_hrf: torch.Tensor) -> torch.Tensor:
        z = self.net(y_hrf)
        return nn.functional.normalize(z, dim=-1)


class BrainHRFModel(nn.Module):
    """brain_backbone: a GRUDecoder/TransformerDecoder instance (heads
    unused, only forward_features() is called), exactly as in Case 2."""

    def __init__(self, brain_backbone: nn.Module, backbone_dim: int, embed_dim: int = 64, hrf_dim: int = 5):
        super().__init__()
        self.brain_backbone = brain_backbone
        self.brain_projection = nn.Linear(backbone_dim, embed_dim)
        self.hrf_encoder = HRFEncoder(hrf_dim, hidden_dim=32, embed_dim=embed_dim)
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def encode_brain(self, x: torch.Tensor) -> torch.Tensor:
        features = self.brain_backbone.forward_features(x)
        z = self.brain_projection(features)
        return nn.functional.normalize(z, dim=-1)

    def encode_hrf(self, y_hrf: torch.Tensor) -> torch.Tensor:
        return self.hrf_encoder(y_hrf)

    def forward(self, x: torch.Tensor, y_hrf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_brain(x), self.encode_hrf(y_hrf)


def symmetric_contrastive_loss(z_brain: torch.Tensor, z_hrf: torch.Tensor, log_temperature: torch.Tensor) -> torch.Tensor:
    """Literal symmetric InfoNCE: brain->HRF and HRF->brain, averaged. Only
    possible because both sides are per-example continuous embeddings, not
    a small closed vocabulary (contrast with Case 2, §2.1 of
    case2-3-design-plan.md)."""
    temperature = log_temperature.exp().clamp(max=100)
    logits = z_brain @ z_hrf.T * temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_b2h = nn.functional.cross_entropy(logits, targets)
    loss_h2b = nn.functional.cross_entropy(logits.T, targets)
    return (loss_b2h + loss_h2b) / 2


def run_epoch(model: BrainHRFModel, loader: DataLoader, device: torch.device, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    with torch.enable_grad() if is_train else torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_hrf = batch["y_hrf"].to(device)
            z_brain, z_hrf = model(x, y_hrf)
            loss = symmetric_contrastive_loss(z_brain, z_hrf, model.log_temperature)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return {"loss": total_loss / max(n_batches, 1)}


def train_case3(
    model: BrainHRFModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    experiment_name: str = "case3",
    checkpoint_dir=None,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = []
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{experiment_name}] epoch {epoch}/{num_epochs} "
              f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "best.pt")
    return {"experiment_name": experiment_name, "history": history, "best_val_loss": best_val_loss}


def extract_brain_embeddings_case3(
    model: BrainHRFModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """[N, backbone_dim] pooled features (pre-projection, matching where
    CAV/TCAV operates for every other case) + [N] true labels. Labels are
    never used in training - only fetched here for post-hoc evaluation."""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            features = model.brain_backbone.forward_features(x)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch["y"].numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


class BrainWithPostHocClassifier(nn.Module):
    """Wraps a self-supervised Case 3 brain backbone with a linear
    classifier fit post-hoc on frozen features. This is the readout Case 3
    doesn't have by construction - needed to (a) report a macro-F1 number
    directly comparable to Case 1/2 despite Case 3 never seeing a label
    during training, and (b) give CAV/TCAV a target-class logit to
    differentiate at all, which lets `concepts.py`'s existing
    labeled-example-probe machinery run on Case 3 completely unchanged."""

    def __init__(self, brain_backbone: nn.Module, backbone_dim: int, num_classes: int = 6):
        super().__init__()
        self.brain_backbone = brain_backbone
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.brain_backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        h = self.forward_features(x)
        return self.classifier(h), None


def fit_post_hoc_classifier(
    case3_model: BrainHRFModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 6,
) -> tuple[BrainWithPostHocClassifier, dict]:
    """Fits a linear probe (sklearn LogisticRegression, for a fast, stable
    fit) on Case 3's frozen self-supervised brain features, then copies the
    learned weights into a real nn.Linear so CAV/TCAV's autograd-based
    directional derivative can run through it. Reports macro-F1 on the
    test set - the number directly comparable to Case 1's 92.0% and Case
    2's 91.8%, despite Case 3 never having seen a single label during
    representation training."""
    backbone_dim = case3_model.brain_projection.in_features
    wrapped = BrainWithPostHocClassifier(case3_model.brain_backbone, backbone_dim, num_classes).to(device)

    train_features, train_labels = extract_brain_embeddings_case3(case3_model, train_loader, device)
    test_features, test_labels = extract_brain_embeddings_case3(case3_model, test_loader, device)

    probe = LogisticRegression(max_iter=2000)
    probe.fit(train_features, train_labels)

    with torch.no_grad():
        wrapped.classifier.weight.copy_(torch.tensor(probe.coef_, dtype=torch.float32))
        wrapped.classifier.bias.copy_(torch.tensor(probe.intercept_, dtype=torch.float32))

    test_pred = probe.predict(test_features)
    metrics = classification_metrics(test_labels, test_pred, num_classes)
    return wrapped, metrics
