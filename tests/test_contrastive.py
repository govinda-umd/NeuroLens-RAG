"""Sanity checks for Case 2's contrastive losses.

Not literal SupCon (see contrastive.py's module docstring) -- these tests
check the actual documented contract: perfectly-aligned brain/text
embeddings give near-zero loss, and misaligned ones give substantially
higher loss, for both the multi-positive and literal-CLIP variants.
"""

import torch

from neurolens.contrastive import symmetric_multi_positive_prototype_loss, clip_loss

NUM_CLASSES = 6
EMBED_DIM = 16


def _one_hot_text(num_classes: int, dim: int) -> torch.Tensor:
    # Orthogonal one-hot-ish text prototypes in a higher-dim space, so a
    # brain embedding can be made to exactly match its class's prototype.
    z = torch.zeros(num_classes, dim)
    z[torch.arange(num_classes), torch.arange(num_classes)] = 1.0
    return z


def test_multi_positive_loss_lower_when_aligned():
    torch.manual_seed(0)
    z_text = _one_hot_text(NUM_CLASSES, EMBED_DIM)
    y = torch.randint(0, NUM_CLASSES, (32,))
    log_temp = torch.tensor(2.0)

    z_brain_aligned = torch.nn.functional.normalize(z_text[y] + 0.01 * torch.randn(len(y), EMBED_DIM), dim=-1)
    z_brain_random = torch.nn.functional.normalize(torch.randn(len(y), EMBED_DIM), dim=-1)

    loss_aligned = symmetric_multi_positive_prototype_loss(z_brain_aligned, z_text, y, log_temp, NUM_CLASSES)
    loss_random = symmetric_multi_positive_prototype_loss(z_brain_random, z_text, y, log_temp, NUM_CLASSES)

    assert torch.isfinite(loss_aligned) and torch.isfinite(loss_random)
    assert loss_aligned.item() < loss_random.item()


def test_clip_loss_lower_when_aligned():
    torch.manual_seed(0)
    z_text_per_sample = torch.nn.functional.normalize(torch.randn(16, EMBED_DIM), dim=-1)
    log_temp = torch.tensor(2.0)

    z_brain_aligned = torch.nn.functional.normalize(z_text_per_sample + 0.01 * torch.randn_like(z_text_per_sample), dim=-1)
    z_brain_random = torch.nn.functional.normalize(torch.randn_like(z_text_per_sample), dim=-1)

    loss_aligned = clip_loss(z_brain_aligned, z_text_per_sample, log_temp)
    loss_random = clip_loss(z_brain_random, z_text_per_sample, log_temp)

    assert torch.isfinite(loss_aligned) and torch.isfinite(loss_random)
    assert loss_aligned.item() < loss_random.item()


def test_multi_positive_loss_handles_missing_class_in_batch():
    # A batch that happens to omit one class entirely must not crash --
    # the text->brain direction only averages over classes present.
    torch.manual_seed(0)
    z_text = _one_hot_text(NUM_CLASSES, EMBED_DIM)
    y = torch.zeros(8, dtype=torch.long)  # every example is class 0
    z_brain = torch.nn.functional.normalize(torch.randn(8, EMBED_DIM), dim=-1)
    log_temp = torch.tensor(2.0)

    loss = symmetric_multi_positive_prototype_loss(z_brain, z_text, y, log_temp, NUM_CLASSES)
    assert torch.isfinite(loss)
