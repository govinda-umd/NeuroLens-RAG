"""Shape and interface contracts for every backbone used across Cases 1-3.

Every model here must expose forward_features(x) -> [B, 128] and
forward(x) -> (logits, hrf_pred | None), since concepts.py's CAV/TCAV code
and contrastive.py/case3.py both depend on that exact interface unchanged
across GRU, Transformer, and the two baseline MLPs.
"""

import torch

from neurolens.model_builder import GRUDecoder, TransformerDecoder
from neurolens.baseline_mlp import FlattenMLP, MeanPoolMLP

BATCH_SIZE = 4
WINDOW_LENGTH = 32
N_ROIS = 300
NUM_CLASSES = 6
NUM_CONDITIONS = 5
BACKBONE_DIM = 128


def _make_input():
    return torch.randn(BATCH_SIZE, WINDOW_LENGTH, N_ROIS)


def _check_backbone(model):
    x = _make_input()
    features = model.forward_features(x)
    assert features.shape == (BATCH_SIZE, BACKBONE_DIM)
    assert torch.isfinite(features).all()

    logits, hrf_pred = model(x)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert torch.isfinite(logits).all()
    if model.hrf_head is not None:
        assert hrf_pred.shape == (BATCH_SIZE, NUM_CONDITIONS)
    else:
        assert hrf_pred is None


def test_gru_decoder_shapes():
    model = GRUDecoder(num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=True)
    _check_backbone(model)


def test_gru_decoder_defaults_to_two_layers():
    # Regression guard: GRU was deliberately changed from 1 to 2 layers to
    # bring its parameter count within 1.15x of Transformer's (was 1.83x).
    model = GRUDecoder(num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=False)
    assert model.gru.num_layers == 2


def test_transformer_decoder_shapes():
    model = TransformerDecoder(num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=True)
    _check_backbone(model)


def test_transformer_causal_mask_changes_output():
    # A causal mask should make position i blind to inputs after it; a
    # bidirectional model's output for a given final-token readout can
    # still change if earlier context changes -- so this checks the more
    # basic contract: causal=True actually runs (no shape/dtype break) and
    # produces a different result than causal=False on the same input,
    # confirming the mask is actually wired into the forward pass.
    x = _make_input()
    causal = TransformerDecoder(num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=False, causal=True)
    noncausal = TransformerDecoder(num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=False, causal=False)
    noncausal.load_state_dict(causal.state_dict())
    logits_causal, _ = causal(x)
    logits_noncausal, _ = noncausal(x)
    assert not torch.allclose(logits_causal, logits_noncausal)


def test_flatten_mlp_shapes():
    model = FlattenMLP(input_size=N_ROIS, window_length=WINDOW_LENGTH, num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=True)
    _check_backbone(model)


def test_meanpool_mlp_shapes():
    model = MeanPoolMLP(input_size=N_ROIS, num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=True)
    _check_backbone(model)


def test_meanpool_mlp_ignores_temporal_order():
    # By construction MeanPoolMLP discards temporal structure -- shuffling
    # the time axis must produce byte-identical features. eval() disables
    # dropout, which is otherwise stochastic and would make two forward
    # passes differ regardless of the input.
    model = MeanPoolMLP(input_size=N_ROIS, num_classes=NUM_CLASSES, num_conditions=NUM_CONDITIONS, include_hrf_head=False)
    model.eval()
    x = _make_input()
    perm = torch.randperm(WINDOW_LENGTH)
    x_shuffled = x[:, perm, :]
    torch.testing.assert_close(model.forward_features(x), model.forward_features(x_shuffled))
