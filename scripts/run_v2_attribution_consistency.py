"""Cross-representation concept-attribution consistency check (v2 extension,
user-proposed 2026-08-26).

CAV *directions* live in each representation's own, independently-trained
128-dim space and can't be compared across representations -- there's no
shared basis. But concept-*attribution* (backprop h(x).v_C to the raw
input, aggregated to the 7 Yeo RSNs) lives in the shared, physically
meaningful input space every representation is trained on. So: does each
of the 6 representations (3 paradigms x 2 architectures) point to the same
resting-state network for the same concept, even though they arrived at
that concept through completely different training objectives? Two
representations both hitting TCAV=1.0 for "tongue" doesn't tell you this --
they could be confident about it while localizing it to different networks.
This does.

Averaged over all 30 resamples per representation (not just resample 0),
per the user's explicit choice given the added compute cost is bounded.
For each resample: build that resample's held-out split once, load all 6
checkpoints, fit each Case 2/3 representation's post-hoc classifier head,
extract train features once per representation, then for each of the 8
known concepts fit a CAV and run concept-attribution on that resample's
test set. Raw [window_length, n_rois] attribution is averaged across all
30 resamples (each resample weighted equally, matching this project's
established repeated-split convention) before aggregating to networks --
not averaged after network-aggregation -- so the final comparison is one
number per (representation, concept, network), not 30 independently noisy
ones.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from neurolens import verification_v2 as v2
from neurolens.concepts import EXTENDED_CONCEPT_DEFINITIONS, train_cav
from neurolens.data_setup import make_dataloaders
from neurolens.interpretability import NETWORK_NAMES, aggregate_attribution_to_networks, load_roi_to_network, network_roi_indices

OUT_PATH = Path("results/v2_attribution_consistency.json")
N_RESAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else float("nan")


def main() -> None:
    device = torch.device("cpu")
    network_indices = network_roi_indices(load_roi_to_network(v2.ROI_LABELS_PATH))

    splits = v2.load_splits()[:N_RESAMPLES]
    accum_attr: dict[tuple[str, str, str], np.ndarray] = {}
    accum_probe_acc: dict[tuple[str, str, str], list[float]] = {}

    t_start = time.time()
    for resample_idx, split in enumerate(splits):
        t0 = time.time()
        train_loader, _, test_loader, info = make_dataloaders(
            v2.PROCESSED_ROOT, batch_size=64,
            train_subjects=split["train_subjects"], val_subjects=split["val_subjects"], test_subjects=split["test_subjects"],
        )

        for case, arch in v2.REPRESENTATIONS:
            model = v2.load_representation(case, arch, resample_idx, info, None, device)
            wrapped = v2.fit_differentiable_head(model, case, train_loader, device, num_classes=info["num_classes"])
            feats, labels = v2.get_features_and_labels(wrapped, case, train_loader, device)

            for concept_name in v2.CONCEPT_NAMES:
                positive_classes, negative_classes = EXTENDED_CONCEPT_DEFINITIONS[concept_name]
                cav = train_cav(feats, labels, positive_classes, negative_classes)
                attr = v2.concept_attribution(wrapped, case, test_loader, cav["direction"], device)

                key = (case, arch, concept_name)
                accum_attr[key] = attr if key not in accum_attr else accum_attr[key] + attr
                accum_probe_acc.setdefault(key, []).append(cav["probe_accuracy"])

        elapsed = time.time() - t_start
        rate = (resample_idx + 1) / elapsed
        remaining = (N_RESAMPLES - (resample_idx + 1)) / rate if rate > 0 else float("nan")
        print(
            f"[resample {resample_idx + 1}/{N_RESAMPLES}] this_resample={time.time() - t0:.1f}s "
            f"elapsed={elapsed / 60:.1f}min est_remaining={remaining / 60:.1f}min",
            flush=True,
        )

    # Average raw attribution across resamples, THEN aggregate to networks --
    # one stable number per (representation, concept, network), not 30 noisy ones.
    per_rep_concept_network: dict[tuple[str, str, str], np.ndarray] = {}
    for key, summed_attr in accum_attr.items():
        mean_attr = summed_attr / N_RESAMPLES
        per_rep_concept_network[key] = aggregate_attribution_to_networks(mean_attr, network_indices)

    results = {}
    for concept_name in v2.CONCEPT_NAMES:
        per_representation = {
            f"{case}_{arch}": per_rep_concept_network[(case, arch, concept_name)].tolist()
            for case, arch in v2.REPRESENTATIONS
        }
        top_networks = {
            rep: NETWORK_NAMES[int(np.argmax(vec))] for rep, vec in per_representation.items()
        }
        top_network_counts = {}
        for net in top_networks.values():
            top_network_counts[net] = top_network_counts.get(net, 0) + 1
        majority_network, majority_count = max(top_network_counts.items(), key=lambda kv: kv[1])

        reps = list(per_representation.keys())
        similarity_matrix = {
            r1: {r2: cosine_similarity(np.array(per_representation[r1]), np.array(per_representation[r2])) for r2 in reps}
            for r1 in reps
        }

        # same-architecture, cross-case comparison (isolates training
        # objective from architecture, matching this project's established
        # comparison convention)
        cross_case_same_arch = {}
        for arch in v2.ARCHITECTURES:
            arch_reps = [f"{case}_{arch}" for case in v2.CASES]
            pairs = [
                similarity_matrix[arch_reps[i]][arch_reps[j]]
                for i in range(len(arch_reps)) for j in range(i + 1, len(arch_reps))
            ]
            cross_case_same_arch[arch] = float(np.mean(pairs))

        results[concept_name] = {
            "per_representation_network_attribution": per_representation,
            "top_network_by_representation": top_networks,
            "majority_network": majority_network,
            "n_representations_agreeing_on_majority": majority_count,
            "cosine_similarity_matrix": similarity_matrix,
            "mean_cross_case_cosine_similarity_by_arch": cross_case_same_arch,
            "mean_probe_accuracy": {
                f"{case}_{arch}": float(np.mean(accum_probe_acc[(case, arch, concept_name)]))
                for case, arch in v2.REPRESENTATIONS
            },
        }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
