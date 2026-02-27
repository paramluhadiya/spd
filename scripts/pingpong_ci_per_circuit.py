"""Analyze CI activation counts per PingPong circuit type (i, j).

For each circuit type (i, j), computes the average number of CI values above a threshold
across a set of input samples, then plots a histogram with error bars.

Usage:
    python scripts/pingpong_ci_per_circuit.py <model_path> [--n_batches 100] [--batch_size 256] [--threshold 0.5]

    model_path: wandb path or local path to a trained PingPong ComponentModel
"""

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.experiments.tms.bss_models import PingPongModel
from spd.experiments.tms.pingpong_decomposition import PingPongDataset
from spd.models.component_model import ComponentModel, SPDRunInfo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None, help="Path to save figure")
    args = parser.parse_args()

    run_info = SPDRunInfo.from_path(args.model_path)
    model = ComponentModel.from_run_info(run_info)
    model = model.to(args.device)
    model.eval()

    target_model = model.target_model
    assert isinstance(target_model, PingPongModel)
    D = target_model.D
    num_blocks = target_model.num_blocks

    dataset = PingPongDataset(
        D=D,
        d=target_model.d,
        num_blocks=num_blocks,
        device=args.device,
    )

    # Per-circuit counts: maps (i, j) -> list of n_above_threshold per sample
    counts_per_circuit: dict[tuple[int, int], list[float]] = defaultdict(list)

    with torch.no_grad():
        for _ in range(args.n_batches):
            batch = dataset.generate_batch(args.batch_size)

            # Extract (i, j) from input encoding
            i_indices = batch[:, D : D + num_blocks].argmax(dim=1)
            j_indices = batch[:, D + num_blocks :].argmax(dim=1)

            output = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=output.cache,
                sampling="continuous",
            )

            # Sum CI > threshold across all layers and components for each sample
            n_above = torch.zeros(args.batch_size, device=args.device)
            for layer_ci in ci.upper_leaky.values():
                n_above += (layer_ci > args.threshold).float().sum(dim=-1)

            for idx in range(args.batch_size):
                i, j = int(i_indices[idx].item()), int(j_indices[idx].item())
                counts_per_circuit[(i, j)].append(n_above[idx].item())

    # Compute mean and std per circuit
    circuits = sorted(counts_per_circuit.keys())
    labels = [f"({i},{j})" for i, j in circuits]
    means = [np.mean(counts_per_circuit[c]) for c in circuits]
    stds = [np.std(counts_per_circuit[c]) for c in circuits]

    fig, ax = plt.subplots(figsize=(max(12, len(circuits) * 0.3), 6))
    x = np.arange(len(circuits))
    ax.bar(x, means, yerr=stds, capsize=2, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Circuit (i, j)")
    ax.set_ylabel(f"Mean # CI > {args.threshold}")
    ax.set_title(f"Active subcomponents per circuit (threshold={args.threshold})")
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
