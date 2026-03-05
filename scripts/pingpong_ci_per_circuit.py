"""Analyze CI activation counts per PingPong circuit type (i, j).

For each circuit type (i, j), computes the average number of CI values above a threshold
across a set of input samples, then plots a histogram with error bars.

Usage:
    python scripts/pingpong_ci_per_circuit.py <model_path> [--n_batches 100] [--batch_size 256] [--threshold 0.9]

    model_path: wandb path or local path to a trained PingPong ComponentModel
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.experiments.tms.bss_models import PingPongModel
from spd.experiments.tms.pingpong_decomposition import PingPongDataset
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.settings import SPD_OUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None, help="Path to save figure (default: spd_out/)")
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

    layer_names = model.target_module_paths

    # Per-circuit, per-layer counts: maps (layer, i, j) -> list of counts per sample
    counts_per_layer_circuit: dict[str, dict[tuple[int, int], list[float]]] = {
        name: defaultdict(list) for name in layer_names
    }
    # Also track totals
    counts_total: dict[tuple[int, int], list[float]] = defaultdict(list)

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

            n_above_total = torch.zeros(args.batch_size, device=args.device)
            per_layer_n_above: dict[str, torch.Tensor] = {}
            for name, layer_ci in ci.upper_leaky.items():
                n = (layer_ci > args.threshold).float().sum(dim=-1)
                per_layer_n_above[name] = n
                n_above_total += n

            for idx in range(args.batch_size):
                i, j = int(i_indices[idx].item()), int(j_indices[idx].item())
                counts_total[(i, j)].append(n_above_total[idx].item())
                for name in layer_names:
                    counts_per_layer_circuit[name][(i, j)].append(
                        per_layer_n_above[name][idx].item()
                    )

    circuits = sorted(counts_total.keys())
    labels = [f"({i},{j})" for i, j in circuits]
    x = np.arange(len(circuits))
    n_plots = len(layer_names) + 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(max(12, len(circuits) * 0.3), 4 * n_plots))

    # Total plot
    means = [np.mean(counts_total[c]) for c in circuits]
    stds = [np.std(counts_total[c]) for c in circuits]
    axes[0].bar(x, means, yerr=stds, capsize=2, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_ylabel(f"Mean # CI > {args.threshold}")
    axes[0].set_title(f"Total active subcomponents per circuit (threshold={args.threshold})")

    # Per-layer plots with shared y-axis scale
    layer_max_y = 0.0
    layer_plot_data: list[tuple[list[float], list[float]]] = []
    for name in layer_names:
        layer_counts = counts_per_layer_circuit[name]
        m = [np.mean(layer_counts[c]) for c in circuits]
        s = [np.std(layer_counts[c]) for c in circuits]
        layer_plot_data.append((m, s))
        layer_max_y = max(layer_max_y, max(mi + si for mi, si in zip(m, s)))

    for idx, name in enumerate(layer_names):
        ax = axes[idx + 1]
        m, s = layer_plot_data[idx]
        ax.bar(x, m, yerr=s, capsize=2, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel(f"Mean # CI > {args.threshold}")
        ax.set_title(f"{name}: active subcomponents per circuit")
        ax.set_ylim(0, layer_max_y * 1.05)

    fig.tight_layout()

    out_path = Path(args.out) if args.out else SPD_OUT_DIR / "ci_per_circuit.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
