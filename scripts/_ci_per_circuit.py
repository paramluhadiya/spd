"""CI distribution for (i=0, j) circuits at model.0.

For each j in 0..7, generates inputs with source block i=0 and route j,
computes CI, excludes pre-initialized masking components (c=72-79),
and reports + plots the distribution.
"""

import matplotlib.pyplot as plt
import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
LAYER = "model.0"
MASK_RANGE = range(72, 80)  # ohj masks for model.0
N_SAMPLES = 512


def main() -> None:
    model = ComponentModel.from_pretrained("wandb:paramluhadiya/spd/s-64ee330f")
    model.eval()
    device = next(model.parameters()).device

    C = model.components[LAYER].V.shape[1]
    non_mask = torch.ones(C, dtype=torch.bool)
    non_mask[list(MASK_RANGE)] = False

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("CI distribution for (i=0, j=*) circuits at model.0\n(excluding mask components 72-79)", fontsize=14)

    for j_val in range(NUM_BLOCKS):
        x = torch.zeros(N_SAMPLES, 80, device=device)
        x[:, 0:d] = torch.rand(N_SAMPLES, d, device=device)
        x[:, D] = 1.0  # ohi = 0
        x[:, D + NUM_BLOCKS + j_val] = 1.0  # ohj = j_val

        with torch.no_grad():
            ci_out = model.calc_causal_importances({LAYER: x}, sampling="continuous")
            ci_vals = ci_out.lower_leaky[LAYER].mean(dim=0).cpu()

        ci_non_mask = ci_vals[non_mask]

        # Sort descending for the bar plot
        sorted_ci, sorted_idx = ci_non_mask.sort(descending=True)

        # Print stats
        print(f"\n=== j={j_val} ===")
        for thresh in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.01]:
            count = int((ci_non_mask > thresh).sum())
            print(f"  #CI > {thresh}: {count}")

        # Top 15
        top_vals, top_idxs = ci_vals.clone().topk(20)
        print(f"  Top 15 (including masks):")
        for val, idx in zip(top_vals[:15], top_idxs[:15]):
            tag = " [MASK]" if int(idx) in range(72, 80) else ""
            print(f"    c={int(idx):>3d}  CI={val:.4f}{tag}")

        # Plot
        ax = axes[j_val // 4, j_val % 4]
        n_show = min(60, len(sorted_ci))
        colors = ["#d62728" if v > 0.5 else "#1f77b4" if v > 0.1 else "#cccccc" for v in sorted_ci[:n_show]]
        ax.bar(range(n_show), sorted_ci[:n_show].numpy(), color=colors)
        ax.set_title(f"j={j_val}  (#>0.5: {int((ci_non_mask > 0.5).sum())}, #>0.1: {int((ci_non_mask > 0.1).sum())})")
        ax.set_ylabel("Mean CI")
        ax.set_xlabel("Component rank")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("ci_per_circuit_i0.png", dpi=150)
    print(f"\nSaved plot to ci_per_circuit_i0.png")


if __name__ == "__main__":
    main()
