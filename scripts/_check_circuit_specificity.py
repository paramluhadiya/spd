"""Check whether j-specific components are also i-specific (i.e. circuit-specific).

For each component flagged j-specific at i=0 by `_eval_run.py`, compute the full
(i, j) CI grid (8x8 = 64 cells, 512 samples each) and report:
  - the full grid
  - all (i, j) cells where mean CI > CI_ACTIVE
  - a tag: circuit-specific (1 cell), partially specific (2-3), not i-specific (>3)

A heatmap grid plot is saved with the originally flagged (i=0, j_orig) cell
outlined in red.

Usage:
    python scripts/_check_circuit_specificity.py /path/to/checkpoint
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
LAYER = "model.0"
N_SAMPLES = 512
CI_ACTIVE = 0.3


def generate_inputs(i_val: int, j_val: int, n: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n, 80, device=device)
    x[:, i_val * d : (i_val + 1) * d] = torch.rand(n, d, device=device)
    x[:, D + i_val] = 1.0
    x[:, D + NUM_BLOCKS + j_val] = 1.0
    return x


def compute_ci(model: ComponentModel, layer: str, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ci_out = model.calc_causal_importances({layer: x}, sampling="continuous")
    return ci_out.lower_leaky[layer].mean(dim=0).cpu()


def main() -> None:
    path = sys.argv[1]
    print(f"Loading {path} ...")
    model = ComponentModel.from_pretrained(path)
    model.eval()
    device = next(model.parameters()).device

    C = model.components[LAYER].V.shape[1]
    V = model.components[LAYER].V.detach().cpu().float()

    # Auto-detect ohj mask components (clean V one-hot on ohj dims)
    mask_comps: set[int] = set()
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        if v_argmax >= D + NUM_BLOCKS and one_hotness > 0.95:
            mask_comps.add(c)
    print(f"Detected {len(mask_comps)} mask components: {sorted(mask_comps)}")

    # Full 8x8 grid of mean CI per component
    print("Computing CI grid for all 64 (i, j) circuits ...")
    ci_grid = torch.zeros(NUM_BLOCKS, NUM_BLOCKS, C)
    for i_val in range(NUM_BLOCKS):
        for j_val in range(NUM_BLOCKS):
            x = generate_inputs(i_val, j_val, N_SAMPLES, device)
            ci_grid[i_val, j_val] = compute_ci(model, LAYER, x)
    print("Done.")

    # Re-derive j-specific @ i=0 (same logic as _eval_run.py)
    j_specific: dict[int, list[int]] = {j: [] for j in range(NUM_BLOCKS)}
    for c in range(C):
        if c in mask_comps:
            continue
        cis_i0 = ci_grid[0, :, c].tolist()
        if max(cis_i0) < CI_ACTIVE:
            continue
        if sum(1 for ci in cis_i0 if ci > CI_ACTIVE) == NUM_BLOCKS:
            continue  # shared
        for j_val in range(NUM_BLOCKS):
            if cis_i0[j_val] > CI_ACTIVE:
                other_max = max(cis_i0[jj] for jj in range(NUM_BLOCKS) if jj != j_val)
                if other_max < CI_ACTIVE:
                    j_specific[j_val].append(c)

    n_total = sum(len(v) for v in j_specific.values())
    print(f"\nFound {n_total} j-specific components @ i=0")
    for j_val in range(NUM_BLOCKS):
        print(f"  j={j_val}: {sorted(j_specific[j_val])}")

    # Per-component (i, j) firing pattern
    print(f"\n{'='*72}")
    print("FULL (i, j) CI GRID PER J-SPECIFIC COMPONENT")
    print(f"{'='*72}")

    n_circuit_specific = 0
    n_partial = 0
    n_diffuse = 0

    for j_val in range(NUM_BLOCKS):
        for c in sorted(j_specific[j_val], key=lambda c: -ci_grid[0, j_val, c].item()):
            grid = ci_grid[:, :, c]
            active = [
                (i, j, grid[i, j].item())
                for i in range(NUM_BLOCKS)
                for j in range(NUM_BLOCKS)
                if grid[i, j].item() > CI_ACTIVE
            ]
            n_active = len(active)

            if n_active == 1:
                tag = "[CIRCUIT-SPECIFIC]"
                n_circuit_specific += 1
            elif n_active <= 3:
                tag = f"[partial: {n_active} cells]"
                n_partial += 1
            else:
                tag = f"[DIFFUSE: {n_active} cells]"
                n_diffuse += 1

            print(f"\n--- c={c}  (flagged j={j_val}-specific @ i=0)  {tag}")
            print("        " + " ".join(f"j={jj:<5d}" for jj in range(NUM_BLOCKS)))
            for i_val in range(NUM_BLOCKS):
                row = " ".join(f"{grid[i_val, jj].item():>5.3f} " for jj in range(NUM_BLOCKS))
                star = " <- flagged i=0" if i_val == 0 else ""
                print(f"  i={i_val}   {row}{star}")
            print(f"  active cells (CI > {CI_ACTIVE}): {n_active}")
            for i, j, v in sorted(active, key=lambda t: -t[2]):
                print(f"    (i={i}, j={j})  CI={v:.3f}")

    print(f"\n{'='*72}")
    print("AGGREGATE")
    print(f"{'='*72}")
    print(f"  Total j-specific @ i=0:  {n_total}")
    print(f"    circuit-specific (1 cell active):    {n_circuit_specific}")
    print(f"    partially specific (2-3 cells):      {n_partial}")
    print(f"    diffuse (>3 cells active):           {n_diffuse}")

    # Heatmap plot
    all_comps: list[tuple[int, int]] = []
    for j_val in range(NUM_BLOCKS):
        for c in sorted(j_specific[j_val], key=lambda c: -ci_grid[0, j_val, c].item()):
            all_comps.append((j_val, c))

    if not all_comps:
        print("\nNo j-specific components — skipping plot.")
        return

    cols = 6
    rows = (len(all_comps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.6))
    axes = np.array(axes).reshape(rows, cols)

    for k, (j_orig, c) in enumerate(all_comps):
        ax = axes[k // cols, k % cols]
        grid = ci_grid[:, :, c].numpy()
        n_active = int((grid > CI_ACTIVE).sum())
        ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="equal")
        ax.set_title(f"c={c} (j={j_orig}-spec) #act={n_active}", fontsize=8)
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_xticks(range(NUM_BLOCKS))
        ax.set_yticks(range(NUM_BLOCKS))
        ax.tick_params(labelsize=6)
        # Outline the originally flagged (i=0, j_orig) cell in red
        ax.add_patch(
            plt.Rectangle((j_orig - 0.5, -0.5), 1, 1, fill=False, edgecolor="red", lw=1.2)
        )

    for k in range(len(all_comps), rows * cols):
        axes[k // cols, k % cols].axis("off")

    plt.tight_layout()
    out = "circuit_specificity.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
