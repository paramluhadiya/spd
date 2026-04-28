"""Probe what input distinguishes a circuit-specific component from background.

For each component flagged "circuit-specific" by `_check_circuit_specificity.py`
(active in exactly one (i*, j*) cell at threshold 0.3), measure CI under several
input families with routing fixed to (i*, j*):

    1. ZERO          — source block all 0
    2. CONST(v)      — source block uniformly v ∈ {0.125, 0.5, 1.0}
    3. ONE-HOT(r)    — x[i*·d + r] = 1, all other source dims 0, for r ∈ 0..7
    4. RANDOM        — source block ~ Uniform[0, 1] (already-computed baseline,
                       averaged over N_SAMPLES)

Hypotheses being tested:
    H1 (routing-only):  CI(zero) ≈ 1, all one-hot CIs similar.
    H2 (source-gated):  CI(zero) ≈ 0, one-hot CIs spike at preferred r.

Usage:
    python scripts/_probe_inputs.py /path/to/checkpoint
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


def random_inputs(i_val: int, j_val: int, n: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n, 80, device=device)
    x[:, i_val * d : (i_val + 1) * d] = torch.rand(n, d, device=device)
    x[:, D + i_val] = 1.0
    x[:, D + NUM_BLOCKS + j_val] = 1.0
    return x


def constant_input(i_val: int, j_val: int, val: float, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, 80, device=device)
    x[0, i_val * d : (i_val + 1) * d] = val
    x[0, D + i_val] = 1.0
    x[0, D + NUM_BLOCKS + j_val] = 1.0
    return x


def one_hot_input(i_val: int, j_val: int, r: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, 80, device=device)
    x[0, i_val * d + r] = 1.0
    x[0, D + i_val] = 1.0
    x[0, D + NUM_BLOCKS + j_val] = 1.0
    return x


def compute_ci_batch(model: ComponentModel, layer: str, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ci_out = model.calc_causal_importances({layer: x}, sampling="continuous")
    return ci_out.lower_leaky[layer].cpu()  # (N, C)


def main() -> None:
    path = sys.argv[1]
    print(f"Loading {path} ...")
    model = ComponentModel.from_pretrained(path)
    model.eval()
    device = next(model.parameters()).device

    C = model.components[LAYER].V.shape[1]
    V = model.components[LAYER].V.detach().cpu().float()

    # Auto-detect ohj masks
    mask_comps: set[int] = set()
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        if v_argmax >= D + NUM_BLOCKS and one_hotness > 0.95:
            mask_comps.add(c)
    print(f"Detected {len(mask_comps)} mask components")

    # Compute (i, j) baseline grid (random uniform inputs)
    print("Computing baseline (i, j) CI grid ...")
    ci_grid = torch.zeros(NUM_BLOCKS, NUM_BLOCKS, C)
    for i_val in range(NUM_BLOCKS):
        for j_val in range(NUM_BLOCKS):
            x = random_inputs(i_val, j_val, N_SAMPLES, device)
            ci_grid[i_val, j_val] = compute_ci_batch(model, LAYER, x).mean(dim=0)
    print("Done.")

    # Identify circuit-specific components (active in exactly one (i, j) cell)
    circuit_specific: list[tuple[int, int, int]] = []
    for c in range(C):
        if c in mask_comps:
            continue
        active = [
            (i, j)
            for i in range(NUM_BLOCKS)
            for j in range(NUM_BLOCKS)
            if ci_grid[i, j, c].item() > CI_ACTIVE
        ]
        if len(active) == 1:
            i_val, j_val = active[0]
            circuit_specific.append((i_val, j_val, c))

    print(f"\nFound {len(circuit_specific)} circuit-specific components")

    # Probe each one
    print(f"\n{'='*88}")
    print("INPUT-FAMILY PROBE")
    print(f"{'='*88}")

    n_h1 = 0
    n_h2 = 0
    n_other = 0
    results: list[dict] = []

    for i_val, j_val, c in sorted(circuit_specific, key=lambda t: (t[1], t[0], t[2])):
        ci_random = ci_grid[i_val, j_val, c].item()

        ci_zero = compute_ci_batch(
            model, LAYER, constant_input(i_val, j_val, 0.0, device)
        )[0, c].item()
        ci_low = compute_ci_batch(
            model, LAYER, constant_input(i_val, j_val, 0.125, device)
        )[0, c].item()
        ci_half = compute_ci_batch(
            model, LAYER, constant_input(i_val, j_val, 0.5, device)
        )[0, c].item()
        ci_one = compute_ci_batch(
            model, LAYER, constant_input(i_val, j_val, 1.0, device)
        )[0, c].item()

        ci_onehots: list[float] = []
        for r in range(d):
            ci_r = compute_ci_batch(
                model, LAYER, one_hot_input(i_val, j_val, r, device)
            )[0, c].item()
            ci_onehots.append(ci_r)

        max_oh = max(ci_onehots)
        min_oh = min(ci_onehots)
        oh_spread = max_oh - min_oh

        if ci_zero > 0.5 and oh_spread < 0.3:
            tag = "H1 (routing-gated)"
            n_h1 += 1
        elif ci_zero < 0.2 and oh_spread > 0.5:
            tag = "H2 (source-gated)"
            n_h2 += 1
        else:
            tag = "mixed"
            n_other += 1

        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        v_1hot = (v[v_argmax] ** 2 / (v ** 2).sum()).item()

        print(
            f"\n--- c={c}  circuit=({i_val}, {j_val})  V_argmax={v_argmax} "
            f"(V_1hot={v_1hot:.3f})  [{tag}]"
        )
        print(
            f"  random_U[0,1]={ci_random:.3f}   zero={ci_zero:.3f}   "
            f"const_0.125={ci_low:.3f}   const_0.5={ci_half:.3f}   const_1.0={ci_one:.3f}"
        )
        argmax_r = int(np.argmax(ci_onehots))
        print(
            "  one-hot r=0..7: " + " ".join(f"{v:.3f}" for v in ci_onehots)
            + f"   pref_r={argmax_r}  spread={oh_spread:.3f}"
        )

        results.append(
            {
                "c": c,
                "i": i_val,
                "j": j_val,
                "v_argmax": v_argmax,
                "ci_random": ci_random,
                "ci_zero": ci_zero,
                "ci_low": ci_low,
                "ci_half": ci_half,
                "ci_one": ci_one,
                "ci_onehots": ci_onehots,
                "tag": tag,
            }
        )

    print(f"\n{'='*88}")
    print("AGGREGATE")
    print(f"{'='*88}")
    print(f"  Total circuit-specific: {len(circuit_specific)}")
    print(f"    H1 (routing-gated): {n_h1}")
    print(f"    H2 (source-gated):  {n_h2}")
    print(f"    mixed:              {n_other}")

    # Plot
    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for k, r in enumerate(results):
        ax = axes[k // cols, k % cols]
        # Layout: r=0..7 | zero | 0.125 | 0.5 | 1.0 | rand
        positions = list(range(d)) + [d + 1, d + 2, d + 3, d + 4, d + 6]
        labels = [f"r{ii}" for ii in range(d)] + ["zero", "0.125", "0.5", "1.0", "rand"]
        values = r["ci_onehots"] + [
            r["ci_zero"],
            r["ci_low"],
            r["ci_half"],
            r["ci_one"],
            r["ci_random"],
        ]
        colors = ["#1f77b4"] * d + ["#888888"] * 4 + ["#d62728"]
        ax.bar(positions, values, color=colors)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)
        ax.set_title(
            f"c={r['c']} circ=({r['i']},{r['j']}) Vmax={r['v_argmax']} [{r['tag']}]",
            fontsize=8,
        )
        ax.set_ylabel("CI")

    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")

    plt.tight_layout()
    out = "input_probe.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
