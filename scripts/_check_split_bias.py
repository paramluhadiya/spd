"""Test the 'split bias' hypothesis for H1 (routing-gated) circuit-specific components.

Hypothesis: H1 components with V pointing at a routing one-hot (D+i_v) are the
target model's bias contribution for source-i_v, but split per (i_v, j*) circuit by
the beta=inf importance-minimality pressure.

For circuit (i_v, j*) the prediction is that the component's contribution to the
bias column W[:, V_argmax] = V[V_argmax, c] · U[c, :] is concentrated in the
j*-block.

Per H1 component we report:
  - U mass concentration on block j*  (j*_frac)
  - cos(V[V_argmax, c] · U[c, :], W[:, V_argmax])              FULL
  - cos(V[V_argmax, c] · U[c, :][j*-block], W[j*-block, V_argmax])  SPLIT

The V-weighting is critical: cosine of raw U with W[:, V_argmax] flips sign when
V[V_argmax, c] is negative — so without weighting an aligned component looks
anti-aligned. (For a single component this reduces to sign(V[V_argmax, c]) · U
since cosine is scale-invariant.)

Aggregate stats are also reported across H1, H2, mixed, and circuit-specific overall.

Usage:
    python scripts/_check_split_bias.py /path/to/checkpoint
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
    return ci_out.lower_leaky[layer].cpu()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    n = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / n).item()


def classify(ci_zero: float, ci_onehots: list[float]) -> str:
    spread = max(ci_onehots) - min(ci_onehots)
    if ci_zero > 0.5 and spread < 0.3:
        return "H1"
    if ci_zero < 0.2 and spread > 0.5:
        return "H2"
    return "mixed"


def main() -> None:
    path = sys.argv[1]
    print(f"Loading {path} ...")
    model = ComponentModel.from_pretrained(path)
    model.eval()
    device = next(model.parameters()).device

    C = model.components[LAYER].V.shape[1]
    V = model.components[LAYER].V.detach().cpu().float()
    U = model.components[LAYER].U.detach().cpu().float()
    W = model.target_weight(LAYER).detach().cpu().float()  # (d_out=80, d_in=80)
    print(f"W shape: {tuple(W.shape)}")

    # Auto-detect ohj masks
    mask_comps: set[int] = set()
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        if v_argmax >= D + NUM_BLOCKS and one_hotness > 0.95:
            mask_comps.add(c)

    # Baseline (i, j) grid
    print("Computing baseline (i, j) grid ...")
    ci_grid = torch.zeros(NUM_BLOCKS, NUM_BLOCKS, C)
    for i_val in range(NUM_BLOCKS):
        for j_val in range(NUM_BLOCKS):
            x = random_inputs(i_val, j_val, N_SAMPLES, device)
            ci_grid[i_val, j_val] = compute_ci_batch(model, LAYER, x).mean(dim=0)

    # Find circuit-specific components
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

    # Classify each
    print("Classifying H1/H2/mixed ...")
    h_class: dict[tuple[int, int, int], str] = {}
    for i_val, j_val, c in circuit_specific:
        ci_zero = compute_ci_batch(
            model, LAYER, constant_input(i_val, j_val, 0.0, device)
        )[0, c].item()
        ci_onehots = [
            compute_ci_batch(model, LAYER, one_hot_input(i_val, j_val, r, device))[0, c].item()
            for r in range(d)
        ]
        h_class[(i_val, j_val, c)] = classify(ci_zero, ci_onehots)

    # Compute split-bias metrics for everyone
    rows: list[dict] = []
    for (i_val, j_val, c), cls in h_class.items():
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        v_1hot = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        u_full = U[c, :]

        # U block analysis on output computational dims
        u_comp = u_full[:D].reshape(NUM_BLOCKS, d)
        block_mass = (u_comp ** 2).sum(dim=1)
        total_comp = block_mass.sum().clamp_min(1e-30)
        block_frac = block_mass / total_comp
        u_best_block = int(block_frac.argmax())
        u_best_frac = block_frac.max().item()
        j_star_frac = block_frac[j_val].item()

        # Bias targets — based on V_argmax (whichever input dim V points at)
        # Component c's contribution to W[:, v_argmax] is V[v_argmax, c] · U[c, :].
        # Cosine is scale-invariant in magnitude, so this is sign(V[v_argmax, c]) · U.
        v_proj = float(v[v_argmax].item())
        v_sign = float(np.sign(v_proj)) if v_proj != 0 else 1.0
        u_signed = v_sign * u_full

        full_bias_tgt = W[:, v_argmax]  # (80,)
        cos_full = cos(u_signed, full_bias_tgt)

        # Split bias: only j*-block of the bias column. Compare the j*-block slice
        # of the signed U against the j*-block of W[:, v_argmax].
        u_signed_j = u_signed[j_val * d : (j_val + 1) * d]
        bias_j = W[j_val * d : (j_val + 1) * d, v_argmax]
        cos_split = cos(u_signed_j, bias_j)

        # Categorize V_argmax location
        if v_argmax < D:
            v_loc = f"comp blk {v_argmax // d} dim {v_argmax % d}"
        elif v_argmax < D + NUM_BLOCKS:
            v_loc = f"ohi[{v_argmax - D}]"
        else:
            v_loc = f"ohj[{v_argmax - D - NUM_BLOCKS}]"

        rows.append(
            dict(
                c=c,
                i=i_val,
                j=j_val,
                cls=cls,
                v_argmax=v_argmax,
                v_loc=v_loc,
                v_1hot=v_1hot,
                v_proj=v_proj,
                v_sign=v_sign,
                u_best_block=u_best_block,
                u_best_frac=u_best_frac,
                j_star_frac=j_star_frac,
                cos_full=cos_full,
                cos_split=cos_split,
            )
        )

    # H1 detail table
    h1_rows = sorted([r for r in rows if r["cls"] == "H1"], key=lambda r: (r["i"], r["j"]))
    print(f"\n{'='*120}")
    print(f"H1 COMPONENTS (N={len(h1_rows)}) — split-bias check")
    print(f"{'='*120}")
    print(
        f"\n{'c':>4s}  {'circ':>7s}  {'V_argmax':>8s}  {'V_loc':>16s}  {'V_1hot':>7s}  "
        f"{'V_proj':>8s}  {'U_best':>7s}  {'U_best_frac':>11s}  {'j*_frac':>8s}  "
        f"{'cos_full':>9s}  {'cos_split':>10s}"
    )
    for r in h1_rows:
        match = "*" if r["u_best_block"] == r["j"] else " "
        print(
            f"{r['c']:>4d}  ({r['i']},{r['j']})  {r['v_argmax']:>8d}  {r['v_loc']:>16s}  "
            f"{r['v_1hot']:>7.4f}  {r['v_proj']:>+8.3f}  {r['u_best_block']:>6d}{match}  "
            f"{r['u_best_frac']:>11.4f}  {r['j_star_frac']:>8.4f}  "
            f"{r['cos_full']:>9.4f}  {r['cos_split']:>10.4f}"
        )

    # H1 split: by V location (ohi-V vs computational-V)
    h1_ohi = [r for r in h1_rows if r["v_loc"].startswith("ohi")]
    h1_other = [r for r in h1_rows if not r["v_loc"].startswith("ohi")]
    print(f"\n  H1 with V → ohi (split-bias candidates): {len(h1_ohi)}")
    if h1_ohi:
        n_match = sum(1 for r in h1_ohi if r["u_best_block"] == r["j"])
        cos_splits = [r["cos_split"] for r in h1_ohi]
        cos_fulls = [r["cos_full"] for r in h1_ohi]
        j_fracs = [r["j_star_frac"] for r in h1_ohi]
        print(f"    #(U_best == j*):          {n_match}/{len(h1_ohi)}")
        print(f"    j*_frac:    mean={np.mean(j_fracs):.3f}  median={np.median(j_fracs):.3f}")
        print(f"    cos_full:   mean={np.mean(cos_fulls):.3f}  median={np.median(cos_fulls):.3f}")
        print(f"    cos_split:  mean={np.mean(cos_splits):.3f}  median={np.median(cos_splits):.3f}")
    print(f"\n  H1 with V → other (computational): {len(h1_other)}")
    if h1_other:
        n_match = sum(1 for r in h1_other if r["u_best_block"] == r["j"])
        cos_splits = [r["cos_split"] for r in h1_other]
        cos_fulls = [r["cos_full"] for r in h1_other]
        j_fracs = [r["j_star_frac"] for r in h1_other]
        print(f"    #(U_best == j*):          {n_match}/{len(h1_other)}")
        print(f"    j*_frac:    mean={np.mean(j_fracs):.3f}  median={np.median(j_fracs):.3f}")
        print(f"    cos_full:   mean={np.mean(cos_fulls):.3f}  median={np.median(cos_fulls):.3f}")
        print(f"    cos_split:  mean={np.mean(cos_splits):.3f}  median={np.median(cos_splits):.3f}")

    # Aggregate by classification
    print(f"\n{'='*120}")
    print("AGGREGATES BY CLASSIFICATION")
    print(f"{'='*120}")
    for cls in ["H1", "H2", "mixed"]:
        cls_rows = [r for r in rows if r["cls"] == cls]
        if not cls_rows:
            continue
        cos_splits = [r["cos_split"] for r in cls_rows]
        cos_fulls = [r["cos_full"] for r in cls_rows]
        j_fracs = [r["j_star_frac"] for r in cls_rows]
        n_match = sum(1 for r in cls_rows if r["u_best_block"] == r["j"])
        print(
            f"  {cls:>5s}  N={len(cls_rows):>3d}  "
            f"#(U_best=j*)={n_match}/{len(cls_rows):<3d}  "
            f"j*_frac mean={np.mean(j_fracs):.3f}  "
            f"cos_full mean={np.mean(cos_fulls):.3f}  "
            f"cos_split mean={np.mean(cos_splits):.3f}"
        )

    # Plot the H1-ohi components
    if h1_ohi:
        n = len(h1_ohi)
        fig, axes = plt.subplots(n, 2, figsize=(14, 1.4 * n))
        if n == 1:
            axes = axes.reshape(1, 2)

        for k, r in enumerate(h1_ohi):
            i_val, j_val, c, v_argmax = r["i"], r["j"], r["c"], r["v_argmax"]
            v_sign = r["v_sign"]
            # Plot V-signed U so visual sign matches the bias contribution direction
            u_full = (v_sign * U[c, :]).numpy()
            split_target = np.zeros(80)
            split_target[j_val * d : (j_val + 1) * d] = W[
                j_val * d : (j_val + 1) * d, v_argmax
            ].numpy()

            u_norm = u_full / max(np.abs(u_full).max(), 1e-30)
            tgt_norm = split_target / max(np.abs(split_target).max(), 1e-30)

            ax = axes[k, 0]
            colors = ["#1f77b4"] * 80
            for x_idx in range(j_val * d, (j_val + 1) * d):
                colors[x_idx] = "#d62728"
            ax.bar(range(80), u_norm, color=colors)
            for r_block in range(NUM_BLOCKS + 1):
                ax.axvline(r_block * d - 0.5, color="gray", linestyle=":", alpha=0.4)
            ax.set_xlim(-0.5, 79.5)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(
                f"c={c}  U[c,:]  circuit ({i_val},{j_val})  cos_split={r['cos_split']:.2f}",
                fontsize=8,
            )

            ax = axes[k, 1]
            ax.bar(range(80), tgt_norm, color="#d62728")
            for r_block in range(NUM_BLOCKS + 1):
                ax.axvline(r_block * d - 0.5, color="gray", linestyle=":", alpha=0.4)
            ax.set_xlim(-0.5, 79.5)
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(
                f"split-bias target  W[blk-{j_val}, src=ohi[{v_argmax - D}]]", fontsize=8
            )

        plt.tight_layout()
        out = "split_bias.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
