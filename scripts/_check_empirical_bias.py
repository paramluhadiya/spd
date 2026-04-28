"""Empirical-vs-explained bias check for the PingPong SPD decomposition.

For each circuit (i, j), the layer-L bias contribution is the deterministic part
of the layer output (j-block, for even layers) when source content is mean-zero
random in block i and the routing one-hots are set: ohi[i] = 1, ohj[j] = 1.

This script measures three quantities per (i, j) and compares them:

  EMP_T         empirical bias from the target model:
                mean over batch of  M_T(x)[j-block]
  EMP_S         empirical bias from the full SPD model (sum over ALL components):
                mean over batch of  Σ_c m_c(x) · (x @ V[:, c]) · U[c, j-block]
  EXPLAINED     subset sum over components we *claim* shape the bias:
                Σ_{c in explainer_set} (mean over batch of  m_c(x) · (x @ V[:, c])) · U[c, j-block]

The explainer set per (i, j) is auto-detected:
  - ohj-routing[j]    (1 component, V → e_{D+8+j})
  - NC[i]             (i-specific, j-noncommittal: active for ALL j at fixed i, only that i)
  - H1-ohi[i, j]      (V → e_{D+i}, single-circuit, routing-gated)

For comparison we also report the *static* analytical target W_T[j-block, D+i].

Reported cosines (per circuit, then aggregated):
  cos(EMP_T, b_target)      target sanity check (≈ 1.0)
  cos(EMP_T, EMP_S)         SPD faithfulness on bias-only inputs (ceiling for EXPLAINED)
  cos(EMP_T, EXPLAINED)     hypothesis vs ground truth
  cos(EXPLAINED, b_target)  hypothesis vs ground truth (norm-insensitive variant)
  cos(EMP_S, EXPLAINED)     direct: how much of SPD's bias contribution does our subset capture
                            (isolates "missing components" from SPD imperfection)

Usage:
    python scripts/_check_empirical_bias.py wandb:paramluhadiya/spd/runs/<run_id> \\
        [--layer model.0] [--n_samples 4096] [--out_prefix bias]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8

# Each layer outputs to either the j-block (even) or i-block (odd). The "bias"
# is the column of W that becomes active because of the routing one-hot of the
# OPPOSITE family (even: ohi → bias, ohj → suppression; odd: vice versa).
LAYER_OUTPUT_BLOCK_AXIS: dict[str, str] = {
    "model.0": "j",  # even
    "model.2": "i",  # odd
    "model.4": "j",  # even
}
LAYER_BIAS_INPUT_OFFSET: dict[str, str] = {
    "model.0": "ohi",  # even: bias rides on ohi
    "model.2": "ohj",  # odd:  bias rides on ohj
    "model.4": "ohi",
}


@dataclass
class ComponentInfo:
    c: int
    v_argmax: int
    v_one_hotness: float

    @property
    def v_loc(self) -> str:
        if self.v_argmax < D:
            return f"comp[{self.v_argmax // d},{self.v_argmax % d}]"
        if self.v_argmax < D + NUM_BLOCKS:
            return f"ohi[{self.v_argmax - D}]"
        return f"ohj[{self.v_argmax - D - NUM_BLOCKS}]"


def random_inputs_meanzero(
    i_val: int, j_val: int, n: int, device: torch.device, scale: float = 0.5
) -> torch.Tensor:
    """Build batch with mean-zero source in block i and one-hots set."""
    x = torch.zeros(n, D + 2 * NUM_BLOCKS, device=device)
    x[:, i_val * d : (i_val + 1) * d] = (torch.rand(n, d, device=device) - 0.5) * (2 * scale)
    x[:, D + i_val] = 1.0
    x[:, D + NUM_BLOCKS + j_val] = 1.0
    return x


def compute_ci_batch(model: ComponentModel, layer: str, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ci_out = model.calc_causal_importances({layer: x}, sampling="continuous")
    return ci_out.lower_leaky[layer].detach()  # (batch, C)


def constant_input(i_val: int, j_val: int, val: float, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, D + 2 * NUM_BLOCKS, device=device)
    x[0, i_val * d : (i_val + 1) * d] = val
    x[0, D + i_val] = 1.0
    x[0, D + NUM_BLOCKS + j_val] = 1.0
    return x


def one_hot_input(i_val: int, j_val: int, r: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, D + 2 * NUM_BLOCKS, device=device)
    x[0, i_val * d + r] = 1.0
    x[0, D + i_val] = 1.0
    x[0, D + NUM_BLOCKS + j_val] = 1.0
    return x


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    n = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / n).item()


def detect_components(
    V: torch.Tensor,
    layer: str,
) -> tuple[set[int], list[ComponentInfo]]:
    """Auto-detect routing-mask components for this layer's gating-one-hot family.

    Returns (mask_comps, all_info) where mask_comps is the set of routing-mask
    component indices (one per gate value) and all_info has component metadata.
    """
    C = V.shape[1]
    bias_offset = LAYER_BIAS_INPUT_OFFSET[layer]
    # The SUPPRESSION (routing-mask) family is the OPPOSITE of bias family.
    # Even layers: ohj suppression. Odd layers: ohi suppression.
    suppress_family = "ohj" if bias_offset == "ohi" else "ohi"
    suppress_offset = D + (NUM_BLOCKS if suppress_family == "ohj" else 0)
    mask_comps: set[int] = set()
    all_info: list[ComponentInfo] = []
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v**2).sum()).item()
        all_info.append(ComponentInfo(c=c, v_argmax=v_argmax, v_one_hotness=one_hotness))
        if (
            suppress_offset <= v_argmax < suppress_offset + NUM_BLOCKS
            and one_hotness > 0.95
        ):
            mask_comps.add(c)
    return mask_comps, all_info


def detect_nc_h1(
    model: ComponentModel,
    layer: str,
    V: torch.Tensor,
    mask_comps: set[int],
    n_samples: int,
    device: torch.device,
    ci_active: float = 0.3,
) -> tuple[
    torch.Tensor,
    dict[int, list[int]],
    dict[tuple[int, int], list[int]],
]:
    """Compute (i, j) CI grid, then auto-detect NC[i] and H1-routed[i, j]."""
    C = V.shape[1]
    bias_offset = LAYER_BIAS_INPUT_OFFSET[layer]
    bias_dim_base = D + (0 if bias_offset == "ohi" else NUM_BLOCKS)

    print("Computing baseline (i, j) CI grid ...")
    ci_grid = torch.zeros(NUM_BLOCKS, NUM_BLOCKS, C)
    for i_val in range(NUM_BLOCKS):
        for j_val in range(NUM_BLOCKS):
            x = random_inputs_meanzero(i_val, j_val, n_samples, device)
            ci_grid[i_val, j_val] = compute_ci_batch(model, layer, x).mean(dim=0).cpu()

    # NC is keyed by the bias-axis index: for even layers (bias_offset="ohi") that's i,
    # for odd layers (bias_offset="ohj") that's j. NC[b] = "always active across the
    # noncommittal axis at fixed bias-axis value b, never active at other b".
    nc: dict[int, list[int]] = {b: [] for b in range(NUM_BLOCKS)}
    for c in range(C):
        if c in mask_comps:
            continue
        for b in range(NUM_BLOCKS):
            if bias_offset == "ohi":
                # bias-axis = i, noncommittal-axis = j
                n_at_b = sum(
                    1 for j in range(NUM_BLOCKS) if ci_grid[b, j, c].item() > ci_active
                )
                other_max = max(
                    (float(ci_grid[bb, :, c].max().item()) for bb in range(NUM_BLOCKS) if bb != b),
                    default=0.0,
                )
            else:
                # bias-axis = j, noncommittal-axis = i
                n_at_b = sum(
                    1 for i in range(NUM_BLOCKS) if ci_grid[i, b, c].item() > ci_active
                )
                other_max = max(
                    (float(ci_grid[:, bb, c].max().item()) for bb in range(NUM_BLOCKS) if bb != b),
                    default=0.0,
                )
            if n_at_b == NUM_BLOCKS and other_max < ci_active:
                nc[b].append(c)

    h1: dict[tuple[int, int], list[int]] = {
        (i, j): [] for i in range(NUM_BLOCKS) for j in range(NUM_BLOCKS)
    }
    for c in range(C):
        if c in mask_comps:
            continue
        active = [
            (i, j)
            for i in range(NUM_BLOCKS)
            for j in range(NUM_BLOCKS)
            if ci_grid[i, j, c].item() > ci_active
        ]
        if len(active) != 1:
            continue
        i_val, j_val = active[0]
        # V must point at the bias-family one-hot for this circuit
        target_dim = bias_dim_base + (i_val if bias_offset == "ohi" else j_val)
        v = V[:, c]
        if int(v.abs().argmax()) != target_dim:
            continue
        # Routing-gated test: high CI at zero-source, flat across source one-hots
        ci_zero = compute_ci_batch(
            model, layer, constant_input(i_val, j_val, 0.0, device)
        )[0, c].item()
        ci_onehots = [
            compute_ci_batch(
                model, layer, one_hot_input(i_val, j_val, r, device)
            )[0, c].item()
            for r in range(d)
        ]
        spread = max(ci_onehots) - min(ci_onehots)
        if ci_zero > 0.5 and spread < 0.3:
            h1[(i_val, j_val)].append(c)

    return ci_grid, nc, h1


def measure_bias(
    target_layer: torch.nn.Linear,
    model: ComponentModel,
    layer: str,
    V: torch.Tensor,
    U: torch.Tensor,
    i_val: int,
    j_val: int,
    explainer_set: list[int],
    n_samples: int,
    device: torch.device,
) -> dict[str, torch.Tensor | float]:
    """Compute EMP_T, EMP_S, EXPLAINED, b_target for a single (i, j) circuit."""
    out_axis = LAYER_OUTPUT_BLOCK_AXIS[layer]
    out_block_idx = j_val if out_axis == "j" else i_val
    out_slice = slice(out_block_idx * d, (out_block_idx + 1) * d)

    bias_offset = LAYER_BIAS_INPUT_OFFSET[layer]
    bias_dim = D + (i_val if bias_offset == "ohi" else j_val + NUM_BLOCKS)

    x = random_inputs_meanzero(i_val, j_val, n_samples, device)

    # EMP_T from target layer (pre-activation)
    with torch.no_grad():
        y_target = target_layer(x)  # (batch, 80)
    emp_t = y_target.mean(dim=0).cpu()  # (80,)

    # Per-component scalar = mean_b [m_c(x) * (x @ V[:, c])]; SPD reconstruction
    # of layer output is Σ_c scalar_c · U[c, :].
    ci = compute_ci_batch(model, layer, x)  # (batch, C)
    proj = x @ V.to(device)  # (batch, C)
    scalars = (ci * proj).mean(dim=0).cpu()  # (C,)
    # Full SPD reconstruction: scalars @ U
    emp_s = scalars @ U  # (80,)

    # Explained subset
    if explainer_set:
        explained = scalars[explainer_set] @ U[explainer_set, :]
    else:
        explained = torch.zeros(D + 2 * NUM_BLOCKS)

    # Analytical static target: column of W_T at the bias dim
    W_T = target_layer.weight.detach().cpu().T  # (in=80, out=80)? PyTorch stores as (out, in)
    # PyTorch: weight has shape (out_features, in_features). We want the column of
    # W (input-to-output) at input-dim = bias_dim, which is W[:, bias_dim] in
    # input-output convention, i.e. weight[:, bias_dim] in PyTorch (out_features-shaped).
    b_target = target_layer.weight.detach().cpu()[:, bias_dim]  # (80,)

    # All restricted to the output block
    return dict(
        out_slice=out_slice,
        emp_t=emp_t,
        emp_t_block=emp_t[out_slice],
        emp_s=emp_s,
        emp_s_block=emp_s[out_slice],
        explained=explained,
        explained_block=explained[out_slice],
        b_target=b_target,
        b_target_block=b_target[out_slice],
        # Cosines
        cos_emp_t_vs_target=cos(emp_t[out_slice], b_target[out_slice]),
        cos_emp_t_vs_emp_s=cos(emp_t[out_slice], emp_s[out_slice]),
        cos_emp_t_vs_explained=cos(emp_t[out_slice], explained[out_slice]),
        cos_explained_vs_target=cos(explained[out_slice], b_target[out_slice]),
        cos_emp_s_vs_explained=cos(emp_s[out_slice], explained[out_slice]),
        # Norms
        norm_emp_t=float(emp_t[out_slice].norm()),
        norm_emp_s=float(emp_s[out_slice].norm()),
        norm_explained=float(explained[out_slice].norm()),
        norm_target=float(b_target[out_slice].norm()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="wandb: or local path to ComponentModel checkpoint")
    parser.add_argument("--layer", default="model.0", choices=list(LAYER_OUTPUT_BLOCK_AXIS))
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--out_prefix", default="empirical_bias")
    args = parser.parse_args()

    print(f"Loading {args.path} ...")
    model = ComponentModel.from_pretrained(args.path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    layer = args.layer
    V = model.components[layer].V.detach().float()
    U = model.components[layer].U.detach().float().cpu()
    target_layer = dict(model.target_model.model.named_children())[layer.split(".")[1]]
    assert isinstance(target_layer, torch.nn.Linear), f"Expected Linear, got {type(target_layer)}"

    print(f"Layer: {layer}  (output block axis: {LAYER_OUTPUT_BLOCK_AXIS[layer]}, "
          f"bias on: {LAYER_BIAS_INPUT_OFFSET[layer]})")

    mask_comps, _ = detect_components(V.cpu(), layer)
    print(f"Detected {len(mask_comps)} routing-mask components: {sorted(mask_comps)}")

    _, nc, h1 = detect_nc_h1(model, layer, V.cpu(), mask_comps, args.n_samples, device)

    bias_axis = "i" if LAYER_BIAS_INPUT_OFFSET[layer] == "ohi" else "j"
    print(f"\nNC[{bias_axis}] counts: ", {b: len(nc[b]) for b in range(NUM_BLOCKS)})
    h1_count = sum(len(v) for v in h1.values())
    h1_circuits = [(i, j) for (i, j), cs in h1.items() if cs]
    print(f"H1 components total: {h1_count}, on {len(h1_circuits)} circuits")

    # Per-(i, j) explainer set: NC[i] + H1[i, j] + ohj-routing[j] (or ohi-routing[i] for odd layer)
    suppress_family = "ohj" if LAYER_BIAS_INPUT_OFFSET[layer] == "ohi" else "ohi"
    suppress_offset = D + (NUM_BLOCKS if suppress_family == "ohj" else 0)

    def routing_mask_for(i_val: int, j_val: int) -> list[int]:
        """Find the routing-mask component whose V points at the relevant gate one-hot."""
        target_gate_idx = j_val if suppress_family == "ohj" else i_val
        target_dim = suppress_offset + target_gate_idx
        for c in mask_comps:
            if int(V[:, c].abs().argmax().item()) == target_dim:
                return [c]
        return []

    # Run measurement on full 8x8 grid
    print(f"\n{'=' * 90}")
    print(f"PER-CIRCUIT EMPIRICAL vs EXPLAINED BIAS  (layer={layer})")
    print(f"{'=' * 90}")
    print(
        f"{'i':>2s} {'j':>2s}  {'|NC|':>4s} {'|H1|':>4s} {'|gate|':>6s}   "
        f"{'cos(EMP_T,target)':>17s} {'cos(EMP_T,EMP_S)':>17s} "
        f"{'cos(EMP_T,EXPL)':>16s} {'cos(EXPL,target)':>16s} "
        f"{'cos(EMP_S,EXPL)':>16s}   "
        f"{'||target||':>10s} {'||EXPL||':>9s}"
    )

    grid_results: dict[tuple[int, int], dict[str, torch.Tensor | float]] = {}
    cos_emp_t_target = np.zeros((NUM_BLOCKS, NUM_BLOCKS))
    cos_emp_t_emp_s = np.zeros((NUM_BLOCKS, NUM_BLOCKS))
    cos_emp_t_explained = np.zeros((NUM_BLOCKS, NUM_BLOCKS))
    cos_explained_target = np.zeros((NUM_BLOCKS, NUM_BLOCKS))
    cos_emp_s_explained = np.zeros((NUM_BLOCKS, NUM_BLOCKS))

    for i_val in range(NUM_BLOCKS):
        for j_val in range(NUM_BLOCKS):
            bias_idx = i_val if LAYER_BIAS_INPUT_OFFSET[layer] == "ohi" else j_val
            gate = routing_mask_for(i_val, j_val)
            explainer = nc[bias_idx] + h1[(i_val, j_val)] + gate
            res = measure_bias(
                target_layer=target_layer,
                model=model,
                layer=layer,
                V=V,
                U=U,
                i_val=i_val,
                j_val=j_val,
                explainer_set=explainer,
                n_samples=args.n_samples,
                device=device,
            )
            grid_results[(i_val, j_val)] = res
            cos_emp_t_target[i_val, j_val] = res["cos_emp_t_vs_target"]
            cos_emp_t_emp_s[i_val, j_val] = res["cos_emp_t_vs_emp_s"]
            cos_emp_t_explained[i_val, j_val] = res["cos_emp_t_vs_explained"]
            cos_explained_target[i_val, j_val] = res["cos_explained_vs_target"]
            cos_emp_s_explained[i_val, j_val] = res["cos_emp_s_vs_explained"]

            print(
                f"{i_val:>2d} {j_val:>2d}  {len(nc[bias_idx]):>4d} {len(h1[(i_val, j_val)]):>4d} "
                f"{len(gate):>6d}   "
                f"{res['cos_emp_t_vs_target']:>+17.4f} {res['cos_emp_t_vs_emp_s']:>+17.4f} "
                f"{res['cos_emp_t_vs_explained']:>+16.4f} {res['cos_explained_vs_target']:>+16.4f} "
                f"{res['cos_emp_s_vs_explained']:>+16.4f}"
                f"   {res['norm_target']:>10.3f} {res['norm_explained']:>9.3f}"
            )

    print(f"\n{'=' * 90}")
    print("AGGREGATE  (over all 64 circuits)")
    print(f"{'=' * 90}")
    for name, arr in [
        ("cos(EMP_T, b_target)", cos_emp_t_target),
        ("cos(EMP_T, EMP_S)  [faithfulness]", cos_emp_t_emp_s),
        ("cos(EMP_T, EXPLAINED)", cos_emp_t_explained),
        ("cos(EXPLAINED, b_target)", cos_explained_target),
        ("cos(EMP_S, EXPLAINED)  [direct emp vs expl]", cos_emp_s_explained),
    ]:
        print(
            f"  {name:<36s}  mean={arr.mean():+.4f}  median={np.median(arr):+.4f}  "
            f"min={arr.min():+.4f}  #(>0.9)={int((arr > 0.9).sum())}/64  "
            f"#(>0.5)={int((arr > 0.5).sum())}/64"
        )

    # ------------------------------------------------------------------
    # Plots: 4 heatmaps + a few sample bar charts
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))
    titles = [
        "cos(EMP_T, b_target)",
        "cos(EMP_T, EMP_S)\n[SPD faithfulness]",
        "cos(EMP_T, EXPLAINED)",
        "cos(EXPLAINED, b_target)",
        "cos(EMP_S, EXPLAINED)\n[direct emp vs expl]",
    ]
    grids = [
        cos_emp_t_target,
        cos_emp_t_emp_s,
        cos_emp_t_explained,
        cos_explained_target,
        cos_emp_s_explained,
    ]
    for ax, title, g in zip(axes, titles, grids, strict=True):
        im = ax.imshow(g, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(NUM_BLOCKS))
        ax.set_yticks(range(NUM_BLOCKS))
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.set_title(f"{title}\nmean={g.mean():.3f}", fontsize=9)
        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                ax.text(
                    j, i, f"{g[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(g[i, j]) > 0.5 else "black",
                )
        plt.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle(
        f"{args.path} — layer {layer}  (mean-zero source, n={args.n_samples})",
        fontsize=10,
    )
    plt.tight_layout()
    out_grid = f"{args.out_prefix}_{layer.replace('.', '_')}_grid.png"
    plt.savefig(out_grid, dpi=140)
    print(f"\nSaved {out_grid}")

    # Sample bar charts: first 6 circuits with EXPLAINED actually populated
    populated = [
        (i, j) for i in range(NUM_BLOCKS) for j in range(NUM_BLOCKS)
        if grid_results[(i, j)]["norm_explained"] > 1e-3
    ]
    samples = populated[:6]
    if samples:
        fig, axes = plt.subplots(len(samples), 1, figsize=(10, 1.6 * len(samples)))
        if len(samples) == 1:
            axes = [axes]
        for ax, (i_val, j_val) in zip(axes, samples, strict=True):
            r = grid_results[(i_val, j_val)]
            x_pos = np.arange(d)
            w = 0.22
            ax.bar(x_pos - 1.5 * w, r["b_target_block"].numpy(), w, label="b_target", color="#d62728")
            ax.bar(x_pos - 0.5 * w, r["emp_t_block"].numpy(), w, label="EMP_T", color="#ff9896")
            ax.bar(x_pos + 0.5 * w, r["emp_s_block"].numpy(), w, label="EMP_S", color="#1f77b4")
            ax.bar(x_pos + 1.5 * w, r["explained_block"].numpy(), w, label="EXPLAINED", color="#2ca02c")
            ax.axhline(0, color="black", lw=0.3)
            ax.set_xticks(x_pos)
            ax.set_title(
                f"(i={i_val}, j={j_val}) "
                f"cos(EMP_T,EXPL)={r['cos_emp_t_vs_explained']:+.3f}  "
                f"cos(EXPL,target)={r['cos_explained_vs_target']:+.3f}  "
                f"cos(EMP_S,EXPL)={r['cos_emp_s_vs_explained']:+.3f}",
                fontsize=8,
            )
            ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        out_bars = f"{args.out_prefix}_{layer.replace('.', '_')}_samples.png"
        plt.savefig(out_bars, dpi=140)
        print(f"Saved {out_bars}")


if __name__ == "__main__":
    main()
