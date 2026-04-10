"""Diff learned SPD components against per-circuit ideal components.

Loads a PingPong SPD run and compares each learned (V[:,c], U[c,:]) pair against
two candidate "true" decompositions:
  - Option 1: one component per input dim (V=e_k, U=row k of W^T). 64 computational.
  - Option 2: one component per (src, route, neuron) triplet (V=e_k,
    U=single-block-sparse row k of W^T restricted to one route block). 512 computational.

Both options share 16 indexing targets (V=e_k for k in [64,80), U=W^T row k).

Usage:
    python scripts/diff_pingpong_components.py
    python scripts/diff_pingpong_components.py --run wandb:paramluhadiya/spd/s-1c8b8e5d
    python scripts/diff_pingpong_components.py --no_figures
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from spd.experiments.tms.pingpong_percircuit_ideal_init_decomposition import (
    D,
    N_COMPUTATIONAL,
    N_INDEXING,
    NUM_BLOCKS,
    comp_index,
    d,
)
from spd.models.component_model import ComponentModel
from spd.settings import SPD_OUT_DIR

LAYER_NAMES = ["model.0", "model.2", "model.4"]
N_SAMPLE_HEATMAPS = 12


def build_option1_targets(W_T: torch.Tensor) -> torch.Tensor:
    """Option 1: 64 computational outer-product targets, indexed by k in [0, 64).

    V1[:, k] = e_k, U1[k, :] = W_T[k, :]. Outer product has nonzeros only on row k.
    Shape: (64, d_in, d_out).
    """
    d_in, d_out = W_T.shape
    T1 = torch.zeros(D, d_in, d_out)
    for k in range(D):
        T1[k, k, :] = W_T[k, :]
    return T1


def build_option2_targets(W_T: torch.Tensor) -> torch.Tensor:
    """Option 2: 512 computational outer-product targets, indexed by comp_index.

    For idx = comp_index(src, route, neuron), k = src*d + neuron:
        V2[:, idx] = e_k, U2[idx, route*d:(route+1)*d] = W_T[k, route*d:(route+1)*d].
    Outer product has 8 nonzeros on row k, cols route*d:(route+1)*d.
    Shape: (512, d_in, d_out).
    """
    d_in, d_out = W_T.shape
    T2 = torch.zeros(N_COMPUTATIONAL, d_in, d_out)
    for src in range(NUM_BLOCKS):
        for route in range(NUM_BLOCKS):
            for neuron in range(d):
                idx = comp_index(src, route, neuron)
                k = src * d + neuron
                T2[idx, k, route * d : (route + 1) * d] = W_T[
                    k, route * d : (route + 1) * d
                ]
    return T2


def build_indexing_targets(W_T: torch.Tensor) -> torch.Tensor:
    """16 indexing outer-product targets. Row D+i for i in [0, 16).

    Shape: (16, d_in, d_out).
    """
    d_in, d_out = W_T.shape
    T_idx = torch.zeros(N_INDEXING, d_in, d_out)
    for i in range(N_INDEXING):
        k = D + i
        T_idx[i, k, :] = W_T[k, :]
    return T_idx


def cosine_max_match(
    L_flat: torch.Tensor, T_flat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row of L, find the best-matching row in T by cosine similarity."""
    L_norm = F.normalize(L_flat, dim=-1, eps=1e-12)
    T_norm = F.normalize(T_flat, dim=-1, eps=1e-12)
    sims = L_norm @ T_norm.T
    best_cos, best_idx = sims.max(dim=-1)
    return best_cos, best_idx


def coverage_max_match(
    T_flat: torch.Tensor, L_flat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row of T, find the best-matching learned component in L."""
    T_norm = F.normalize(T_flat, dim=-1, eps=1e-12)
    L_norm = F.normalize(L_flat, dim=-1, eps=1e-12)
    sims = T_norm @ L_norm.T
    best_cos, best_idx = sims.max(dim=-1)
    return best_cos, best_idx


def analyse_layer(
    module_name: str,
    V: torch.Tensor,
    U: torch.Tensor,
    W_T: torch.Tensor,
) -> dict[str, Any]:
    d_in, C = V.shape
    _, d_out = U.shape
    assert d_in == d_out == 80, f"Expected 80-dim linear, got V={V.shape} U={U.shape}"

    # --- (A) V one-hotness: squared cosine to nearest standard basis vector ---
    v_abs = V.abs()
    k_c = v_abs.argmax(dim=0)  # (C,)
    v_norm_sq = (V * V).sum(dim=0)  # (C,)
    v_top = V[k_c, torch.arange(C)]
    one_hotness = (v_top * v_top) / v_norm_sq.clamp_min(1e-30)  # (C,)

    is_comp = k_c < D  # (C,)
    is_idx = k_c >= D  # (C,)

    # --- (B) U block-sparsity (computational part only) ---
    U_comp_part = U[:, : d * NUM_BLOCKS]  # (C, 64)
    U_blocks = U_comp_part.reshape(C, NUM_BLOCKS, d)  # (C, 8, 8)
    m = (U_blocks * U_blocks).sum(dim=-1)  # (C, 8)
    total_mass = m.sum(dim=-1).clamp_min(1e-30)
    block_concentration = m.max(dim=-1).values / total_mass  # (C,)
    probs = m / total_mass.unsqueeze(-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1)
    block_entropy = entropy / math.log(NUM_BLOCKS)

    # --- (C) Outer-product cosine vs option 1, option 2, indexing targets ---
    L = torch.einsum("dc,ce->cde", V, U)  # (C, d_in, d_out)
    L_flat = L.reshape(C, -1)

    T1 = build_option1_targets(W_T)
    T2 = build_option2_targets(W_T)
    T_idx = build_indexing_targets(W_T)

    cos1, match1 = cosine_max_match(L_flat, T1.reshape(D, -1))
    cos2, match2 = cosine_max_match(L_flat, T2.reshape(N_COMPUTATIONAL, -1))
    cos_idx, match_idx = cosine_max_match(L_flat, T_idx.reshape(N_INDEXING, -1))

    # --- (D) Coverage: for each target, max cosine over learned components ---
    cov1_best, cov1_by = coverage_max_match(T1.reshape(D, -1), L_flat)
    cov2_best, cov2_by = coverage_max_match(T2.reshape(N_COMPUTATIONAL, -1), L_flat)
    cov_idx_best, cov_idx_by = coverage_max_match(T_idx.reshape(N_INDEXING, -1), L_flat)

    # --- Indexing U alignment: cosine(U[c,:], W_T[k_c,:]) ---
    Wt_rows_for_k = W_T[k_c, :]  # (C, d_out)
    u_cos_to_k = F.cosine_similarity(U, Wt_rows_for_k, dim=-1)  # (C,)

    # --- Aggregate counts ---
    n_comp = int(is_comp.sum().item())
    n_idx = int(is_idx.sum().item())
    n_low_oh = int((one_hotness < 0.5).sum().item())

    def _mean(t: torch.Tensor, mask: torch.Tensor) -> float | None:
        if not bool(mask.any()):
            return None
        return float(t[mask].mean().item())

    results: dict[str, Any] = {
        "module_name": module_name,
        "C": C,
        "n_computational": n_comp,
        "n_indexing": n_idx,
        "n_low_one_hotness": n_low_oh,
        "mean_one_hotness_all": float(one_hotness.mean().item()),
        # Computational half
        "mean_block_concentration_comp": _mean(block_concentration, is_comp),
        "mean_block_entropy_comp": _mean(block_entropy, is_comp),
        "n_block_concentration_gt_0_9_comp": int(
            ((block_concentration > 0.9) & is_comp).sum().item()
        ),
        "n_block_concentration_lt_0_2_comp": int(
            ((block_concentration < 0.2) & is_comp).sum().item()
        ),
        "mean_cos1_comp": _mean(cos1, is_comp),
        "mean_cos2_comp": _mean(cos2, is_comp),
        "n_cos2_gt_cos1_comp": int(((cos2 > cos1) & is_comp).sum().item()),
        "n_cos1_gt_cos2_comp": int(((cos1 > cos2) & is_comp).sum().item()),
        "opt1_coverage_mean": float(cov1_best.mean().item()),
        "opt1_coverage_n_gt_0_9": int((cov1_best > 0.9).sum().item()),
        "opt2_coverage_mean": float(cov2_best.mean().item()),
        "opt2_coverage_n_gt_0_9": int((cov2_best > 0.9).sum().item()),
        # Indexing half
        "mean_idx_v_one_hotness": _mean(one_hotness, is_idx),
        "mean_idx_u_cos_to_k": _mean(u_cos_to_k, is_idx),
        "idx_coverage_mean": float(cov_idx_best.mean().item()),
        "idx_coverage_n_gt_0_9": int((cov_idx_best > 0.9).sum().item()),
    }

    per_comp: dict[str, list[Any]] = {
        "k_c": k_c.tolist(),
        "one_hotness": one_hotness.tolist(),
        "is_computational": is_comp.tolist(),
        "is_indexing": is_idx.tolist(),
        "block_concentration": block_concentration.tolist(),
        "block_entropy": block_entropy.tolist(),
        "cos1": cos1.tolist(),
        "cos2": cos2.tolist(),
        "cos_idx": cos_idx.tolist(),
        "match1": match1.tolist(),
        "match2": match2.tolist(),
        "match_idx": match_idx.tolist(),
        "u_cos_to_k": u_cos_to_k.tolist(),
    }

    coverage = {
        "opt1_best_cos_per_target": cov1_best.tolist(),
        "opt1_best_component_per_target": cov1_by.tolist(),
        "opt2_best_cos_per_target": cov2_best.tolist(),
        "opt2_best_component_per_target": cov2_by.tolist(),
        "idx_best_cos_per_target": cov_idx_best.tolist(),
        "idx_best_component_per_target": cov_idx_by.tolist(),
    }

    figure_tensors = {
        "V": V,
        "U": U,
        "W_T": W_T,
        "one_hotness": one_hotness,
        "block_concentration": block_concentration,
        "cos1": cos1,
        "cos2": cos2,
        "k_c": k_c,
        "is_comp": is_comp,
        "match2": match2,
    }

    return {
        "results": results,
        "per_comp": per_comp,
        "coverage": coverage,
        "figure_tensors": figure_tensors,
    }


def save_figures(layer_name: str, t: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    one_hotness = t["one_hotness"].numpy()
    block_concentration = t["block_concentration"].numpy()
    cos1 = t["cos1"].numpy()
    cos2 = t["cos2"].numpy()
    is_comp = t["is_comp"].numpy().astype(bool)
    k_c = t["k_c"].numpy()
    U = t["U"]
    W_T = t["W_T"]
    match2 = t["match2"]

    # Histogram: V one_hotness
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(one_hotness, bins=50)
    ax.set_title(f"{layer_name} — V one_hotness")
    ax.set_xlabel("max(v_k)^2 / ||v||^2")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_one_hotness.png", dpi=120)
    plt.close(fig)

    # Histogram: U block_concentration (computational only)
    fig, ax = plt.subplots(figsize=(6, 4))
    if is_comp.any():
        ax.hist(block_concentration[is_comp], bins=50)
    ax.set_title(f"{layer_name} — U block_concentration (computational)")
    ax.set_xlabel("max block L2^2 / total mass")
    ax.set_ylabel("count")
    ax.axvline(1.0 / NUM_BLOCKS, linestyle=":", color="gray", label="1/8 (option 1)")
    ax.axvline(1.0, linestyle=":", color="green", label="1 (option 2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_block_concentration.png", dpi=120)
    plt.close(fig)

    # Scatter: cos1 vs cos2 for computational
    fig, ax = plt.subplots(figsize=(6, 6))
    if is_comp.any():
        ax.scatter(cos1[is_comp], cos2[is_comp], s=6, alpha=0.5)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("cos vs option 1 (full-row)")
    ax.set_ylabel("cos vs option 2 (block-sparse)")
    ax.set_title(f"{layer_name} — computational cos1 vs cos2")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_cos1_vs_cos2.png", dpi=120)
    plt.close(fig)

    # Sample component heatmaps — quantile-sampled by one_hotness among computational
    comp_ids = np.where(is_comp)[0]
    if len(comp_ids) == 0:
        return
    order = comp_ids[np.argsort(-one_hotness[comp_ids])]
    n = min(N_SAMPLE_HEATMAPS, len(order))
    picks = np.unique(np.linspace(0, len(order) - 1, n).astype(int))
    sampled = order[picks]
    n = len(sampled)

    fig, axes = plt.subplots(n, 3, figsize=(11, 1.1 * n + 1), squeeze=False)
    for row, c in enumerate(sampled):
        k = int(k_c[c])
        learned_u = U[c].numpy()
        opt1_u = W_T[k].numpy()

        opt2_idx = int(match2[c].item())
        src = opt2_idx // (NUM_BLOCKS * d)
        route = (opt2_idx // d) % NUM_BLOCKS
        neuron = opt2_idx % d
        k_opt2 = src * d + neuron
        opt2_u = np.zeros_like(learned_u)
        opt2_u[route * d : (route + 1) * d] = W_T[
            k_opt2, route * d : (route + 1) * d
        ].numpy()

        vmax = float(max(abs(learned_u).max(), abs(opt1_u).max(), abs(opt2_u).max(), 1e-12))
        for col, (label, row_u) in enumerate(
            [("learned U[c,:]", learned_u), ("opt1 U[k,:]", opt1_u), ("opt2 U[idx,:]", opt2_u)]
        ):
            ax = axes[row, col]
            ax.imshow(row_u[None, :], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_yticks([])
            ax.set_xticks([])
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(
                    f"c={c}\nk={k}\noh={one_hotness[c]:.2f}\nbc={block_concentration[c]:.2f}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=7,
                )
    fig.suptitle(f"{layer_name} — learned vs opt1/opt2 U rows", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{layer_name}_sample_heatmaps.png", dpi=120)
    plt.close(fig)


def format_summary(run_path: str, all_layer_results: list[dict[str, Any]]) -> str:
    def _f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    lines = [f"Component diff for {run_path}", "=" * 72, ""]
    for r in all_layer_results:
        lines.append(f"--- {r['module_name']} (C={r['C']}) ---")
        lines.append(f"  #computational (k_c < 64): {r['n_computational']}")
        lines.append(f"  #indexing      (k_c >= 64): {r['n_indexing']}")
        lines.append(f"  #low one_hotness (<0.5):   {r['n_low_one_hotness']}")
        lines.append(f"  mean one_hotness (all):    {r['mean_one_hotness_all']:.4f}")
        lines.append("")
        lines.append("  [computational half]")
        lines.append(
            f"    block_concentration   mean={_f(r['mean_block_concentration_comp'])}"
            f"  #>0.9={r['n_block_concentration_gt_0_9_comp']}"
            f"  #<0.2={r['n_block_concentration_lt_0_2_comp']}"
        )
        lines.append(
            f"    block_entropy (norm)  mean={_f(r['mean_block_entropy_comp'])}"
        )
        lines.append(f"    cosine vs opt1        mean={_f(r['mean_cos1_comp'])}")
        lines.append(
            f"    cosine vs opt2        mean={_f(r['mean_cos2_comp'])}"
            f"  #cos2>cos1={r['n_cos2_gt_cos1_comp']}"
            f"  #cos1>cos2={r['n_cos1_gt_cos2_comp']}"
        )
        lines.append(
            f"    opt1 target coverage  mean={r['opt1_coverage_mean']:.4f}"
            f"  #targets cos>0.9 = {r['opt1_coverage_n_gt_0_9']}/{D}"
        )
        lines.append(
            f"    opt2 target coverage  mean={r['opt2_coverage_mean']:.4f}"
            f"  #targets cos>0.9 = {r['opt2_coverage_n_gt_0_9']}/{N_COMPUTATIONAL}"
        )
        lines.append("")
        lines.append("  [indexing half]")
        lines.append(
            f"    V one_hotness         mean={_f(r['mean_idx_v_one_hotness'])}"
        )
        lines.append(
            f"    U cos(U[c,:],W_T[k,:]) mean={_f(r['mean_idx_u_cos_to_k'])}"
        )
        lines.append(
            f"    idx target coverage   mean={r['idx_coverage_mean']:.4f}"
            f"  #targets cos>0.9 = {r['idx_coverage_n_gt_0_9']}/{N_INDEXING}"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default="wandb:paramluhadiya/spd/s-64ee330f")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()

    run_path = args.run
    if not run_path.startswith("wandb:") and not Path(run_path).exists():
        run_path = "wandb:" + run_path

    run_id = run_path.rstrip("/").rsplit("/", 1)[-1]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else SPD_OUT_DIR / "component_diff" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    print(f"Loading ComponentModel from {run_path} ...")
    model = ComponentModel.from_pretrained(run_path)
    model.eval()

    all_layer_results: list[dict[str, Any]] = []
    full_json: dict[str, Any] = {"run": run_path, "layers": {}}

    for name in LAYER_NAMES:
        assert name in model.components, (
            f"Layer {name} not in model.components (have {list(model.components)})"
        )
        print(f"\n=== {name} ===")
        components = model.components[name]
        V = components.V.detach().cpu().float()
        U = components.U.detach().cpu().float()
        W_T = model.target_weight(name).detach().cpu().float().T
        print(
            f"  V shape={tuple(V.shape)}  U shape={tuple(U.shape)}  W_T shape={tuple(W_T.shape)}"
        )

        layer_data = analyse_layer(name, V, U, W_T)
        all_layer_results.append(layer_data["results"])
        full_json["layers"][name] = {
            "results": layer_data["results"],
            "per_comp": layer_data["per_comp"],
            "coverage": layer_data["coverage"],
        }

        if not args.no_figures:
            save_figures(name, layer_data["figure_tensors"], figures_dir)

    summary = format_summary(run_path, all_layer_results)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)
    (out_dir / "results.json").write_text(json.dumps(full_json, indent=2))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
