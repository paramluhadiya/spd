"""Comprehensive eval of a PingPong SPD run.

Runs all diagnostics:
1. CI per (i=0, j) circuit — distribution + counts
2. Identify shared vs j-specific components
3. U block-sparsity for shared components (bias hypothesis)
4. U block-sparsity for j-specific components (per-route writing)
5. CI across source blocks i for shared components (i-specificity)
6. Cosine similarity of shared components against W_T bias columns
7. Repeat CI analysis for a second source block (i=3)

Usage:
    python scripts/_eval_run.py /path/to/checkpoint
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
N_SAMPLES = 512

# Mask component ranges per layer
MASK_RANGE: dict[str, set[int]] = {
    "model.0": set(range(72, 80)),   # ohj masks (old 64-comp layout)
    "model.2": set(range(64, 72)),   # ohi masks
    "model.4": set(range(72, 80)),   # ohj masks
}

LAYER = "model.0"


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
    path = sys.argv[1] if len(sys.argv) > 1 else "wandb:paramluhadiya/spd/s-ee28ad05"
    print(f"Loading {path} ...")
    model = ComponentModel.from_pretrained(path)
    model.eval()
    device = next(model.parameters()).device

    C = model.components[LAYER].V.shape[1]
    V = model.components[LAYER].V.detach().cpu().float()
    U = model.components[LAYER].U.detach().cpu().float()
    W_T = model.target_weight(LAYER).detach().cpu().float().T

    # Detect mask range: check if C > 80 (new 512-comp layout) or old 64-comp layout
    # For new layout, masks are at N_COMPUTATIONAL..N_COMPUTATIONAL+16
    # For old layout, masks are at 64..80
    # Heuristic: check V for components that are clean one-hots on indexing dims
    print(f"C = {C}")

    # Try to auto-detect mask components by checking which have V one-hot on indexing dims
    # and CI ≈ 1.0 for matched j
    # For simplicity, just check both possible ranges
    mask_comps: set[int] = set()
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        if v_argmax >= D + NUM_BLOCKS and one_hotness > 0.95:
            # Likely an ohj mask
            mask_comps.add(c)
        elif v_argmax >= D and v_argmax < D + NUM_BLOCKS and one_hotness > 0.95:
            # Could be ohi bias or mask - mark for now
            pass
    print(f"Auto-detected {len(mask_comps)} likely ohj mask components: {sorted(mask_comps)}")

    # Also manually check the known ranges
    for label, rng in [("64-71", range(64, 72)), ("72-79", range(72, 80)),
                       ("512-519", range(512, min(520, C))), ("520-527", range(520, min(528, C)))]:
        if rng.stop <= C:
            vals = []
            for c in rng:
                v = V[:, c]
                v_argmax = int(v.abs().argmax())
                one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
                vals.append(f"c={c}:dim{v_argmax}({one_hotness:.3f})")
            print(f"  Range {label}: {', '.join(vals)}")

    # ========================================================================
    # 1. CI per (i=0, j) circuit
    # ========================================================================
    print(f"\n{'='*72}")
    print("1. CI DISTRIBUTION PER (i=0, j) CIRCUIT")
    print(f"{'='*72}")

    ci_per_j: dict[int, torch.Tensor] = {}
    for j_val in range(NUM_BLOCKS):
        x = generate_inputs(0, j_val, N_SAMPLES, device)
        ci_per_j[j_val] = compute_ci(model, LAYER, x)

    non_mask = torch.ones(C, dtype=torch.bool)
    for c in mask_comps:
        non_mask[c] = False

    for j_val in range(NUM_BLOCKS):
        ci_j = ci_per_j[j_val][non_mask]
        print(f"\n  j={j_val}:")
        for thresh in [0.9, 0.8, 0.5, 0.3, 0.1, 0.01]:
            print(f"    #CI > {thresh}: {int((ci_j > thresh).sum())}")

        # Top 15 overall
        ci_all = ci_per_j[j_val].clone()
        top_vals, top_idxs = ci_all.topk(min(15, C))
        top_str = "  ".join(
            f"c={int(idx)}({'M' if int(idx) in mask_comps else ''}{ci_all[idx]:.3f})"
            for val, idx in zip(top_vals, top_idxs) if val > 0.01
        )
        print(f"    Top: {top_str}")

    # ========================================================================
    # 2. Identify shared vs j-specific components
    # ========================================================================
    print(f"\n{'='*72}")
    print("2. SHARED VS J-SPECIFIC COMPONENTS")
    print(f"{'='*72}")

    CI_ACTIVE = 0.3
    CI_SHARED_THRESH = 0.3  # must be > this for ALL j to be shared

    # For each non-mask component, check if it's active for all j or specific j's
    shared_comps = set()
    j_specific: dict[int, list[int]] = {j: [] for j in range(NUM_BLOCKS)}

    for c in range(C):
        if c in mask_comps:
            continue
        cis = [ci_per_j[j][c].item() for j in range(NUM_BLOCKS)]
        max_ci = max(cis)
        if max_ci < CI_ACTIVE:
            continue

        n_active = sum(1 for ci in cis if ci > CI_SHARED_THRESH)
        if n_active == NUM_BLOCKS:
            shared_comps.add(c)
        else:
            for j in range(NUM_BLOCKS):
                if cis[j] > CI_ACTIVE:
                    # Check it's not equally active elsewhere
                    other_max = max(cis[jj] for jj in range(NUM_BLOCKS) if jj != j)
                    if other_max < CI_ACTIVE:
                        j_specific[j].append(c)

    print(f"\n  Shared components (CI > {CI_SHARED_THRESH} for all j): {len(shared_comps)}")
    print(f"    {sorted(shared_comps)}")
    for j in range(NUM_BLOCKS):
        print(f"  j={j} specific: {len(j_specific[j])}  {j_specific[j]}")

    # Components active for some but not cleanly j-specific
    all_classified = shared_comps.copy()
    for j in range(NUM_BLOCKS):
        all_classified.update(j_specific[j])

    ambiguous = []
    for c in range(C):
        if c in mask_comps or c in all_classified:
            continue
        max_ci = max(ci_per_j[j][c].item() for j in range(NUM_BLOCKS))
        if max_ci > CI_ACTIVE:
            ambiguous.append(c)
    if ambiguous:
        print(f"  Ambiguous (active but not cleanly shared or j-specific): {len(ambiguous)}")
        for c in ambiguous[:10]:
            cis = [ci_per_j[j][c].item() for j in range(NUM_BLOCKS)]
            print(f"    c={c}: {['%.3f' % ci for ci in cis]}")

    # ========================================================================
    # 3. U block-sparsity for shared components
    # ========================================================================
    print(f"\n{'='*72}")
    print("3. U BLOCK-SPARSITY FOR SHARED COMPONENTS")
    print(f"{'='*72}")

    print(f"\n  {'comp':>5s}  {'blk_conc':>8s}  {'best_blk':>8s}  ", end="")
    for r in range(NUM_BLOCKS):
        print(f"{'b'+str(r):>7s}", end="")
    print(f"  {'V_argmax':>8s}  {'V_1hot':>7s}")

    for c in sorted(shared_comps):
        u = U[c, :]
        u_comp = u[:D].reshape(NUM_BLOCKS, d)
        block_mass = (u_comp ** 2).sum(dim=1)
        total_comp = block_mass.sum()
        block_frac = block_mass / total_comp.clamp_min(1e-30)

        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        v_1hot = (v[v_argmax] ** 2 / (v ** 2).sum()).item()

        print(f"  {c:>5d}  {block_frac.max().item():>8.4f}  {int(block_frac.argmax()):>8d}  ", end="")
        for r in range(NUM_BLOCKS):
            print(f"{block_frac[r].item():>7.4f}", end="")
        print(f"  {v_argmax:>8d}  {v_1hot:>7.4f}")

    # ========================================================================
    # 4. U block-sparsity for j-specific components
    # ========================================================================
    print(f"\n{'='*72}")
    print("4. U BLOCK-SPARSITY FOR J-SPECIFIC COMPONENTS")
    print(f"{'='*72}")

    print(f"\n  {'j':>2s}  {'comp':>5s}  {'CI_j':>6s}  {'blk_conc':>8s}  {'best_blk':>8s}  ", end="")
    for r in range(NUM_BLOCKS):
        print(f"{'b'+str(r):>7s}", end="")
    print(f"  {'V_argmax':>8s}  {'V_1hot':>7s}")

    for j_val in range(NUM_BLOCKS):
        for c in sorted(j_specific[j_val], key=lambda c: -ci_per_j[j_val][c].item()):
            u = U[c, :]
            u_comp = u[:D].reshape(NUM_BLOCKS, d)
            block_mass = (u_comp ** 2).sum(dim=1)
            total_comp = block_mass.sum()
            block_frac = block_mass / total_comp.clamp_min(1e-30)

            v = V[:, c]
            v_argmax = int(v.abs().argmax())
            v_1hot = (v[v_argmax] ** 2 / (v ** 2).sum()).item()

            ci_j = ci_per_j[j_val][c].item()
            print(f"  {j_val:>2d}  {c:>5d}  {ci_j:>6.4f}  {block_frac.max().item():>8.4f}  {int(block_frac.argmax()):>8d}  ", end="")
            for r in range(NUM_BLOCKS):
                print(f"{block_frac[r].item():>7.4f}", end="")
            print(f"  {v_argmax:>8d}  {v_1hot:>7.4f}")
        print()

    # ========================================================================
    # 5. Shared components: CI across source blocks i
    # ========================================================================
    print(f"\n{'='*72}")
    print("5. SHARED COMPONENTS: CI ACROSS SOURCE BLOCKS i (j=0)")
    print(f"{'='*72}")

    print(f"\n  {'i':>3s}", end="")
    for c in sorted(shared_comps):
        print(f"  c={c:>3d}", end="")
    print()

    for i_val in range(NUM_BLOCKS):
        x = generate_inputs(i_val, 0, N_SAMPLES, device)
        ci_vals = compute_ci(model, LAYER, x)
        print(f"  {i_val:>3d}", end="")
        for c in sorted(shared_comps):
            print(f"  {ci_vals[c]:>6.4f}", end="")
        print()

    # ========================================================================
    # 6. Cosine similarity against W_T bias columns
    # ========================================================================
    print(f"\n{'='*72}")
    print("6. SHARED COMPONENTS: COSINE SIM AGAINST W_T BIAS COLUMNS")
    print(f"{'='*72}")

    print(f"\n  {'comp':>5s}", end="")
    for i_val in range(NUM_BLOCKS):
        print(f"  {'bias_i='+str(i_val):>10s}", end="")
    print()

    for c in sorted(shared_comps):
        L_c = (V[:, c].unsqueeze(1) * U[c, :].unsqueeze(0)).flatten()
        L_c_norm = L_c / L_c.norm().clamp_min(1e-30)

        print(f"  {c:>5d}", end="")
        for i_val in range(NUM_BLOCKS):
            bias_target = torch.zeros(80, 80)
            bias_target[D + i_val, :] = W_T[D + i_val, :]
            bt_flat = bias_target.flatten()
            bt_norm = bt_flat / bt_flat.norm().clamp_min(1e-30)
            cos = (L_c_norm * bt_norm).sum().item()
            print(f"  {cos:>10.4f}", end="")
        print()

    # ========================================================================
    # 7. Repeat for i=3
    # ========================================================================
    print(f"\n{'='*72}")
    print("7. CI DISTRIBUTION FOR (i=3, j) CIRCUITS")
    print(f"{'='*72}")

    ci_per_j_i3: dict[int, torch.Tensor] = {}
    for j_val in range(NUM_BLOCKS):
        x = generate_inputs(3, j_val, N_SAMPLES, device)
        ci_per_j_i3[j_val] = compute_ci(model, LAYER, x)

    shared_i3 = set()
    j_specific_i3: dict[int, list[int]] = {j: [] for j in range(NUM_BLOCKS)}

    for c in range(C):
        if c in mask_comps:
            continue
        cis = [ci_per_j_i3[j][c].item() for j in range(NUM_BLOCKS)]
        max_ci = max(cis)
        if max_ci < CI_ACTIVE:
            continue
        n_active = sum(1 for ci in cis if ci > CI_SHARED_THRESH)
        if n_active == NUM_BLOCKS:
            shared_i3.add(c)
        else:
            for j in range(NUM_BLOCKS):
                if cis[j] > CI_ACTIVE:
                    other_max = max(cis[jj] for jj in range(NUM_BLOCKS) if jj != j)
                    if other_max < CI_ACTIVE:
                        j_specific_i3[j].append(c)

    print(f"\n  Shared for i=3: {len(shared_i3)}  {sorted(shared_i3)}")
    for j in range(NUM_BLOCKS):
        print(f"  j={j} specific: {len(j_specific_i3[j])}  {j_specific_i3[j]}")

    # Check overlap between i=0 and i=3 shared/j-specific
    print(f"\n  Overlap shared i=0 ∩ i=3: {sorted(shared_comps & shared_i3)}")
    for j in range(NUM_BLOCKS):
        overlap = set(j_specific[j]) & set(j_specific_i3[j])
        if overlap:
            print(f"  Overlap j-specific j={j} i=0 ∩ i=3: {sorted(overlap)}")

    # ========================================================================
    # 8. Plot CI distributions
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"CI distribution for (i=0, j=*) circuits at {LAYER}\n(excluding mask components)", fontsize=14)

    for j_val in range(NUM_BLOCKS):
        ci_j = ci_per_j[j_val].clone()
        for c in mask_comps:
            ci_j[c] = -1
        ci_clean = ci_j[ci_j >= 0]
        sorted_ci, _ = ci_clean.sort(descending=True)

        ax = axes[j_val // 4, j_val % 4]
        n_show = min(60, len(sorted_ci))
        colors = ["#d62728" if v > 0.5 else "#1f77b4" if v > 0.1 else "#cccccc" for v in sorted_ci[:n_show]]
        ax.bar(range(n_show), sorted_ci[:n_show].numpy(), color=colors)
        n_above_half = int((ci_clean > 0.5).sum())
        n_above_01 = int((ci_clean > 0.1).sum())
        ax.set_title(f"j={j_val}  (#>0.5: {n_above_half}, #>0.1: {n_above_01})")
        ax.set_ylabel("Mean CI")
        ax.set_xlabel("Component rank")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("ci_eval_i0.png", dpi=150)
    print("\nSaved ci_eval_i0.png")

    # i=3 plot
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle(f"CI distribution for (i=3, j=*) circuits at {LAYER}\n(excluding mask components)", fontsize=14)

    for j_val in range(NUM_BLOCKS):
        ci_j = ci_per_j_i3[j_val].clone()
        for c in mask_comps:
            ci_j[c] = -1
        ci_clean = ci_j[ci_j >= 0]
        sorted_ci, _ = ci_clean.sort(descending=True)

        ax = axes2[j_val // 4, j_val % 4]
        n_show = min(60, len(sorted_ci))
        colors = ["#d62728" if v > 0.5 else "#1f77b4" if v > 0.1 else "#cccccc" for v in sorted_ci[:n_show]]
        ax.bar(range(n_show), sorted_ci[:n_show].numpy(), color=colors)
        n_above_half = int((ci_clean > 0.5).sum())
        n_above_01 = int((ci_clean > 0.1).sum())
        ax.set_title(f"j={j_val}  (#>0.5: {n_above_half}, #>0.1: {n_above_01})")
        ax.set_ylabel("Mean CI")
        ax.set_xlabel("Component rank")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("ci_eval_i3.png", dpi=150)
    print("Saved ci_eval_i3.png")


if __name__ == "__main__":
    main()
