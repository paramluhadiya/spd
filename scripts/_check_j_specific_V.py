"""Check V-sparsity and V block assignment for j-specific components."""

import sys

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


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "wandb:paramluhadiya/spd/s-ee28ad05"
    model = ComponentModel.from_pretrained(path)
    model.eval()
    device = next(model.parameters()).device

    V = model.components[LAYER].V.detach().cpu().float()  # (80, C)
    C = V.shape[1]
    # Auto-detect masks: V one-hot on ohj dims (D+NUM_BLOCKS..D+2*NUM_BLOCKS)
    mask_comps: set[int] = set()
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        if v_argmax >= D + NUM_BLOCKS and one_hotness > 0.95:
            mask_comps.add(c)
    print(f"Auto-detected mask components: {sorted(mask_comps)}")

    # Compute CI for i=0, all j
    ci_per_j: dict[int, torch.Tensor] = {}
    for j_val in range(NUM_BLOCKS):
        x = generate_inputs(0, j_val, N_SAMPLES, device)
        with torch.no_grad():
            ci_out = model.calc_causal_importances({LAYER: x}, sampling="continuous")
            ci_per_j[j_val] = ci_out.lower_leaky[LAYER].mean(dim=0).cpu()

    # Identify shared and j-specific
    shared = set()
    j_specific: dict[int, list[int]] = {j: [] for j in range(NUM_BLOCKS)}
    for c in range(C):
        if c in mask_comps:
            continue
        cis = [ci_per_j[j][c].item() for j in range(NUM_BLOCKS)]
        max_ci = max(cis)
        if max_ci < CI_ACTIVE:
            continue
        if all(ci > CI_ACTIVE for ci in cis):
            shared.add(c)
            continue
        for j in range(NUM_BLOCKS):
            if cis[j] > CI_ACTIVE:
                other_max = max(cis[jj] for jj in range(NUM_BLOCKS) if jj != j)
                if other_max < CI_ACTIVE:
                    j_specific[j].append(c)

    # Analyse V for j-specific components
    print(f"{'j':>2s}  {'comp':>5s}  {'CI_j':>6s}  {'V_argmax':>8s}  {'1hot':>6s}  "
          f"{'comp_frac':>9s}  {'idx_frac':>9s}  {'top3_dims':>30s}  "
          f"{'V_blk_conc':>10s}  {'V_best_blk':>10s}")

    all_one_hotness = []
    all_comp_frac = []
    all_v_blk_conc = []
    all_best_blk_matches_j = []

    for j_val in range(NUM_BLOCKS):
        for c in sorted(j_specific[j_val], key=lambda c: -ci_per_j[j_val][c].item()):
            v = V[:, c]
            v_sq = v ** 2
            total = v_sq.sum()

            v_argmax = int(v.abs().argmax())
            one_hotness = (v_sq[v_argmax] / total).item()

            comp_mass = v_sq[:D].sum().item()
            idx_mass = v_sq[D:].sum().item()
            comp_frac = comp_mass / (comp_mass + idx_mass)
            idx_frac = idx_mass / (comp_mass + idx_mass)

            # Top 3 dims by mass
            top3_vals, top3_idxs = v_sq.topk(3)
            top3_str = ", ".join(f"d{int(idx)}({top3_vals[i]/total:.2f})" for i, idx in enumerate(top3_idxs))

            # V block concentration (over computational dims only)
            v_comp = v[:D].reshape(NUM_BLOCKS, d)
            v_block_mass = (v_comp ** 2).sum(dim=1)
            v_total_comp = v_block_mass.sum()
            if v_total_comp > 1e-30:
                v_block_frac = v_block_mass / v_total_comp
                v_blk_conc = v_block_frac.max().item()
                v_best_blk = int(v_block_frac.argmax())
            else:
                v_blk_conc = 0.0
                v_best_blk = -1

            all_one_hotness.append(one_hotness)
            all_comp_frac.append(comp_frac)
            all_v_blk_conc.append(v_blk_conc)
            all_best_blk_matches_j.append(v_best_blk == j_val)

            print(f"{j_val:>2d}  {c:>5d}  {ci_per_j[j_val][c].item():>6.3f}  {v_argmax:>8d}  "
                  f"{one_hotness:>6.3f}  {comp_frac:>9.4f}  {idx_frac:>9.4f}  {top3_str:>30s}  "
                  f"{v_blk_conc:>10.4f}  {v_best_blk:>10d}")
        print()

    # Summary stats
    n = len(all_one_hotness)
    print(f"{'='*72}")
    print(f"SUMMARY ({n} j-specific components)")
    print(f"  V one-hotness: mean={sum(all_one_hotness)/n:.4f}, "
          f"#>0.9={sum(1 for x in all_one_hotness if x > 0.9)}, "
          f"#>0.5={sum(1 for x in all_one_hotness if x > 0.5)}, "
          f"#>0.3={sum(1 for x in all_one_hotness if x > 0.3)}")
    print(f"  comp_frac (V mass in dims 0:64): mean={sum(all_comp_frac)/n:.4f}, "
          f"#>0.9={sum(1 for x in all_comp_frac if x > 0.9)}, "
          f"#>0.5={sum(1 for x in all_comp_frac if x > 0.5)}")
    print(f"  V block concentration (within comp dims): mean={sum(all_v_blk_conc)/n:.4f}, "
          f"#>0.9={sum(1 for x in all_v_blk_conc if x > 0.9)}, "
          f"#>0.5={sum(1 for x in all_v_blk_conc if x > 0.5)}")
    print(f"  V best block matches j: {sum(all_best_blk_matches_j)}/{n}")


if __name__ == "__main__":
    main()
