"""CI investigation excluding pre-initialized masking components.

For model.0/model.4: exclude c=72-79 (ohj masks)
For model.2: exclude c=64-71 (ohi masks)
"""

import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8

# Masking component ranges per layer
MASK_RANGE: dict[str, range] = {
    "model.0": range(72, 80),  # ohj
    "model.2": range(64, 72),  # ohi
    "model.4": range(72, 80),  # ohj
}


def generate_block_inputs(
    block_b: int, n_samples: int, device: torch.device
) -> torch.Tensor:
    x = torch.zeros(n_samples, 80, device=device)
    x[:, block_b * d : (block_b + 1) * d] = torch.rand(n_samples, d, device=device)
    x[:, D + block_b] = 1.0
    for s in range(n_samples):
        x[s, D + NUM_BLOCKS + (s % NUM_BLOCKS)] = 1.0
    return x


def main() -> None:
    model = ComponentModel.from_pretrained("wandb:paramluhadiya/spd/s-64ee330f")
    model.eval()
    device = next(model.parameters()).device

    for layer_name in ["model.0", "model.2", "model.4"]:
        C = model.components[layer_name].V.shape[1]
        mask_range = MASK_RANGE[layer_name]
        non_mask = torch.ones(C, dtype=torch.bool)
        non_mask[mask_range] = False

        V = model.components[layer_name].V.detach().cpu().float()
        comp_mass = (V[:D, :] ** 2).sum(dim=0)
        idx_mass = (V[D:, :] ** 2).sum(dim=0)
        comp_frac = comp_mass / (comp_mass + idx_mass).clamp_min(1e-30)

        # V block assignment
        V_blocks = V[:D, :].reshape(NUM_BLOCKS, d, C)
        block_mass = (V_blocks ** 2).sum(dim=1)  # (8, C)
        total_comp_mass = block_mass.sum(dim=0)
        V_block_conc = block_mass.max(dim=0).values / total_comp_mass.clamp_min(1e-30)
        best_block = block_mass.argmax(dim=0)

        print(f"\n{'='*72}")
        print(f"=== {layer_name} (C={C}, excluding mask comps {mask_range}) ===")
        print(f"{'='*72}")

        # Generate inputs for each block and compute CI
        ci_per_block: dict[int, torch.Tensor] = {}
        for b in range(NUM_BLOCKS):
            x_b = generate_block_inputs(b, 256, device)
            with torch.no_grad():
                ci_out = model.calc_causal_importances(
                    {layer_name: x_b}, sampling="continuous"
                )
                ci_vals = ci_out.lower_leaky[layer_name].mean(dim=0).cpu()
            ci_per_block[b] = ci_vals

        # CI distribution per block (excluding masks)
        print(f"\n  CI distribution per block (excluding mask comps):")
        print(f"    {'b':>2s}  {'#CI>0.9':>8s}  {'#CI>0.5':>8s}  {'#CI>0.3':>8s}  {'#CI>0.1':>8s}  {'#CI>0.01':>9s}")
        for b in range(NUM_BLOCKS):
            ci_b = ci_per_block[b][non_mask]
            print(
                f"    {b:>2d}  "
                f"{int((ci_b > 0.9).sum()):>8d}  "
                f"{int((ci_b > 0.5).sum()):>8d}  "
                f"{int((ci_b > 0.3).sum()):>8d}  "
                f"{int((ci_b > 0.1).sum()):>8d}  "
                f"{int((ci_b > 0.01).sum()):>9d}"
            )

        # Top 20 CI components per block (excluding masks)
        for b in [0, 3]:
            ci_b = ci_per_block[b].clone()
            ci_b[~non_mask] = -1.0  # exclude masks
            top_vals, top_idxs = ci_b.topk(20)

            print(f"\n  Top 20 non-mask components for block {b}:")
            print(f"    {'c':>5s}  {'CI':>8s}  {'V_block':>7s}  {'bconc':>6s}  {'comp_frac':>9s}")
            for val, idx in zip(top_vals, top_idxs):
                c = int(idx)
                print(
                    f"    {c:>5d}  {val:>8.4f}  {int(best_block[c]):>7d}  "
                    f"{V_block_conc[c]:>6.3f}  {comp_frac[c]:>9.3f}"
                )

        # Per-block: generate circuit-specific inputs (fix both i and j) and check CI
        print(f"\n  CI for specific circuits (i, j) — top 10 non-mask comps per circuit:")
        for i_val, j_val in [(0, 0), (0, 3), (3, 0), (3, 3)]:
            x = torch.zeros(64, 80, device=device)
            x[:, i_val * d : (i_val + 1) * d] = torch.rand(64, d, device=device)
            x[:, D + i_val] = 1.0
            x[:, D + NUM_BLOCKS + j_val] = 1.0
            with torch.no_grad():
                ci_out = model.calc_causal_importances(
                    {layer_name: x}, sampling="continuous"
                )
                ci_vals = ci_out.lower_leaky[layer_name].mean(dim=0).cpu()

            ci_vals[~non_mask] = -1.0
            top_vals, top_idxs = ci_vals.topk(10)
            comps_str = "  ".join(
                f"c={int(idx)}({val:.3f})" for val, idx in zip(top_vals, top_idxs) if val > 0.01
            )
            print(f"    circuit (i={i_val},j={j_val}): {comps_str}")

        # Check: do the top CI components change depending on j?
        print(f"\n  Top 5 non-mask comps for block i=0, varying j:")
        print(f"    {'j':>3s}  {'top 5 (c:CI)':60s}")
        for j_val in range(NUM_BLOCKS):
            x = torch.zeros(64, 80, device=device)
            x[:, 0:d] = torch.rand(64, d, device=device)
            x[:, D] = 1.0  # i=0
            x[:, D + NUM_BLOCKS + j_val] = 1.0
            with torch.no_grad():
                ci_out = model.calc_causal_importances(
                    {layer_name: x}, sampling="continuous"
                )
                ci_vals = ci_out.lower_leaky[layer_name].mean(dim=0).cpu()
            ci_vals[~non_mask] = -1.0
            top_vals, top_idxs = ci_vals.topk(5)
            row = "  ".join(f"{int(idx)}:{val:.3f}" for val, idx in zip(top_vals, top_idxs))
            print(f"    {j_val:>3d}  {row}")


if __name__ == "__main__":
    main()
