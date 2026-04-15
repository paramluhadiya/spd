"""Check U block-sparsity for j-specific components at model.0.

For each j, take the components that fire specifically for that j (excluding
the 6 shared/bias components and the mask components 72-79), and check
whether their U vectors write to block j only.
"""

import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
LAYER = "model.0"
MASK_RANGE = set(range(72, 80))
SHARED = {292, 533, 373, 444, 552, 19}
N_SAMPLES = 512
CI_THRESH = 0.3


def main() -> None:
    model = ComponentModel.from_pretrained("wandb:paramluhadiya/spd/s-64ee330f")
    model.eval()
    device = next(model.parameters()).device

    V = model.components[LAYER].V.detach().cpu().float()
    U = model.components[LAYER].U.detach().cpu().float()
    C = V.shape[1]

    exclude = MASK_RANGE | SHARED

    # Compute CI for each j (i=0 fixed)
    ci_per_j: dict[int, torch.Tensor] = {}
    for j_val in range(NUM_BLOCKS):
        x = torch.zeros(N_SAMPLES, 80, device=device)
        x[:, 0:d] = torch.rand(N_SAMPLES, d, device=device)
        x[:, D] = 1.0
        x[:, D + NUM_BLOCKS + j_val] = 1.0

        with torch.no_grad():
            ci_out = model.calc_causal_importances({LAYER: x}, sampling="continuous")
            ci_vals = ci_out.lower_leaky[LAYER].mean(dim=0).cpu()
        ci_per_j[j_val] = ci_vals

    # For each j, find j-specific components: high CI for this j, low for others
    print(f"{'j':>2s}  {'comp':>5s}  {'CI_j':>6s}  {'CI_other':>8s}  {'blk_conc':>8s}  {'best_blk':>8s}  ", end="")
    for r in range(NUM_BLOCKS):
        print(f"{'b' + str(r):>7s}", end="")
    print(f"  {'V_argmax':>8s}  {'V_1hot':>7s}")

    for j_val in range(NUM_BLOCKS):
        ci_j = ci_per_j[j_val]

        # Find components with CI > threshold for this j, excluding masks/shared
        active = []
        for c in range(C):
            if c in exclude:
                continue
            if ci_j[c] > CI_THRESH:
                # Check it's not equally active for all j (would be shared)
                other_cis = [ci_per_j[jj][c].item() for jj in range(NUM_BLOCKS) if jj != j_val]
                max_other = max(other_cis)
                active.append((c, ci_j[c].item(), max_other))

        for c, ci_val, max_other in sorted(active, key=lambda x: -x[1]):
            u = U[c, :]
            u_comp = u[:D].reshape(NUM_BLOCKS, d)
            block_mass = (u_comp ** 2).sum(dim=1)
            total_comp = block_mass.sum()
            block_frac = block_mass / total_comp.clamp_min(1e-30)
            block_conc = block_frac.max().item()
            best_block = int(block_frac.argmax())

            v = V[:, c]
            v_argmax = int(v.abs().argmax())
            v_one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()

            print(f"{j_val:>2d}  {c:>5d}  {ci_val:>6.4f}  {max_other:>8.4f}  {block_conc:>8.4f}  {best_block:>8d}  ", end="")
            for r in range(NUM_BLOCKS):
                print(f"{block_frac[r].item():>7.4f}", end="")
            print(f"  {v_argmax:>8d}  {v_one_hotness:>7.4f}")

        if not active:
            print(f"{j_val:>2d}  (no j-specific components above CI>{CI_THRESH})")
        print()


if __name__ == "__main__":
    main()
