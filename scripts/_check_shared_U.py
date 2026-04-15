"""Check U block-sparsity for the 6 shared components at model.0."""

import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
LAYER = "model.0"
SHARED = [292, 533, 373, 444, 552, 19]


def main() -> None:
    model = ComponentModel.from_pretrained("wandb:paramluhadiya/spd/s-64ee330f")
    model.eval()

    U = model.components[LAYER].U.detach().cpu().float()  # (C, 80)
    V = model.components[LAYER].V.detach().cpu().float()  # (80, C)

    print(f"{'comp':>5s}  {'blk_conc':>8s}  {'argmax_blk':>10s}  ", end="")
    for r in range(NUM_BLOCKS):
        print(f"{'b'+str(r):>8s}", end="")
    print(f"  {'idx_mass':>8s}  {'V_argmax':>8s}  {'V_1hot':>8s}")

    for c in SHARED:
        u = U[c, :]
        u_comp = u[:D]  # first 64 dims = computational
        u_idx = u[D:]   # last 16 dims = indexing

        # Per-block L2 mass
        blocks = u_comp.reshape(NUM_BLOCKS, d)
        block_mass = (blocks ** 2).sum(dim=1)
        total_comp = block_mass.sum()
        block_frac = block_mass / total_comp.clamp_min(1e-30)
        block_conc = block_frac.max().item()
        best_block = int(block_frac.argmax())

        # V info
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        v_one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()

        idx_frac = (u_idx ** 2).sum() / ((u ** 2).sum()).clamp_min(1e-30)

        print(f"{c:>5d}  {block_conc:>8.4f}  {best_block:>10d}  ", end="")
        for r in range(NUM_BLOCKS):
            print(f"{block_frac[r].item():>8.4f}", end="")
        print(f"  {idx_frac.item():>8.4f}  {v_argmax:>8d}  {v_one_hotness:>8.4f}")


if __name__ == "__main__":
    main()
