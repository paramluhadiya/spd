"""Track how V vectors evolve from neuron basis to distributed directions.

At init, V[:,k] = e_k (one-hot). If the model learns SVD-like directions,
each V vector should eventually have all 8 dims active.

We measure:
1. L0.1: count of entries with |v_i| > 0.1 * ||v|| (how many dims are "on")
   One-hot: L0.1 = 1.  Uniform: L0.1 = 8.

2. Singular value spectrum of M_i (the 64x8 block matrix for source block i).
   If singular values are degenerate, SVD directions aren't unique.

3. Subspace alignment: does the span of top-k learned V vectors match
   the span of top-k SVD right singular vectors?
"""

import torch

from spd.models.component_model import ComponentModel, SPDRunInfo

D = 64
d = 8
NUM_BLOCKS = 8


def l0_threshold(v: torch.Tensor, frac: float = 0.1) -> int:
    """Count entries with |v_i| > frac * ||v||."""
    threshold = frac * v.norm().item()
    return (v.abs() > threshold).sum().item()


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Compute mean cos^2(principal angles) between column spans of A and B.

    Args:
        A: (n, k) matrix
        B: (n, k) matrix
    Returns:
        Mean cos^2 of principal angles. 1.0 = identical subspaces.
    """
    Q_a, _ = torch.linalg.qr(A)
    Q_b, _ = torch.linalg.qr(B)
    svals = torch.linalg.svdvals(Q_a.T @ Q_b)
    return svals.pow(2).mean().item()


@torch.no_grad()
def main() -> None:
    checkpoints = [
        (step, f"wandb/s-e2e8fb27/files/model_{step}.pth")
        for step in range(10000, 110000, 10000)
    ]

    # First: check singular value spectrum of M_i (doesn't change over training)
    run_info = SPDRunInfo.from_path(checkpoints[0][1])
    model = ComponentModel.from_run_info(run_info)
    target_model = model.target_model

    print("=" * 70)
    print("Singular value spectrum of M_i (target model, fixed)")
    print("=" * 70)

    for layer_name, layer_idx in [("model.0", 0), ("model.2", 1), ("model.4", 2)]:
        W = dict(target_model.named_modules())[layer_name].weight.detach()
        W_cc = W[:D, :D]

        print(f"\n  {layer_name}:")
        for src_block in range(NUM_BLOCKS):
            src_s = d * src_block
            M_i = W_cc[:, src_s : src_s + d]
            _, S, _ = torch.linalg.svd(M_i, full_matrices=False)
            ratio = S[0].item() / S[-1].item()
            s_str = " ".join([f"{s:.2f}" for s in S.tolist()])
            print(f"    block {src_block}: σ = [{s_str}]  ratio σ1/σ8 = {ratio:.2f}")

    # Now track V distribution and subspace alignment over training
    for layer_name, layer_idx in [("model.0", 0), ("model.2", 1), ("model.4", 2)]:
        W = dict(target_model.named_modules())[layer_name].weight.detach()
        W_cc = W[:D, :D]

        print(f"\n{'='*70}")
        print(f"Layer: {layer_name}")
        print(f"{'='*70}")
        print(f"  {'Step':>6s} | {'mean L0.1':>8s} | ", end="")
        for src_block in [0, 2, 7]:
            print(f"b{src_block} L0.1 | ", end="")
        for k in [2, 3, 4]:
            print(f"b0 sub{k} | ", end="")
            print(f"b2 sub{k} | ", end="")
            print(f"b7 sub{k} | ", end="")
        print()

        for step, ckpt_path in checkpoints:
            run_info = SPDRunInfo.from_path(ckpt_path)
            model = ComponentModel.from_run_info(run_info)
            model.eval()

            comp = model.components[layer_name]
            V = comp.V.detach()

            # L0.1 per block
            all_l0 = []
            block_l0 = {}
            for src_block in range(NUM_BLOCKS):
                src_s, src_e = d * src_block, d * (src_block + 1)
                comp_start = src_block * d
                V_src = V[src_s:src_e, comp_start : comp_start + d]

                block_vals = []
                for k_idx in range(d):
                    v = V_src[:, k_idx]
                    if v.norm().item() < 1e-10:
                        continue
                    l0 = l0_threshold(v)
                    block_vals.append(l0)
                    all_l0.append(l0)
                block_l0[src_block] = sum(block_vals) / len(block_vals) if block_vals else 0

            mean_l0 = sum(all_l0) / len(all_l0)

            # Subspace alignment for select blocks
            sub_results = {}
            for src_block in [0, 2, 7]:
                src_s = d * src_block
                M_i = W_cc[:, src_s : src_s + d]
                _, _, Vh = torch.linalg.svd(M_i, full_matrices=False)

                comp_start = src_block * d
                V_src = V[src_s : src_s + d, comp_start : comp_start + d]

                # Sort learned V by norm
                v_norms = V_src.norm(dim=0)
                sorted_idx = v_norms.argsort(descending=True)

                for k in [2, 3, 4]:
                    svd_basis = Vh[:k, :].T  # (8, k)
                    learned_basis = V_src[:, sorted_idx[:k]]  # (8, k)
                    overlap = subspace_overlap(svd_basis, learned_basis)
                    sub_results[(src_block, k)] = overlap

            # Print row
            print(f"  {step:6d} | {mean_l0:8.2f} | ", end="")
            for src_block in [0, 2, 7]:
                print(f"{block_l0[src_block]:6.2f} | ", end="")
            for k in [2, 3, 4]:
                for src_block in [0, 2, 7]:
                    print(f"  {sub_results[(src_block, k)]:.3f} | ", end="")
            print()


if __name__ == "__main__":
    main()
