"""Check if the 6 shared components are learning bias terms.

Two checks:
1. Do these components fire for other source blocks i≠0? If they're bias for i=0,
   they should NOT fire for i=1,2,...
2. Compare their V@U outer products against the bias column of W_T for i=0.
   The bias column for source block i is W_T[:, D+i] (the ohi column), which
   reads from the one-hot indicator and writes bias into the computational output.
"""

import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
LAYER = "model.0"
SHARED = [292, 533, 373, 444, 552, 19]
N_SAMPLES = 512


def main() -> None:
    model = ComponentModel.from_pretrained("wandb:paramluhadiya/spd/s-64ee330f")
    model.eval()
    device = next(model.parameters()).device

    # === Check 1: Do shared components fire for different i values? ===
    print("=== CI of shared components across source blocks i (j=0) ===")
    print(f"{'i':>3s}", end="")
    for c in SHARED:
        print(f"  c={c:>3d}", end="")
    print()

    for i_val in range(NUM_BLOCKS):
        x = torch.zeros(N_SAMPLES, 80, device=device)
        x[:, i_val * d : (i_val + 1) * d] = torch.rand(N_SAMPLES, d, device=device)
        x[:, D + i_val] = 1.0
        x[:, D + NUM_BLOCKS] = 1.0  # j=0

        with torch.no_grad():
            ci_out = model.calc_causal_importances({LAYER: x}, sampling="continuous")
            ci_vals = ci_out.lower_leaky[LAYER].mean(dim=0).cpu()

        print(f"{i_val:>3d}", end="")
        for c in SHARED:
            print(f"  {ci_vals[c]:>6.4f}", end="")
        print()

    # === Check 2: Compare outer products against bias columns of W_T ===
    print("\n=== Comparing shared components against W_T bias columns ===")

    V = model.components[LAYER].V.detach().cpu().float()  # (80, C)
    U = model.components[LAYER].U.detach().cpu().float()  # (C, 80)
    W_T = model.target_weight(LAYER).detach().cpu().float().T  # (80, 80)

    # Bias columns: W_T[:, D+i] for each i — these are the ohi columns
    # Each reads from ohi[i] and writes bias into the output
    print("\nCosine similarity of each shared component's V⊗U against each bias outer product:")
    print(f"{'comp':>5s}", end="")
    for i_val in range(NUM_BLOCKS):
        print(f"  {'bias_i='+str(i_val):>10s}", end="")
    print()

    for c in SHARED:
        L_c = (V[:, c].unsqueeze(1) * U[c, :].unsqueeze(0)).flatten()  # (6400,)
        L_c_norm = L_c / L_c.norm().clamp_min(1e-30)

        print(f"{c:>5d}", end="")
        for i_val in range(NUM_BLOCKS):
            # Bias target: one-hot on dim D+i, row = W_T[D+i, :]
            bias_target = torch.zeros(80, 80)
            bias_target[D + i_val, :] = W_T[D + i_val, :]
            bt_flat = bias_target.flatten()
            bt_norm = bt_flat / bt_flat.norm().clamp_min(1e-30)
            cos = (L_c_norm * bt_norm).sum().item()
            print(f"  {cos:>10.4f}", end="")
        print()

    # Also: sum of shared components' outer products vs bias column for i=0
    print("\n=== Sum of shared components vs bias column for i=0 ===")
    L_sum = torch.zeros(80, 80)
    for c in SHARED:
        L_sum += V[:, c].unsqueeze(1) * U[c, :].unsqueeze(0)

    bias_0 = torch.zeros(80, 80)
    bias_0[D, :] = W_T[D, :]  # ohi[0] column

    cos_sum = (L_sum.flatten() / L_sum.flatten().norm().clamp_min(1e-30)) @ \
              (bias_0.flatten() / bias_0.flatten().norm().clamp_min(1e-30))
    frob_err = (L_sum - bias_0).norm() / bias_0.norm()
    print(f"  cosine(sum_shared, bias_i=0) = {cos_sum:.4f}")
    print(f"  ||sum - bias_i=0|| / ||bias_i=0|| = {frob_err:.4f}")

    # What fraction of mass is in computational vs indexing output dims?
    print(f"\n  L_sum mass in comp output dims [0:64]: {(L_sum[:, :D] ** 2).sum() / (L_sum ** 2).sum():.4f}")
    print(f"  L_sum mass in idx output dims [64:80]: {(L_sum[:, D:] ** 2).sum() / (L_sum ** 2).sum():.4f}")

    # What fraction of mass is in the D+0 row (reading from ohi[0])?
    print(f"  L_sum mass in row D (ohi[0] input): {(L_sum[D, :] ** 2).sum() / (L_sum ** 2).sum():.4f}")
    print(f"  L_sum mass in rows 0:64 (comp input): {(L_sum[:D, :] ** 2).sum() / (L_sum ** 2).sum():.4f}")
    print(f"  L_sum mass in rows 64:80 (idx input): {(L_sum[D:, :] ** 2).sum() / (L_sum ** 2).sum():.4f}")


if __name__ == "__main__":
    main()
