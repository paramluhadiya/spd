"""Analyze effective weight matrix vs target for a specific PingPong circuit.

For circuit (i, j) at layer 0, the relevant input dimensions are:
  - Computational: dims d*i..d*(i+1) (8 dims from block i)
  - one_hot_i[i]: dim D+i
  - one_hot_j[j]: dim D+num_blocks+j

This script computes the CI-weighted effective matrix W_eff = sum(CI_k * V[:,k] * U[k,:])
and compares it column-by-column against the target weight matrix W_target.

Usage:
    python scripts/circuit_weight_analysis.py <checkpoint_path> --circuit 0,0
"""

import argparse

import torch

from spd.experiments.tms.bss_models import PingPongModel
from spd.models.component_model import ComponentModel, SPDRunInfo

D = 64
d = 8
NUM_BLOCKS = 8
N_COMPUTATIONAL = 64
N_ONE_HOT_I = 8
N_ONE_HOT_J = 8
N_TRUE = N_COMPUTATIONAL + N_ONE_HOT_I + N_ONE_HOT_J  # 80
INPUT_DIM = D + 2 * NUM_BLOCKS  # 80


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--circuit", type=str, default="0,0", help="i,j")
    parser.add_argument("--layer", type=str, default="model.0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_samples", type=int, default=2048)
    args = parser.parse_args()

    ci_idx, cj_idx = [int(x) for x in args.circuit.split(",")]

    run_info = SPDRunInfo.from_path(args.checkpoint)
    model = ComponentModel.from_run_info(run_info)
    model.to(args.device)
    model.eval()

    target_model = model.target_model
    assert isinstance(target_model, PingPongModel)

    # Target weight matrix for this layer
    layer_name = args.layer
    target_linear = dict(target_model.named_modules())[layer_name]
    W_target = target_linear.weight.detach()  # (d_out, d_in)
    b_target = target_linear.bias.detach() if target_linear.bias is not None else None

    # Component matrices
    comp = model.components[layer_name]
    V = comp.V.detach()  # (d_in, C)
    U = comp.U.detach()  # (C, d_out)
    C = V.shape[1]

    # Compute CI for this circuit
    batch = _make_circuit_batch(ci_idx, cj_idx, args.n_samples, args.device)
    output = model(batch, cache_type="input")
    ci = model.calc_causal_importances(output.cache, sampling="continuous")
    ci_vals = ci.lower_leaky[layer_name].mean(dim=0)  # (C,)

    # W_eff = sum_k CI_k * (V[:,k] outer U[k,:]) = V @ diag(CI) @ U
    ci_diag = torch.diag(ci_vals)
    W_eff = (V @ ci_diag @ U).T  # (d_out, d_in)

    # Delta
    W_delta = W_target - (V @ U).T
    delta_norm = W_delta.norm().item()
    target_norm = W_target.norm().item()

    print(f"Circuit ({ci_idx}, {cj_idx}), Layer: {layer_name}")
    print(f"C = {C}, N_TRUE = {N_TRUE}")
    print(f"Delta norm: {delta_norm:.4f} ({delta_norm/target_norm*100:.2f}% of target)")
    print()

    # Relevant input columns for circuit (i, j)
    comp_cols = list(range(d * ci_idx, d * (ci_idx + 1)))
    ohi_col = D + ci_idx
    ohj_col = D + NUM_BLOCKS + cj_idx

    relevant_cols = comp_cols + [ohi_col, ohj_col]
    col_labels = [f"comp[{c}]" for c in comp_cols] + [f"ohi[{ci_idx}]", f"ohj[{cj_idx}]"]

    print("=" * 80)
    print("Per-column analysis (only columns with nonzero input for this circuit):")
    print("=" * 80)

    total_target_sq = 0.0
    total_error_sq = 0.0

    for col, label in zip(relevant_cols, col_labels):
        w_tgt = W_target[:, col]
        w_eff = W_eff[:, col]
        error = w_tgt - w_eff

        col_target_norm = w_tgt.norm().item()
        col_eff_norm = w_eff.norm().item()
        col_error_norm = error.norm().item()
        rel_error = col_error_norm / col_target_norm * 100 if col_target_norm > 1e-8 else 0.0

        total_target_sq += col_target_norm ** 2
        total_error_sq += col_error_norm ** 2

        print(f"  {label:>12s}: |W_target|={col_target_norm:8.4f}  "
              f"|W_eff|={col_eff_norm:8.4f}  |error|={col_error_norm:8.4f}  "
              f"rel_err={rel_error:5.1f}%")

    print(f"\n  Combined: |target|={total_target_sq**0.5:.4f}  "
          f"|error|={total_error_sq**0.5:.4f}  "
          f"rel_err={total_error_sq**0.5 / total_target_sq**0.5 * 100:.1f}%")

    # Show which components contribute to each column
    print()
    print("=" * 80)
    print("CI values for ground-truth components of this circuit:")
    print("=" * 80)

    # Expected computational components: block i for layers 0,2; block j for layer 1
    layer_idx = {"model.0": 0, "model.2": 1, "model.4": 2}[layer_name]
    expected_block = ci_idx if layer_idx in (0, 2) else cj_idx
    expected_comp_range = range(expected_block * d, (expected_block + 1) * d)

    print(f"  Expected computational block: {expected_block} (components {expected_comp_range.start}-{expected_comp_range.stop-1})")
    for k in expected_comp_range:
        print(f"    comp {k:3d}: CI = {ci_vals[k].item():.4f}")

    ohi_comp = N_COMPUTATIONAL + ci_idx
    ohj_comp = N_COMPUTATIONAL + N_ONE_HOT_I + cj_idx
    print(f"  one_hot_i[{ci_idx}] = comp {ohi_comp}: CI = {ci_vals[ohi_comp].item():.4f}")
    print(f"  one_hot_j[{cj_idx}] = comp {ohj_comp}: CI = {ci_vals[ohj_comp].item():.4f}")

    # Show all components with CI > 0.01
    print()
    active = [(k, ci_vals[k].item()) for k in range(C) if ci_vals[k].item() > 0.01]
    active.sort(key=lambda x: -x[1])
    print(f"All components with CI > 0.01: {len(active)}")
    for k, ci_val in active:
        if k < N_COMPUTATIONAL:
            block = k // d
            label = f"comp (block {block}, neuron {k % d})"
        elif k < N_COMPUTATIONAL + N_ONE_HOT_I:
            label = f"one_hot_i[{k - N_COMPUTATIONAL}]"
        elif k < N_TRUE:
            label = f"one_hot_j[{k - N_COMPUTATIONAL - N_ONE_HOT_I}]"
        else:
            label = f"extra[{k}]"
        print(f"    {k:3d} ({label:>30s}): CI = {ci_val:.4f}")

    # Actual output comparison on this circuit's data
    print()
    print("=" * 80)
    print("Output MSE breakdown:")
    print("=" * 80)

    target_out = output.output  # (n_samples, d_out)
    # Compute output with mask=CI (worst case for PGD with adv=0)
    # W_eff @ x + b vs W_target @ x + b
    # The difference is (W_target - W_eff) @ x
    error_matrix = W_target - W_eff
    output_error = batch @ error_matrix.T  # (n_samples, d_out)
    mse = (output_error ** 2).mean().item()
    target_var = (target_out ** 2).mean().item()
    print(f"  MSE(W_eff vs W_target on circuit data): {mse:.6f}")
    print(f"  Target output variance: {target_var:.6f}")
    print(f"  Relative MSE: {mse/target_var*100:.2f}%")

    # Decompose MSE by source: computational error vs indexing error
    # Error from computational columns
    comp_error = torch.zeros_like(batch)
    for col in comp_cols:
        comp_error[:, :] += batch[:, col:col+1] * error_matrix[:, col:col+1].T
    # Hmm this isn't right, let me do it properly
    comp_input = torch.zeros_like(batch)
    comp_input[:, comp_cols[0]:comp_cols[-1]+1] = batch[:, comp_cols[0]:comp_cols[-1]+1]
    idx_input = torch.zeros_like(batch)
    idx_input[:, ohi_col] = batch[:, ohi_col]
    idx_input[:, ohj_col] = batch[:, ohj_col]

    comp_output_error = comp_input @ error_matrix.T
    idx_output_error = idx_input @ error_matrix.T

    comp_mse = (comp_output_error ** 2).mean().item()
    idx_mse = (idx_output_error ** 2).mean().item()

    print(f"\n  MSE from computational columns only: {comp_mse:.6f}")
    print(f"  MSE from indexing columns only: {idx_mse:.6f}")
    print("  (Cross terms account for the rest)")

    # Also show: what does the target output look like?
    comp_target_out = comp_input @ W_target.T
    idx_target_out = idx_input @ W_target.T
    if b_target is not None:
        idx_target_out += b_target
    print(f"\n  Target output variance from comp input: {(comp_target_out**2).mean().item():.6f}")
    print(f"  Target output variance from idx input: {(idx_target_out**2).mean().item():.6f}")


def _make_circuit_batch(i: int, j: int, n_samples: int, device: str) -> torch.Tensor:
    x = torch.zeros(n_samples, INPUT_DIM, device=device)
    x[:, d * i : d * (i + 1)] = torch.randn(n_samples, d, device=device).abs()
    x[:, D + i] = 1.0
    x[:, D + NUM_BLOCKS + j] = 1.0
    return x


if __name__ == "__main__":
    main()
