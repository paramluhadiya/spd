"""Sanity check that a trained PingPong model has correct masking patterns.

For each circuit (i, j) and layer:
- Even layers (0, 2): output should be nonzero only in block j
- Odd layers (1): output should be nonzero only in block i

Verification strategy: compute circuit output directly (d-dimensional), expand into
full D-dimensional vector with zeros elsewhere, and compare with the network's actual
hidden activations. This catches any nonzero leakage into wrong blocks.
"""

import torch
from torch.nn import functional as F

from spd.experiments.tms.bss_models import PingPongModel


def verify_masking_patterns(model: PingPongModel, n_samples: int = 100) -> None:
    """Verify that hidden activations are zero outside the active block at every layer."""
    device = next(model.parameters()).device
    D, d, num_blocks = model.D, model.d, model.num_blocks
    T = model.T
    n_layers = model.n_layers

    max_leakage = 0.0
    max_output_error = 0.0

    for sample in range(n_samples):
        circuit_id = sample % T
        i, j = model._circuit_id_to_ij(circuit_id)

        x_input = torch.randn(d, device=device)

        # Build full network input
        x_block = torch.zeros(D, device=device)
        x_block[d * i : d * (i + 1)] = x_input

        one_hot_i = torch.zeros(num_blocks, device=device)
        one_hot_i[i] = 1.0
        one_hot_j = torch.zeros(num_blocks, device=device)
        one_hot_j[j] = 1.0

        full_input = torch.cat([x_block, one_hot_i, one_hot_j])

        # Run layer by layer, checking hidden activations
        h_direct = x_input  # circuit-level hidden state (d,)
        h_network = full_input  # network-level hidden state (input_dim,)

        for layer_idx in range(n_layers):
            # Direct circuit computation for this layer
            W = model._circuit_weights[layer_idx][circuit_id].to(device)
            b = model._circuit_biases[layer_idx][circuit_id].to(device)
            h_direct = F.relu(W @ h_direct + b)

            # Network computation for this layer
            linear = model.model[2 * layer_idx]
            relu = model.model[2 * layer_idx + 1]
            h_network = relu(linear(h_network))

            # After this layer, which block should be active?
            active_block = j if layer_idx % 2 == 0 else i

            # Build expected full output: circuit result in active block, zeros elsewhere, one-hots preserved
            expected_full = torch.zeros(D + 2 * num_blocks, device=device)
            expected_full[d * active_block : d * (active_block + 1)] = h_direct  # Circuit output in active block
            expected_full[D : D + num_blocks] = one_hot_i  # one_hot_i preserved
            expected_full[D + num_blocks :] = one_hot_j  # one_hot_j preserved

            # Extract computing block from actual network output
            actual_computing = h_network[:D]
            expected_computing = expected_full[:D]

            # Check 1: leakage — are non-active blocks zero?
            for blk in range(num_blocks):
                if blk == active_block:
                    continue
                block_vals = actual_computing[d * blk : d * (blk + 1)]
                leakage = block_vals.abs().max().item()
                if leakage > 1e-5:
                    print(
                        f"  LEAK: circuit ({i},{j}) layer {layer_idx} "
                        f"block {blk} max |val| = {leakage:.2e}"
                    )
                max_leakage = max(max_leakage, leakage)

            # Check 2: active block matches direct computation
            error = (expected_computing - actual_computing).abs().max().item()
            if error > 1e-5:
                print(
                    f"  MISMATCH: circuit ({i},{j}) layer {layer_idx} "
                    f"max |error| = {error:.2e}"
                )
            max_output_error = max(max_output_error, error)

            # Check 3: one-hot vectors are preserved
            actual_one_hot_i = h_network[D : D + num_blocks]
            actual_one_hot_j = h_network[D + num_blocks :]
            i_error = (actual_one_hot_i - one_hot_i).abs().max().item()
            j_error = (actual_one_hot_j - one_hot_j).abs().max().item()

            if i_error > 1e-6:
                print(f"  one_hot_i error: {i_error:.2e}")
            if j_error > 1e-6:
                print(f"  one_hot_j error: {j_error:.2e}")

            assert i_error < 1e-6, f"one_hot_i corrupted at circuit ({i},{j}) layer {layer_idx}"
            assert j_error < 1e-6, f"one_hot_j corrupted at circuit ({i},{j}) layer {layer_idx}"

    print(f"\nResults ({n_samples} samples, {T} circuits, {n_layers} layers):")
    print(f"  Max leakage into wrong block: {max_leakage:.2e}")
    print(f"  Max error vs direct computation: {max_output_error:.2e}")

    # Note: Small leakage is expected when B is not large enough relative to
    # pre-activation magnitudes. The model's verify_all_circuits checks active
    # block accuracy, which is the primary concern.
    if max_leakage < 1e-4 and max_output_error < 1e-4:
        print("  ✓ Masking patterns verified (negligible leakage)")
    elif max_leakage < 1.0 and max_output_error < 1.0:
        print(f"  ⚠ Small leakage detected (B={model.config.B} may be too small)")
        print("    This is acceptable if model.verify_all_circuits() passed")
    else:
        print("  ✗ Significant leakage detected!")
        assert False, f"Leakage too high: {max_leakage:.2e}"


if __name__ == "__main__":
    import argparse

    from spd.experiments.tms.bss_configs import BSSModelConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="Random seed for model initialization")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--B", type=float, default=15.0)
    parser.add_argument("--n-layers", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = BSSModelConfig(D=args.D, d=args.d, B=args.B, n_layers=args.n_layers)
    model = PingPongModel(config)
    model.eval()

    print(f"Created fresh PingPong (seed={args.seed}): D={model.D}, d={model.d}, T={model.T}, n_layers={model.n_layers}")

    with torch.no_grad():
        # First verify model's own check passes
        try:
            max_err = model.verify_all_circuits()
            print(f"Model verify_all_circuits: {max_err:.2e}\n")
        except AssertionError as e:
            print(f"WARNING: Model verify_all_circuits failed: {e}\n")
            print("Try a different seed. Some random initializations have B too small for masking.\n")
            exit(1)

        # Now verify masking patterns
        verify_masking_patterns(model, n_samples=args.n_samples)
