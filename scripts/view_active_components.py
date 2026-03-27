"""View active components for PingPong circuits.

Load a trained SPD model and show which components have CI above a threshold
for each circuit (i, j). For layer 0, also prints the V vector of each active
component, split into computational blocks, i-block, and j-block.

Usage:
    python scripts/view_active_components.py
"""

import torch

from spd.models.component_model import ComponentModel

MODEL_PATH = "wandb:paramluhadiya/spd/runs/s-898e5c08"
CI_THRESHOLD = 0.5
N_SAMPLES = 32
DEVICE = "cpu"
INSPECT_LAYER = "model.0"


def make_circuit_batch(
    i: int,
    j: int,
    n_samples: int,
    D: int,
    d: int,
    num_blocks: int,
    device: str,
) -> torch.Tensor:
    input_dim = D + 2 * num_blocks
    x = torch.zeros(n_samples, input_dim, device=device)
    values = torch.randn(n_samples, d, device=device)
    x[:, d * i : d * (i + 1)] = values
    x[:, D + i] = 1.0
    x[:, D + num_blocks + j] = 1.0
    return x


def format_v_vector(v: torch.Tensor, D: int, d: int, num_blocks: int) -> str:
    """Format a V column vector split into computational blocks, i-block, j-block."""
    comp = v[:D]
    i_block = v[D : D + num_blocks]
    j_block = v[D + num_blocks :]

    lines = []

    # Computational blocks: show per-block norms
    block_norms = [comp[b * d : (b + 1) * d].norm().item() for b in range(num_blocks)]
    norms_str = "  ".join(f"b{b}={n:.2f}" for b, n in enumerate(block_norms))
    lines.append(f"      comp block norms: [{norms_str}]")

    # i-block and j-block: show raw values (only num_blocks entries)
    i_str = "  ".join(f"{x:.2f}" for x in i_block.tolist())
    j_str = "  ".join(f"{x:.2f}" for x in j_block.tolist())
    lines.append(f"      i-block: [{i_str}]")
    lines.append(f"      j-block: [{j_str}]")

    return "\n".join(lines)


def format_u_vector(u: torch.Tensor, D: int, d: int, num_blocks: int) -> str:
    """Format a U row vector split into computational blocks, i-block, j-block."""
    comp = u[:D]
    i_block = u[D : D + num_blocks]
    j_block = u[D + num_blocks :]

    lines = []

    block_norms = [comp[b * d : (b + 1) * d].norm().item() for b in range(num_blocks)]
    norms_str = "  ".join(f"b{b}={n:.2f}" for b, n in enumerate(block_norms))
    lines.append(f"      comp block norms: [{norms_str}]")

    # Also show per-block mean to see sign (positive = activate, negative = suppress)
    block_means = [comp[b * d : (b + 1) * d].mean().item() for b in range(num_blocks)]
    means_str = "  ".join(f"b{b}={m:.2f}" for b, m in enumerate(block_means))
    lines.append(f"      comp block means: [{means_str}]")

    i_str = "  ".join(f"{x:.2f}" for x in i_block.tolist())
    j_str = "  ".join(f"{x:.2f}" for x in j_block.tolist())
    lines.append(f"      i-block: [{i_str}]")
    lines.append(f"      j-block: [{j_str}]")

    return "\n".join(lines)


@torch.no_grad()
def main() -> None:
    model = ComponentModel.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(DEVICE)

    from spd.experiments.tms.bss_models import PingPongModel

    assert isinstance(model.target_model, PingPongModel)
    D = model.target_model.D
    d = model.target_model.d
    num_blocks = model.target_model.num_blocks

    # V matrix for the inspect layer: shape (d_in, C)
    # U matrix for the inspect layer: shape (C, d_out)
    V = model.components[INSPECT_LAYER].V.detach()
    U = model.components[INSPECT_LAYER].U.detach()

    # Get the first two layers of the target model (Linear + ReLU) for post-layer-0 check
    target_sequential = model.target_model.model
    layer0_linear = target_sequential[0]
    layer0_relu = target_sequential[1]

    for i, j in [(0, 0)]:
            batch = make_circuit_batch(i, j, N_SAMPLES, D, d, num_blocks, DEVICE)
            out = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                sampling="binomial",
                detach_inputs=True,
            )

            # Post-layer-0 activation from TARGET model: ReLU(W @ x + b)
            target_post = layer0_relu(layer0_linear(batch))  # (N_SAMPLES, input_dim)
            target_mean = target_post.mean(dim=0)[:D]
            target_norms = [target_mean[b * d : (b + 1) * d].norm().item() for b in range(num_blocks)]
            target_str = "  ".join(f"b{b}={n:.2f}" for b, n in enumerate(target_norms))

            # Post-layer-0 activation from DECOMPOSITION (all components): ReLU(V @ U @ x)
            components = model.components[INSPECT_LAYER]
            decomp_post = layer0_relu(components(batch))
            decomp_mean = decomp_post.mean(dim=0)[:D]
            decomp_norms = [decomp_mean[b * d : (b + 1) * d].norm().item() for b in range(num_blocks)]
            decomp_str = "  ".join(f"b{b}={n:.2f}" for b, n in enumerate(decomp_norms))

            # Post-layer-0 with hard CI mask: zero out components below threshold
            ci_vals_layer0 = ci.lower_leaky[INSPECT_LAYER]  # (N_SAMPLES, C)
            hard_mask = torch.where(ci_vals_layer0 > CI_THRESHOLD, ci_vals_layer0, torch.zeros_like(ci_vals_layer0))
            masked_post = layer0_relu(components(batch, mask=hard_mask))
            masked_mean = masked_post.mean(dim=0)[:D]
            masked_norms = [masked_mean[b * d : (b + 1) * d].norm().item() for b in range(num_blocks)]
            masked_str = "  ".join(f"b{b}={n:.2f}" for b, n in enumerate(masked_norms))

            print(f"\n=== Circuit ({i}, {j}) ===")
            print(f"  Post-layer-0 block norms (expect only b{j} nonzero):")
            print(f"    target:     [{target_str}]")
            print(f"    decomp:     [{decomp_str}]")
            print(f"    CI-masked:  [{masked_str}]")

            for layer_name in model.target_module_paths:
                ci_vals = ci.lower_leaky[layer_name].mean(dim=0)  # (C,)
                active_mask = ci_vals > CI_THRESHOLD
                active_indices = active_mask.nonzero(as_tuple=True)[0]
                active_vals = ci_vals[active_indices]

                sorted_order = active_vals.argsort(descending=True)
                active_indices = active_indices[sorted_order]
                active_vals = active_vals[sorted_order]

                print(f"  {layer_name}: {len(active_indices)} active components")
                for idx, val in zip(active_indices, active_vals):
                    print(f"    component {idx.item():>3d}  CI={val.item():.3f}")
                    if layer_name == INSPECT_LAYER:
                        print("    V:")
                        print(format_v_vector(V[:, idx], D, d, num_blocks))
                        print("    U:")
                        print(format_u_vector(U[idx, :], D, d, num_blocks))


if __name__ == "__main__":
    main()
