"""Evaluate a trained PingPong per-circuit ideal init checkpoint.

Compares the trained model against a freshly-built ideal init, and runs
circuit-level eval metrics (faithfulness, CI correctness, masking pattern).

Usage:
    python scripts/eval_pingpong_ideal_init.py <wandb_path>
    python scripts/eval_pingpong_ideal_init.py wandb:paramluhadiya/spd/runs/s-1c8b8e5d
"""

import sys

import torch

from spd.configs import Config
from spd.experiments.tms.bss_models import PingPongModel, PingPongTargetRunInfo
from spd.experiments.tms.pingpong_decomposition import PingPongDataset
from spd.experiments.tms.pingpong_percircuit_ideal_init_decomposition import (
    D,
    LAYER_ROUTING,
    N_COMPUTATIONAL,
    N_INDEXING,
    N_TRUE,
    NUM_BLOCKS,
    comp_index,
    d,
    initialize_ci_fns_from_ground_truth,
    initialize_components_from_ground_truth,
    ohi_index,
    ohj_index,
    scale_down_unused_components,
)
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.module_utils import expand_module_patterns

CONFIG_PATH = "spd/experiments/tms/pingpong_percircuit_ideal_init_64-8_config.yaml"
LAYER_NAMES = ["model.0", "model.2", "model.4"]


def build_ideal_init() -> tuple[ComponentModel, PingPongModel]:
    config = Config.from_file(CONFIG_PATH)
    target_run_info = PingPongTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = PingPongModel.from_run_info(target_run_info)
    target_model.eval()
    target_model.requires_grad_(False)
    module_path_info = expand_module_patterns(target_model, config.all_module_info)
    component_model = ComponentModel(
        target_model=target_model,
        module_path_info=module_path_info,
        ci_fn_type=config.ci_fn_type,
        ci_fn_hidden_dims=config.ci_fn_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        sigmoid_type=config.sigmoid_type,
    )
    initialize_components_from_ground_truth(component_model, target_model)
    scale_down_unused_components(component_model)
    initialize_ci_fns_from_ground_truth(component_model, target_model)
    return component_model, target_model


def load_trained(wandb_path: str) -> ComponentModel:
    return ComponentModel.from_pretrained(wandb_path)


def make_circuit_batch(i: int, j: int, n_samples: int = 512) -> torch.Tensor:
    x = torch.zeros(n_samples, D + 2 * NUM_BLOCKS)
    x[:, i * d : (i + 1) * d] = torch.randn(n_samples, d).abs()
    x[:, D + i] = 1.0
    x[:, D + NUM_BLOCKS + j] = 1.0
    return x


# ── Section 1: Parameter comparison ──────────────────────────────────────────


def compare_parameters(trained: ComponentModel, ideal: ComponentModel) -> None:
    print("=" * 70)
    print("SECTION 1: PARAMETER COMPARISON (trained vs ideal init)")
    print("=" * 70)

    # Compare V and U
    print("\n--- V and U matrices ---")
    for layer in LAYER_NAMES:
        V_t = trained.components[layer].V.data
        V_i = ideal.components[layer].V.data
        U_t = trained.components[layer].U.data
        U_i = ideal.components[layer].U.data

        v_diff = (V_t - V_i).norm() / V_i.norm()
        u_diff = (U_t - U_i).norm() / U_i.norm()
        print(f"  {layer}: V rel_diff={v_diff:.6f}, U rel_diff={u_diff:.6f}")

    # Compare CI function parameters
    print("\n--- CI function parameters ---")
    for layer in LAYER_NAMES:
        ci_t = trained.ci_fns[layer]
        ci_i = ideal.ci_fns[layer]
        for idx, (pt, pi) in enumerate(zip(ci_t.parameters(), ci_i.parameters())):
            diff = (pt.data - pi.data).norm()
            rel_diff = diff / (pi.data.norm() + 1e-10)
            name = f"param_{idx} ({list(pt.shape)})"
            if rel_diff > 0.01:
                print(f"  {layer} {name}: abs_diff={diff:.6f}, rel_diff={rel_diff:.6f} ⚠️")
            else:
                print(f"  {layer} {name}: abs_diff={diff:.6f}, rel_diff={rel_diff:.6f}")

    # Weight reconstruction (faithfulness)
    print("\n--- Faithfulness (weight delta norms) ---")
    for model_name, model_label in [("trained", trained), ("ideal", ideal)]:
        deltas = model_label.calc_weight_deltas()
        for layer, delta in deltas.items():
            print(f"  {model_label.__class__.__name__}({model_name}) {layer}: ||delta||={delta.norm().item():.6e}")


# ── Section 2: CI evaluation ─────────────────────────────────────────────────


def eval_ci(model: ComponentModel, label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"SECTION 2: CI EVALUATION ({label})")
    print("=" * 70)
    model.eval()

    neuron_clearly_on = 0.04
    neuron_clearly_off = 0.01
    ci_on_thresh = 0.8
    ci_off_thresh = 0.1

    leak_count = 0
    mismatch_on_count = 0
    mismatch_off_count = 0
    total_checked = 0

    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            x = make_circuit_batch(i, j, n_samples=256)
            output = model(x, cache_type="input")
            ci = model.calc_causal_importances(output.cache, sampling="continuous")

            for layer_name in LAYER_NAMES:
                ci_vals = ci.upper_leaky[layer_name]  # (batch, C)
                ci_mean = ci_vals.mean(dim=0)
                layer_input = output.cache[layer_name]

                routing = LAYER_ROUTING[layer_name]
                src, route = (i, j) if routing == "ohj" else (j, i)
                expected_comp = {comp_index(src, route, n) for n in range(d)}
                expected_idx = {ohi_index(i), ohj_index(j)}
                expected_all = expected_comp | expected_idx

                # No-leak check
                for k in range(N_TRUE):
                    if k in expected_all:
                        continue
                    total_checked += 1
                    if ci_mean[k].item() > ci_off_thresh:
                        leak_count += 1

                # Activation matching check
                for neuron in range(d):
                    k = src * d + neuron
                    comp_idx = comp_index(src, route, neuron)
                    ci_per_sample = ci_vals[:, comp_idx]

                    on_mask = layer_input[:, k] > neuron_clearly_on
                    off_mask = layer_input[:, k] < neuron_clearly_off

                    if on_mask.any():
                        if ci_per_sample[on_mask].mean().item() < ci_on_thresh:
                            mismatch_on_count += 1

                    if off_mask.any():
                        if ci_per_sample[off_mask].mean().item() > ci_off_thresh:
                            mismatch_off_count += 1

    print(f"  Leak failures (off-block CI > {ci_off_thresh}): {leak_count}")
    print(f"  Mismatch ON (neuron active but CI low): {mismatch_on_count}")
    print(f"  Mismatch OFF (neuron inactive but CI high): {mismatch_off_count}")


# ── Section 3: Masking pattern eval ──────────────────────────────────────────


def eval_masking_pattern(model: ComponentModel, target_model: PingPongModel, label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"SECTION 3: MASKING PATTERN EVAL ({label})")
    print("=" * 70)
    model.eval()
    target_model.eval()

    cos_sim_threshold = 0.75
    ci_threshold = 0.1
    max_mse = 0.0
    worst_circuit = (0, 0)
    total_correct = 0
    total_circuits = NUM_BLOCKS * NUM_BLOCKS * len(LAYER_NAMES)

    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            x = make_circuit_batch(i, j, n_samples=256)

            target_out = target_model(x)
            comp_out = model(x, cache_type="input")
            ci = model.calc_causal_importances(comp_out.cache, sampling="continuous")

            mse = (comp_out.output - target_out).pow(2).mean().item()
            if mse > max_mse:
                max_mse = mse
                worst_circuit = (i, j)

            for layer_name in LAYER_NAMES:
                ci_mean = ci.upper_leaky[layer_name].mean(dim=0)
                routing = LAYER_ROUTING[layer_name]
                src, route = (i, j) if routing == "ohj" else (j, i)
                expected_comp = {comp_index(src, route, n) for n in range(d)}
                expected_idx = {ohi_index(i), ohj_index(j)}

                # Check: are the top-CI components the expected ones?
                top_k = ci_mean[:N_TRUE].topk(d + 2).indices.tolist()
                expected_all = expected_comp | expected_idx
                overlap = len(set(top_k) & expected_all)
                if overlap >= d:  # at least the 8 computational + most indexing
                    total_correct += 1

    print(f"  Output MSE (worst circuit {worst_circuit}): {max_mse:.6e}")
    print(f"  Circuit identification: {total_correct}/{total_circuits} correct (top-{d+2} CI overlap ≥ {d})")


# ── Section 4: Random data eval ──────────────────────────────────────────────


def eval_random_data(model: ComponentModel, target_model: PingPongModel, label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"SECTION 4: RANDOM DATA METRICS ({label})")
    print("=" * 70)
    model.eval()
    target_model.eval()

    dataset = PingPongDataset(D=64, d=8, num_blocks=8, device="cpu", value_range=(0.0, 1.0))

    total_mse = 0.0
    n_batches = 20
    ci_l0_per_layer: dict[str, float] = {l: 0.0 for l in LAYER_NAMES}

    for _ in range(n_batches):
        x = dataset.generate_batch(1024)
        target_out = target_model(x)
        comp_out = model(x, cache_type="input")
        ci = model.calc_causal_importances(comp_out.cache, sampling="continuous")

        total_mse += (comp_out.output - target_out).pow(2).mean().item()

        for layer_name in LAYER_NAMES:
            ci_vals = ci.upper_leaky[layer_name]
            ci_l0_per_layer[layer_name] += (ci_vals > 0.1).float().mean().item()

    print(f"  Mean output MSE: {total_mse / n_batches:.6e}")
    print(f"  Mean CI-L0 (fraction > 0.1):")
    for layer_name in LAYER_NAMES:
        mean_l0 = ci_l0_per_layer[layer_name] / n_batches
        approx_n_active = mean_l0 * 600
        print(f"    {layer_name}: {mean_l0:.4f} (~{approx_n_active:.1f} components)")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <wandb_path>"
    wandb_path = sys.argv[1]

    print(f"Loading trained model from {wandb_path}...")
    trained = load_trained(wandb_path)

    print("Building ideal init model...")
    ideal, target_model = build_ideal_init()

    with torch.no_grad():
        compare_parameters(trained, ideal)
        eval_ci(trained, "trained")
        eval_ci(ideal, "ideal init")
        eval_masking_pattern(trained, target_model, "trained")
        eval_masking_pattern(ideal, target_model, "ideal init")
        eval_random_data(trained, target_model, "trained")
        eval_random_data(ideal, target_model, "ideal init")


if __name__ == "__main__":
    main()
