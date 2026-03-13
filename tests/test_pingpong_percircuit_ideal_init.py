"""Tests for the per-circuit ideal initialization of PingPong decomposition.

Verifies:
1. Faithfulness: V @ U reconstructs the target weight matrix
2. CI correctness: for every circuit (i,j) at every layer, exactly the right 10
   components are active (8 computational + ohi + ohj) and all others are off
3. Masked output: the decomposition with ideal CI matches the target model output
"""

import torch

from spd.configs import Config
from spd.experiments.tms.bss_models import PingPongModel, PingPongTargetRunInfo
from spd.experiments.tms.pingpong_percircuit_ideal_init_decomposition import (
    D,
    LAYER_ROUTING,
    N_COMPUTATIONAL,
    N_OHI,
    N_OHJ,
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
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns

CONFIG_PATH = "spd/experiments/tms/pingpong_percircuit_ideal_init_64-8_config.yaml"
LAYER_NAMES = ["model.0", "model.2", "model.4"]


def _build_initialized_model() -> tuple[ComponentModel, PingPongModel]:
    """Build and initialize a ComponentModel with per-circuit ideal init."""
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


def _make_circuit_batch(i: int, j: int, n_samples: int = 512) -> torch.Tensor:
    """Create a batch of inputs for circuit (i, j)."""
    x = torch.zeros(n_samples, D + 2 * NUM_BLOCKS)
    x[:, i * d : (i + 1) * d] = torch.randn(n_samples, d).abs()
    x[:, D + i] = 1.0
    x[:, D + NUM_BLOCKS + j] = 1.0
    return x


def _expected_active_comp_indices(layer_name: str, i: int, j: int) -> set[int]:
    """Return the set of component indices expected to be active for circuit (i,j) at layer."""
    routing = LAYER_ROUTING[layer_name]
    # Layer 0,2: src=i, route=j. Layer 1: src=j, route=i.
    if routing == "ohj":
        src, route = i, j
    else:
        src, route = j, i
    return {comp_index(src, route, neuron) for neuron in range(d)}


def _expected_active_indexing_indices(i: int, j: int) -> set[int]:
    """Return the set of indexing component indices expected active for circuit (i,j)."""
    return {ohi_index(i), ohj_index(j)}


class TestFaithfulness:
    """V @ U should reconstruct the target weight matrix."""

    def test_weight_reconstruction(self) -> None:
        component_model, target_model = _build_initialized_model()
        weight_deltas = component_model.calc_weight_deltas()
        for layer_name, delta in weight_deltas.items():
            assert delta.norm().item() < 0.01, (
                f"{layer_name}: weight delta norm {delta.norm().item():.4e} too large"
            )


class TestCICorrectness:
    """For every circuit (i,j) at every layer, the CI MLP should output ~1 for the
    correct 10 components and ~0 for everything else."""

    def test_all_circuits_all_layers(self) -> None:
        component_model, target_model = _build_initialized_model()
        component_model.eval()

        ci_on_threshold = 0.8
        ci_off_threshold = 0.1

        failures: list[str] = []

        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                x = _make_circuit_batch(i, j, n_samples=256)
                output = component_model(x, cache_type="input")
                ci = component_model.calc_causal_importances(
                    output.cache, sampling="continuous"
                )

                for layer_name in LAYER_NAMES:
                    ci_vals = ci.upper_leaky[layer_name].mean(dim=0)  # (C,)

                    expected_comp = _expected_active_comp_indices(layer_name, i, j)
                    expected_idx = _expected_active_indexing_indices(i, j)
                    expected_all = expected_comp | expected_idx

                    # Check expected-ON components have CI > threshold
                    for k in expected_all:
                        if ci_vals[k].item() < ci_on_threshold:
                            failures.append(
                                f"({i},{j}) {layer_name} comp {k}: "
                                f"CI={ci_vals[k].item():.4f} < {ci_on_threshold} (should be ON)"
                            )

                    # Check expected-OFF components have CI < threshold
                    for k in range(N_TRUE):
                        if k in expected_all:
                            continue
                        if ci_vals[k].item() > ci_off_threshold:
                            failures.append(
                                f"({i},{j}) {layer_name} comp {k}: "
                                f"CI={ci_vals[k].item():.4f} > {ci_off_threshold} (should be OFF)"
                            )

        assert not failures, (
            f"{len(failures)} CI failures:\n" + "\n".join(failures[:20])
        )

    def test_correct_active_count(self) -> None:
        """Each circuit should have exactly 10 active components per layer."""
        component_model, _ = _build_initialized_model()
        component_model.eval()

        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                x = _make_circuit_batch(i, j, n_samples=256)
                output = component_model(x, cache_type="input")
                ci = component_model.calc_causal_importances(
                    output.cache, sampling="continuous"
                )

                for layer_name in LAYER_NAMES:
                    ci_vals = ci.upper_leaky[layer_name].mean(dim=0)
                    n_active = (ci_vals[:N_TRUE] > 0.5).sum().item()
                    assert n_active == 10, (
                        f"({i},{j}) {layer_name}: {n_active} active (expected 10)"
                    )


class TestComponentIndexing:
    """Verify the component index mapping covers exactly 0..511."""

    def test_comp_indices_unique_and_complete(self) -> None:
        all_indices = set()
        for src in range(NUM_BLOCKS):
            for route in range(NUM_BLOCKS):
                for neuron in range(d):
                    idx = comp_index(src, route, neuron)
                    assert idx not in all_indices, f"Duplicate index {idx}"
                    all_indices.add(idx)
        assert all_indices == set(range(N_COMPUTATIONAL))

    def test_indexing_indices_no_overlap(self) -> None:
        comp_indices = set(range(N_COMPUTATIONAL))
        idx_indices = {ohi_index(i) for i in range(N_OHI)} | {
            ohj_index(j) for j in range(N_OHJ)
        }
        assert not comp_indices & idx_indices
        assert len(idx_indices) == N_OHI + N_OHJ
        assert max(idx_indices) == N_TRUE - 1


class TestUMatrixStructure:
    """U for computational components should only have nonzero entries in the route block."""

    def test_u_block_sparsity(self) -> None:
        component_model, _ = _build_initialized_model()

        for layer_name in LAYER_NAMES:
            components = component_model.components[layer_name]
            U = components.U.detach()

            for src in range(NUM_BLOCKS):
                for route in range(NUM_BLOCKS):
                    for neuron in range(d):
                        idx = comp_index(src, route, neuron)
                        u_row = U[idx, :]

                        # Only the route block should have nonzero entries
                        for b in range(NUM_BLOCKS):
                            block_norm = u_row[b * d : (b + 1) * d].norm().item()
                            if b == route:
                                assert block_norm > 0.01, (
                                    f"{layer_name} comp({src},{route},{neuron}): "
                                    f"route block {b} norm={block_norm:.4f} (should be nonzero)"
                                )
                            else:
                                assert block_norm < 1e-6, (
                                    f"{layer_name} comp({src},{route},{neuron}): "
                                    f"off-route block {b} norm={block_norm:.4f} (should be zero)"
                                )


class TestMaskedOutputMatchesTarget:
    """With ideal CI, the masked output should closely match the target model."""

    def test_output_mse_all_circuits(self) -> None:
        component_model, target_model = _build_initialized_model()
        component_model.eval()
        target_model.eval()

        max_mse = 0.0
        worst_circuit = (0, 0)

        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                x = _make_circuit_batch(i, j, n_samples=256)

                target_out = target_model(x)
                comp_out = component_model(x, cache_type="input")

                mse = (comp_out.output - target_out).pow(2).mean().item()
                if mse > max_mse:
                    max_mse = mse
                    worst_circuit = (i, j)

        assert max_mse < 1e-4, (
            f"Worst circuit {worst_circuit}: MSE={max_mse:.4f} (should be < 0.1)"
        )
