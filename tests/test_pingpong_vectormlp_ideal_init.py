"""Tests for the vector_mlp ideal initialization of routing masking components.

Verifies that at initialization the routing masking CI functions have no leaks:
  - The routing masking CI for the active one-hot fires (≈ 1)
  - The routing masking CIs for inactive one-hots of the same group are silent (≈ 0)

We do NOT test output-to-target match here: computational components are random,
so the full forward pass won't match the target model at init.
"""

import torch

from spd.configs import Config
from spd.experiments.tms.bss_models import PingPongModel, PingPongTargetRunInfo
from spd.experiments.tms.pingpong_vectormlp_ideal_init_decomposition import (
    LAYER_ROUTING,
    N_COMPUTATIONAL,
    N_OHI,
    NUM_BLOCKS,
    D,
    _routing_component_range,
    initialize_routing_ci_fns,
)
from spd.models.component_model import ComponentModel
from spd.utils.module_utils import expand_module_patterns

CONFIG_PATH = "spd/experiments/tms/pingpong_vectormlp_ideal_init_64-8_config.yaml"
LAYER_NAMES = ["model.0", "model.2", "model.4"]
d = D // NUM_BLOCKS  # block size = 8


def _build_initialized_model() -> tuple[ComponentModel, PingPongModel]:
    """Build a ComponentModel with only the routing masking CIs initialized."""
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

    initialize_routing_ci_fns(component_model, target_model)

    return component_model, target_model


def _make_circuit_batch(i: int, j: int, n_samples: int = 256) -> torch.Tensor:
    """Create a batch of inputs for circuit (i, j)."""
    x = torch.zeros(n_samples, D + 2 * NUM_BLOCKS)
    x[:, i * d : (i + 1) * d] = torch.randn(n_samples, d).abs()
    x[:, D + i] = 1.0
    x[:, D + NUM_BLOCKS + j] = 1.0
    return x


def _active_routing_component(layer_name: str, i: int, j: int) -> int:
    """Return the component index of the active routing masking component for circuit (i, j)."""
    routing = LAYER_ROUTING[layer_name]
    if routing == "ohj":
        return N_COMPUTATIONAL + N_OHI + j
    return N_COMPUTATIONAL + i


class TestRoutingMaskingCINoLeaks:
    """Routing masking CI: active one-hot → CI ≈ 1, inactive → CI ≈ 0."""

    def test_no_ci_leaks(self) -> None:
        component_model, _ = _build_initialized_model()
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
                    routing = LAYER_ROUTING[layer_name]
                    mask_range = _routing_component_range(routing)
                    active_k = _active_routing_component(layer_name, i, j)

                    for k in mask_range:
                        ci_val = ci_vals[k].item()
                        if k == active_k:
                            if ci_val < ci_on_threshold:
                                failures.append(
                                    f"({i},{j}) {layer_name} comp {k} (active {routing}): "
                                    f"CI={ci_val:.4f} < {ci_on_threshold} (should be ON)"
                                )
                        else:
                            if ci_val > ci_off_threshold:
                                failures.append(
                                    f"({i},{j}) {layer_name} comp {k} (inactive {routing}): "
                                    f"CI={ci_val:.4f} > {ci_off_threshold} (LEAK)"
                                )

        assert not failures, (
            f"{len(failures)} routing masking CI failures:\n" + "\n".join(failures[:20])
        )
