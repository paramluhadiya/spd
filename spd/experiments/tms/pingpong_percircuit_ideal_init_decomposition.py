"""Run SPD on PingPong with per-circuit ideal initialization.

Unlike the original 80-component decomposition (one per input dimension),
this uses 528 true components:
  - 512 computational: one per (src_block, route_block, neuron) triplet.
    V picks up the neuron's input dim; U maps only to the route block's output dims.
  - 16 indexing: 8 ohi + 8 ohj (same as original)

CI initialization uses AND logic via a 3-layer MLP (2 hidden + output):
  - Layer 0: detect block activity (sum), ohi, ohj from input
  - Layer 1: compute AND(block_src, routing_onehot) for each (src, route) pair
  - Layer 2: map AND results to component CIs

  Layers 0, 2: CI_k = AND(block_src active, ohj[route] = 1)
  Layer 1:     CI_k = AND(block_src active, ohi[route] = 1)

Usage:
    python spd/experiments/tms/pingpong_percircuit_ideal_init_decomposition.py \\
        --config_path spd/experiments/tms/pingpong_percircuit_ideal_init_64-8_config.yaml
"""

import json
from pathlib import Path

import fire
import torch
import wandb

from spd.configs import Config, PingPongTaskConfig
from spd.experiments.tms.bss_models import PingPongModel, PingPongTargetRunInfo
from spd.experiments.tms.pingpong_decomposition import PingPongDataset
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import Linear, LinearComponents, VectorSharedMLPCiFn
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb

D = 64
d = 8
NUM_BLOCKS = 8
N_COMPUTATIONAL = 512  # 8 src × 8 route × 8 neurons
N_OHI = 8
N_OHJ = 8
N_INDEXING = N_OHI + N_OHJ  # 16
N_TRUE = N_COMPUTATIONAL + N_INDEXING  # 528

# Which one-hot determines routing at each layer
LAYER_ROUTING: dict[str, str] = {
    "model.0": "ohj",  # layer 0: src=i, route by ohj[j]
    "model.2": "ohi",  # layer 1: src=j, route by ohi[i]
    "model.4": "ohj",  # layer 2: src=i, route by ohj[j]
}

# --- Hidden dim layout ---
# Layer 0 output (hidden1):
H1_BLOCK_START = 0   # dims 0..15:  block sum detectors (8 blocks × 2)
H1_OHI_START = 16    # dims 16..31: ohi detectors (8 × 2)
H1_OHJ_START = 32    # dims 32..47: ohj detectors (8 × 2)
N_IDEAL_H1 = 48

# Layer 1 output (hidden2):
H2_AND_START = 0     # dims 0..127:   AND detectors (64 pairs × 2)
H2_OHI_START = 128   # dims 128..143: ohi pass-through (8 × 2)
H2_OHJ_START = 144   # dims 144..159: ohj pass-through (8 × 2)
N_IDEAL_H2 = 160


def comp_index(src: int, route: int, neuron: int) -> int:
    """Component index for (src_block, route_block, local_neuron)."""
    return src * (NUM_BLOCKS * d) + route * d + neuron


def ohi_index(i: int) -> int:
    return N_COMPUTATIONAL + i


def ohj_index(j: int) -> int:
    return N_COMPUTATIONAL + N_OHI + j


def initialize_components_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
) -> None:
    """Initialize V and U for the per-circuit decomposition.

    Computational components: V picks up a single neuron, U maps only to the route block.
    Indexing components: same as original (V = one-hot, U = full column of W^T).
    """
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)
        assert components.C >= N_TRUE

        W_T = component_model.target_weight(module_name).T
        d_in = W_T.shape[0]
        assert d_in == target_model.input_dim

        with torch.no_grad():
            eye = torch.eye(d_in, device=components.V.device)

            # Computational: 512 components
            for src in range(NUM_BLOCKS):
                for route in range(NUM_BLOCKS):
                    for neuron in range(d):
                        idx = comp_index(src, route, neuron)
                        k = src * d + neuron
                        components.V.data[:, idx] = eye[k]
                        components.U.data[idx, :] = 0.0
                        components.U.data[idx, route * d : (route + 1) * d] = (
                            W_T[k, route * d : (route + 1) * d]
                        )

            # Indexing: 16 components (ohi + ohj)
            for i in range(N_OHI):
                idx = ohi_index(i)
                k = D + i
                components.V.data[:, idx] = eye[k]
                components.U.data[idx, :] = W_T[k, :]

            for j in range(N_OHJ):
                idx = ohj_index(j)
                k = D + NUM_BLOCKS + j
                components.V.data[:, idx] = eye[k]
                components.U.data[idx, :] = W_T[k, :]


def scale_down_unused_components(component_model: ComponentModel) -> None:
    """Scale down components beyond the true decomposition."""
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)
        with torch.no_grad():
            components.V.data[:, N_TRUE:] *= 0.01
            components.U.data[N_TRUE:, :] *= 0.01


def initialize_ci_fns_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
) -> None:
    """Initialize CI MLP for AND-based per-circuit detection.

    Uses GELU finite-difference trick across 3 layers:
      Layer 0: detect block sums, ohi, ohj from input
      Layer 1: AND(block_src, routing_onehot) + indexing pass-through
      Layer 2: map to component CIs
    """
    S = 50.0
    t = 1.0

    for module_name in component_model.target_module_paths:
        ci_fn = component_model.ci_fns[module_name]
        assert isinstance(ci_fn, VectorSharedMLPCiFn)
        routing = LAYER_ROUTING[module_name]

        layer0 = ci_fn.layers[0]
        layer1 = ci_fn.layers[2]  # layers[1] is GELU
        layer2 = ci_fn.layers[4]  # layers[3] is GELU
        assert isinstance(layer0, Linear)
        assert isinstance(layer1, Linear)
        assert isinstance(layer2, Linear)

        input_dim = layer0.W.shape[0]
        H1 = layer0.W.shape[1]
        H2 = layer1.W.shape[1]
        C = layer2.W.shape[1]
        assert input_dim == target_model.input_dim
        assert H1 >= N_IDEAL_H1, f"H1={H1} < {N_IDEAL_H1}"
        assert H2 >= N_IDEAL_H2, f"H2={H2} < {N_IDEAL_H2}"

        with torch.no_grad():
            # =============================================
            # Layer 0: input (80) -> hidden1 (H1)
            # =============================================
            # Isolate ideal hidden dims from random inputs
            for h in range(N_IDEAL_H1):
                layer0.W.data[:, h] = 0.0
            # Isolate ideal input dims from random hidden dims
            for k in range(input_dim):
                layer0.W.data[k, :] = 0.0

            # Block sum detectors: sum all d neurons in block, apply GELU finite diff
            for b in range(NUM_BLOCKS):
                h_pos = H1_BLOCK_START + 2 * b
                h_neg = h_pos + 1
                for n in range(d):
                    layer0.W.data[b * d + n, h_pos] = S
                    layer0.W.data[b * d + n, h_neg] = S
                layer0.b.data[h_pos] = -t
                layer0.b.data[h_neg] = -(t + 1)

            # ohi detectors
            for i in range(N_OHI):
                h_pos = H1_OHI_START + 2 * i
                h_neg = h_pos + 1
                layer0.W.data[D + i, h_pos] = S
                layer0.W.data[D + i, h_neg] = S
                layer0.b.data[h_pos] = -t
                layer0.b.data[h_neg] = -(t + 1)

            # ohj detectors
            for j in range(N_OHJ):
                h_pos = H1_OHJ_START + 2 * j
                h_neg = h_pos + 1
                layer0.W.data[D + NUM_BLOCKS + j, h_pos] = S
                layer0.W.data[D + NUM_BLOCKS + j, h_neg] = S
                layer0.b.data[h_pos] = -t
                layer0.b.data[h_neg] = -(t + 1)

            # =============================================
            # Layer 1: hidden1 (H1) -> hidden2 (H2)
            # =============================================
            for h in range(N_IDEAL_H2):
                layer1.W.data[:, h] = 0.0
            for h in range(N_IDEAL_H1):
                layer1.W.data[h, :] = 0.0

            # AND detectors: AND(block_src indicator, routing_onehot indicator)
            # Routing uses ohj for layers 0,2 and ohi for layer 1
            routing_h1_start = H1_OHJ_START if routing == "ohj" else H1_OHI_START

            for src in range(NUM_BLOCKS):
                for route in range(NUM_BLOCKS):
                    and_idx = src * NUM_BLOCKS + route
                    h2_pos = H2_AND_START + 2 * and_idx
                    h2_neg = h2_pos + 1

                    # Block src indicator: h1[2*src] - h1[2*src+1]
                    block_h1_pos = H1_BLOCK_START + 2 * src
                    block_h1_neg = block_h1_pos + 1
                    layer1.W.data[block_h1_pos, h2_pos] = S
                    layer1.W.data[block_h1_neg, h2_pos] = -S
                    layer1.W.data[block_h1_pos, h2_neg] = S
                    layer1.W.data[block_h1_neg, h2_neg] = -S

                    # Routing one-hot indicator
                    route_h1_pos = routing_h1_start + 2 * route
                    route_h1_neg = route_h1_pos + 1
                    layer1.W.data[route_h1_pos, h2_pos] = S
                    layer1.W.data[route_h1_neg, h2_pos] = -S
                    layer1.W.data[route_h1_pos, h2_neg] = S
                    layer1.W.data[route_h1_neg, h2_neg] = -S

                    # Threshold at indicator_sum = 1.5: active only when both = 1
                    layer1.b.data[h2_pos] = -1.5 * S
                    layer1.b.data[h2_neg] = -1.5 * S - 1

            # Indexing pass-through: re-apply finite diff to preserve indicators
            for i in range(N_OHI):
                h1_pos = H1_OHI_START + 2 * i
                h1_neg = h1_pos + 1
                h2_pos = H2_OHI_START + 2 * i
                h2_neg = h2_pos + 1
                layer1.W.data[h1_pos, h2_pos] = S
                layer1.W.data[h1_neg, h2_pos] = -S
                layer1.W.data[h1_pos, h2_neg] = S
                layer1.W.data[h1_neg, h2_neg] = -S
                layer1.b.data[h2_pos] = -t
                layer1.b.data[h2_neg] = -(t + 1)

            for j in range(N_OHJ):
                h1_pos = H1_OHJ_START + 2 * j
                h1_neg = h1_pos + 1
                h2_pos = H2_OHJ_START + 2 * j
                h2_neg = h2_pos + 1
                layer1.W.data[h1_pos, h2_pos] = S
                layer1.W.data[h1_neg, h2_pos] = -S
                layer1.W.data[h1_pos, h2_neg] = S
                layer1.W.data[h1_neg, h2_neg] = -S
                layer1.b.data[h2_pos] = -t
                layer1.b.data[h2_neg] = -(t + 1)

            # =============================================
            # Layer 2: hidden2 (H2) -> output (C)
            # =============================================
            for c in range(N_TRUE):
                layer2.W.data[:, c] = 0.0
            for h in range(N_IDEAL_H2):
                layer2.W.data[h, :] = 0.0

            # Computational: all 8 neurons in (src, route) read same AND
            for src in range(NUM_BLOCKS):
                for route in range(NUM_BLOCKS):
                    and_idx = src * NUM_BLOCKS + route
                    h2_pos = H2_AND_START + 2 * and_idx
                    h2_neg = h2_pos + 1
                    for neuron in range(d):
                        c_idx = comp_index(src, route, neuron)
                        layer2.W.data[h2_pos, c_idx] = 1.0
                        layer2.W.data[h2_neg, c_idx] = -1.0
                        layer2.b.data[c_idx] = 0.0

            # Indexing: read from pass-through dims
            for i in range(N_OHI):
                c_idx = ohi_index(i)
                h2_pos = H2_OHI_START + 2 * i
                h2_neg = h2_pos + 1
                layer2.W.data[h2_pos, c_idx] = 1.0
                layer2.W.data[h2_neg, c_idx] = -1.0
                layer2.b.data[c_idx] = 0.0

            for j in range(N_OHJ):
                c_idx = ohj_index(j)
                h2_pos = H2_OHJ_START + 2 * j
                h2_neg = h2_pos + 1
                layer2.W.data[h2_pos, c_idx] = 1.0
                layer2.W.data[h2_neg, c_idx] = -1.0
                layer2.b.data[c_idx] = 0.0

            # Unused components: bias off
            layer2.b.data[N_TRUE:C] = -3.0


def main(
    config_path: Path | str | None = None,
    config_json: str | None = None,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    assert (config_path is not None) != (config_json is not None), (
        "Need exactly one of config_path and config_json"
    )
    if config_path is not None:
        config = Config.from_file(config_path)
    else:
        assert config_json is not None
        config = Config(**json.loads(config_json.removeprefix("json:")))

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    out_dir, run_id, tags = setup_decomposition_run(
        experiment_tag="pingpong-percircuit-ideal-init", evals_id=evals_id, sweep_id=sweep_id
    )
    if config.wandb_project:
        init_wandb(
            config=config,
            project=config.wandb_project,
            run_id=run_id,
            name=config.wandb_run_name,
            tags=tags,
        )
    logger.info(config)

    task_config = config.task_config
    assert isinstance(task_config, PingPongTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = PingPongTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = PingPongModel.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    logger.info(
        f"Loaded PingPong model: D={target_model.D}, d={target_model.d}, "
        f"T={target_model.T}, n_layers={target_model.n_layers}"
    )

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=target_model.config,
        task_name=config.task_config.task_name,
    )

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

    if task_config.init_computational_components == "ideal":
        initialize_components_from_ground_truth(component_model, target_model)
        logger.info(f"Initialized per-circuit components: {N_COMPUTATIONAL} computational + {N_INDEXING} indexing")
    scale_down_unused_components(component_model)

    if task_config.init_computational_ci == "ideal":
        initialize_ci_fns_from_ground_truth(component_model, target_model)
        logger.info("Initialized CI functions with AND logic for per-circuit detection")

    component_model.to(device)

    weight_deltas = component_model.calc_weight_deltas()
    total_faith = sum(torch.norm(wd).item() for wd in weight_deltas.values())
    logger.info(f"Post-init faithfulness (should be ~0): {total_faith:.6e}")

    dataset = PingPongDataset(
        D=target_model.D,
        d=target_model.d,
        num_blocks=target_model.num_blocks,
        device=device,
        value_range=(0.0, 1.0),
    )
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=None,
        pre_built_component_model=component_model,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
