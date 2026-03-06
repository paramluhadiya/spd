
"""Run SPD on PingPong with ground-truth initialization.

Initializes V, U, and CI functions to match the true column-wise rank-1
decomposition of each layer's weight matrix, then runs normal SPD optimization
to see whether the optimizer finds a lower-loss decomposition.

The true decomposition has 80 components per layer (one per input dimension):
  - 64 for the computational block (one per neuron)
  - 8 for one_hot_i (bias/mask + identity pass-through)
  - 8 for one_hot_j (bias/mask + identity pass-through)
The remaining C - 80 components are initialized randomly with low CI.
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

N_TRUE_COMPONENTS = 80  # 64 computational + 8 one_hot_i + 8 one_hot_j


def initialize_components_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
) -> None:
    """Initialize V and U matrices to the column-wise rank-1 decomposition.

    For each layer, V[:, k] = e_k for k=0..79, U[k, :] = W^T[k, :].
    Components 80..C-1 keep their random initialization (scaled down).
    """
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)

        C = components.C
        assert C >= N_TRUE_COMPONENTS, f"C={C} < {N_TRUE_COMPONENTS}"

        W = component_model.target_weight(module_name)
        d_in = W.shape[1]
        assert d_in == target_model.input_dim

        with torch.no_grad():
            # V (d_in, C): first 80 columns are standard basis
            components.V.data[:, :N_TRUE_COMPONENTS] = torch.eye(d_in)
            # U (C, d_out): V @ U = W^T, with V[:,:80] = I this gives U[:80,:] = W^T
            components.U.data[:N_TRUE_COMPONENTS, :] = W.T[:N_TRUE_COMPONENTS, :]

            # Scale down remaining components so they start small
            components.V.data[:, N_TRUE_COMPONENTS:] *= 0.01
            components.U.data[N_TRUE_COMPONENTS:, :] *= 0.01


def initialize_ci_fns_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
) -> None:
    """Initialize CI functions so that CI_k(x) is high when x_k > 0.

    For VectorSharedMLPCiFn with architecture Linear(80, 256) -> GELU -> Linear(256, C):
    - Layer 0: maps input dim k to hidden dim k, so h_k = GELU(scale * x_k - offset)
    - Layer 1: maps hidden dim k to output k, with bias tuned so CI > 0 when active
    - Components 80..C-1 get moderately negative CI bias (off but reachable)
    """
    for module_name in component_model.target_module_paths:
        ci_fn = component_model.ci_fns[module_name]
        assert isinstance(ci_fn, VectorSharedMLPCiFn)

        layer0 = ci_fn.layers[0]
        layer1 = ci_fn.layers[2]  # layers[1] is GELU
        assert isinstance(layer0, Linear)
        assert isinstance(layer1, Linear)

        input_dim = layer0.W.shape[0]
        hidden_dim = layer0.W.shape[1]

        assert input_dim == target_model.input_dim
        assert hidden_dim >= 2 * N_TRUE_COMPONENTS

        with torch.no_grad():
            # We want CI_k(x) ≈ 1 when x_k > 0, ≈ 0 when x_k = 0.
            # upper_leaky_hard sigmoid maps pre-sigmoid=1.0 → exactly 1.0,
            # so we target pre-sigmoid output = 1.
            #
            # Use 2 hidden dims per component to build a saturating step:
            #   h_pos = GELU(S * x_k - t)
            #   h_neg = GELU(S * x_k - (t + 1))
            #   output_k = h_pos - h_neg ≈ GELU'(·) ≈ 1 for x_k > 0
            # Since GELU is asymptotically linear, GELU(a) - GELU(a-1) → 1
            # as a → ∞. For x_k = 0: GELU(-t) - GELU(-t-1) ≈ small negative.
            S = 50.0
            t = 1.0

            layer0.W.data.zero_()
            layer0.b.data.zero_()
            for k in range(N_TRUE_COMPONENTS):
                layer0.W.data[k, 2 * k] = S
                layer0.W.data[k, 2 * k + 1] = S
            layer0.b.data[:2 * N_TRUE_COMPONENTS:2] = -t
            layer0.b.data[1:2 * N_TRUE_COMPONENTS:2] = -(t + 1)

            # Layer 1: output_k = h_pos - h_neg (+ bias for off components)
            layer1.W.data.zero_()
            layer1.b.data.fill_(-3.0)  # Moderately negative default (CI off)
            for k in range(N_TRUE_COMPONENTS):
                layer1.W.data[2 * k, k] = 1.0
                layer1.W.data[2 * k + 1, k] = -1.0
            layer1.b.data[:N_TRUE_COMPONENTS] = 0.0


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
        experiment_tag="pingpong-ideal-init", evals_id=evals_id, sweep_id=sweep_id
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

    # Build ComponentModel and apply ground-truth initialization before optimize()
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

    if task_config.init_components == "ideal":
        initialize_components_from_ground_truth(component_model, target_model)
        logger.info("Initialized components from ground truth")
    else:
        logger.info("Using random component initialization")

    if task_config.init_ci == "ideal":
        initialize_ci_fns_from_ground_truth(component_model, target_model)
        logger.info("Initialized CI functions from ground truth")
    else:
        logger.info("Using random CI initialization")

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
