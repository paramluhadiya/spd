
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

N_COMPUTATIONAL = 64  # One per neuron in the D-dimensional computational block
N_INDEXING = 16  # 8 one_hot_i + 8 one_hot_j
N_TRUE_COMPONENTS = N_COMPUTATIONAL + N_INDEXING  # 80 total


def initialize_components_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
    component_indices: range,
) -> None:
    """Initialize V and U matrices to the column-wise rank-1 decomposition.

    For the specified component indices k, sets V[:, k] = e_k and U[k, :] = W^T[k, :].
    Components outside this range are not modified.
    """
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)
        assert components.C >= N_TRUE_COMPONENTS, f"C={components.C} < {N_TRUE_COMPONENTS}"

        W = component_model.target_weight(module_name)
        d_in = W.shape[1]
        assert d_in == target_model.input_dim

        with torch.no_grad():
            eye = torch.eye(d_in, device=components.V.device)
            for k in component_indices:
                components.V.data[:, k] = eye[k]
                components.U.data[k, :] = W.T[k, :]


def scale_down_unused_components(component_model: ComponentModel) -> None:
    """Scale down components beyond the true decomposition so they start small."""
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)
        with torch.no_grad():
            components.V.data[:, N_TRUE_COMPONENTS:] *= 0.01
            components.U.data[N_TRUE_COMPONENTS:, :] *= 0.01


def initialize_ci_fns_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
    component_indices: list[int],
) -> None:
    """Initialize CI functions so that CI_k(x) ≈ 1 when x_k > 0, ≈ 0 when x_k = 0.

    Only modifies weights for the specified component indices. Non-ideal components
    retain their random initialization from the constructor.

    Each ideal component k uses dedicated hidden dims 2k and 2k+1. To prevent
    cross-talk, we zero: (1) row k of layer0 (input dim k doesn't leak to random
    hidden dims), (2) columns 2k,2k+1 of layer0 (random inputs don't affect ideal
    hidden dims), (3) rows 2k,2k+1 of layer1 (ideal hidden dims don't affect random
    outputs). Then we set the ideal weights.

    Uses a GELU finite-difference trick:
      output_k = GELU(S*x_k - t) - GELU(S*x_k - (t+1)) ≈ 1 for x_k > 0
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

        S = 50.0
        t = 1.0

        with torch.no_grad():
            # Set off-bias for unused components (80..C-1) — these are always off
            C = layer1.W.shape[1]
            layer1.b.data[N_TRUE_COMPONENTS:C] = -3.0

            for k in component_indices:
                # Zero row k of layer0: prevent input dim k from leaking to random hidden dims
                layer0.W.data[k, :] = 0.0
                # Zero columns 2k,2k+1 of layer0: isolate ideal hidden dims from random inputs
                layer0.W.data[:, 2 * k] = 0.0
                layer0.W.data[:, 2 * k + 1] = 0.0
                # Zero rows 2k,2k+1 of layer1: isolate ideal hidden dims from random outputs
                layer1.W.data[2 * k, :] = 0.0
                layer1.W.data[2 * k + 1, :] = 0.0

                # Set ideal weights
                layer0.W.data[k, 2 * k] = S
                layer0.W.data[k, 2 * k + 1] = S
                layer0.b.data[2 * k] = -t
                layer0.b.data[2 * k + 1] = -(t + 1)

                layer1.W.data[2 * k, k] = 1.0
                layer1.W.data[2 * k + 1, k] = -1.0
                layer1.b.data[k] = 0.0


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

    computational_range = range(0, N_COMPUTATIONAL)
    indexing_range = range(N_COMPUTATIONAL, N_TRUE_COMPONENTS)

    if task_config.init_computational_components == "ideal":
        initialize_components_from_ground_truth(component_model, target_model, computational_range)
        logger.info("Initialized computational components (0-63) from ground truth")
    if task_config.init_indexing_components == "ideal":
        initialize_components_from_ground_truth(component_model, target_model, indexing_range)
        logger.info("Initialized indexing components (64-79) from ground truth")
    scale_down_unused_components(component_model)

    ideal_ci_indices: list[int] = []
    if task_config.init_computational_ci == "ideal":
        ideal_ci_indices.extend(computational_range)
        logger.info("Will initialize computational CI (0-63) from ground truth")
    if task_config.init_indexing_ci == "ideal":
        ideal_ci_indices.extend(indexing_range)
        logger.info("Will initialize indexing CI (64-79) from ground truth")
    if ideal_ci_indices:
        initialize_ci_fns_from_ground_truth(component_model, target_model, ideal_ci_indices)

    logger.info(
        f"Init config: computational_components={task_config.init_computational_components}, "
        f"indexing_components={task_config.init_indexing_components}, "
        f"computational_ci={task_config.init_computational_ci}, "
        f"indexing_ci={task_config.init_indexing_ci}"
    )

    component_model.to(device)

    weight_deltas = component_model.calc_weight_deltas()
    total_faith = sum(torch.norm(wd).item() for wd in weight_deltas.values())
    both_ideal = (
        task_config.init_computational_components == "ideal"
        and task_config.init_indexing_components == "ideal"
    )
    expected = "~0 if both component groups ideal" if both_ideal else "nonzero (partial init)"
    logger.info(f"Post-init faithfulness ({expected}): {total_faith:.6e}")

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
