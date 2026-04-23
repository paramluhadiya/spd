"""Run SPD on PingPong with vector_mlp CI and ideal initialization for masking components only.

Per layer, only the 8 routing masking components are ideally initialized — the one-hot that
controls which block survives (ohj for even layers, ohi for the middle layer):

  model.0: ohj masks (8 components) → ideally init'd; ohi bias components → random
  model.2: ohi masks (8 components) → ideally init'd; ohj bias components → random
  model.4: ohj masks (8 components) → ideally init'd; ohi bias components → random

For each ideally-initialized masking component k:
  - V/U: rank-1 decomposition (V = e_{D+k}, U = W^T[D+k, :])
  - CI:  Heaviside(x_{D+k}) via GELU finite-difference on component k's private weights

The 512 computational components and the 8 non-routing indexing components per layer start
fully random (V, U, and CI) and are learned by SPD.

vector_mlp advantage: each component k has its own ParallelLinear weights, so masking CIs
only need W[k, input_k, 0:2] set — no shared hidden-dim allocation or cross-talk.
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
from spd.models.components import LinearComponents, ParallelLinear, VectorMLPCiFn
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.module_utils import expand_module_patterns
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb

N_COMPUTATIONAL = 512  # 8 src × 8 route × 8 neurons
N_OHI = 8
N_OHJ = 8
N_INDEXING = N_OHI + N_OHJ  # 16
N_TRUE_COMPONENTS = N_COMPUTATIONAL + N_INDEXING  # 528

D = 64       # Network width
NUM_BLOCKS = 8

# Which one-hot controls masking (routing) at each layer
LAYER_ROUTING: dict[str, str] = {
    "model.0": "ohj",
    "model.2": "ohi",
    "model.4": "ohj",
}


def _routing_component_range(routing: str) -> range:
    """Component indices for the given routing one-hot type."""
    if routing == "ohi":
        return range(N_COMPUTATIONAL, N_COMPUTATIONAL + N_OHI)
    return range(N_COMPUTATIONAL + N_OHI, N_TRUE_COMPONENTS)


def scale_random_components(component_model: ComponentModel, scale: float) -> None:
    """Multiply V and U by `scale` for every component.

    Run this BEFORE any ideal initialization so that ideally-initialized components
    keep their full scale. Smaller scales encourage symmetry breaking by letting
    random components grow into orthogonal directions instead of competing as
    similarly-large noise contributors at init.
    """
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)
        with torch.no_grad():
            components.V.data *= scale
            components.U.data *= scale


def _input_dim_for_component(k: int) -> int:
    """Input dimension in x corresponding to indexing component k."""
    if k < N_COMPUTATIONAL + N_OHI:
        i = k - N_COMPUTATIONAL
        return D + i           # ohi[i]
    j = k - N_COMPUTATIONAL - N_OHI
    return D + NUM_BLOCKS + j  # ohj[j]


def initialize_routing_components_from_ground_truth(
    component_model: ComponentModel,
    target_model: PingPongModel,
) -> None:
    """Initialize V and U for the 8 routing masking components per layer.

    Only initializes the one-hot group that does masking at each layer;
    the other 8 indexing components (bias group) stay at random init.
    """
    for module_name in component_model.target_module_paths:
        components = component_model.components[module_name]
        assert isinstance(components, LinearComponents)
        assert components.C >= N_TRUE_COMPONENTS

        W = component_model.target_weight(module_name)
        d_in = W.shape[1]
        assert d_in == target_model.input_dim

        routing = LAYER_ROUTING[module_name]
        mask_range = _routing_component_range(routing)

        with torch.no_grad():
            eye = torch.eye(d_in, device=components.V.device)
            for k in mask_range:
                input_dim = _input_dim_for_component(k)
                components.V.data[:, k] = eye[input_dim]
                components.U.data[k, :] = W.T[input_dim, :]



def initialize_routing_ci_fns(
    component_model: ComponentModel,
    target_model: PingPongModel,
) -> None:
    """Initialize CI functions for the 8 routing masking components per layer.

    With 2 hidden layers (ci_fn_hidden_dims: [h1, h2]) and ReLU activations:

    Layer 0: ReLU finite-difference Heaviside on x[input_k] using hidden units 0 and 1.
        h1[k, 0] = ReLU(S * x[input_k] - t)      ≈ S*x - t for x > t/S, else 0
        h1[k, 1] = ReLU(S * x[input_k] - t - 1)
        diff = h1[0] - h1[1] = exactly 0 when x=0, exactly 1 when x=1 (binary inputs).

    Layer 1: re-applies the same finite-difference on the diff ∈ {0, 1}.
        ReLU(-t) = 0 exactly, so there is no bleed when routing=0. Same S and t work.

    Output layer: diff of layer-1 units 0 and 1 ≈ H(x[input_k]).

    Only uses hidden units 0 and 1 in each layer. All other weights for component k are zeroed.
    The other (non-routing) indexing and computational components are untouched.
    """
    S = 50.0
    t = 1.0

    for module_name in component_model.target_module_paths:
        ci_fn = component_model.ci_fns[module_name]
        assert isinstance(ci_fn, VectorMLPCiFn), (
            f"Expected VectorMLPCiFn, got {type(ci_fn)}. Set ci_fn_type='vector_mlp' in config."
        )
        assert len(ci_fn.layers) == 5, (
            f"Expected 2 hidden layers (5 sublayers), got {len(ci_fn.layers)}. "
            "Set ci_fn_hidden_dims to a list of 2 values."
        )

        layer0 = ci_fn.layers[0]       # ParallelLinear(C, input_dim, h1)
        layer1 = ci_fn.layers[2]       # ParallelLinear(C, h1, h2); layers[1] is GELU
        output_layer = ci_fn.layers[4] # ParallelLinear(C, h2, 1); layers[3] is GELU
        assert isinstance(layer0, ParallelLinear)
        assert isinstance(layer1, ParallelLinear)
        assert isinstance(output_layer, ParallelLinear)

        assert layer0.W.shape[2] >= 2, f"h1={layer0.W.shape[2]} < 2"
        assert layer1.W.shape[2] >= 2, f"h2={layer1.W.shape[2]} < 2"
        assert layer0.W.shape[1] == target_model.input_dim

        routing = LAYER_ROUTING[module_name]
        mask_range = _routing_component_range(routing)

        with torch.no_grad():
            for k in mask_range:
                input_k = _input_dim_for_component(k)

                # Layer 0: Heaviside of x[input_k] in hidden units 0 and 1
                layer0.W.data[k, :, :] = 0.0
                layer0.W.data[k, input_k, 0] = S
                layer0.W.data[k, input_k, 1] = S
                layer0.b.data[k, 0] = -t
                layer0.b.data[k, 1] = -(t + 1)

                # Layer 1: same finite-diff on (h1[0] - h1[1]) ∈ {0, 1}
                # ReLU(-t) = 0 exactly, so same S and t work with no bleed.
                layer1.W.data[k, :, :] = 0.0
                layer1.W.data[k, 0, 0] = S
                layer1.W.data[k, 1, 0] = -S
                layer1.W.data[k, 0, 1] = S
                layer1.W.data[k, 1, 1] = -S
                layer1.b.data[k, 0] = -t
                layer1.b.data[k, 1] = -(t + 1)

                # Output: GELU(h2[0]) - GELU(h2[1]) ≈ H(x[input_k])
                output_layer.W.data[k, :, 0] = 0.0
                output_layer.W.data[k, 0, 0] = 1.0
                output_layer.W.data[k, 1, 0] = -1.0
                output_layer.b.data[k, 0] = 0.0


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
        experiment_tag="pingpong-vectormlp-ideal-init", evals_id=evals_id, sweep_id=sweep_id
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

    if task_config.random_init_scale != 1.0:
        scale_random_components(component_model, task_config.random_init_scale)
        logger.info(f"Scaled all V/U at init by {task_config.random_init_scale}")

    initialize_routing_components_from_ground_truth(component_model, target_model)
    logger.info(
        "Initialized routing masking components (V, U) per layer: "
        "ohj for model.0/model.4, ohi for model.2"
    )

    initialize_routing_ci_fns(component_model, target_model)
    logger.info("Initialized routing masking CI functions to Heaviside (8 per layer)")

    component_model.to(device)

    weight_deltas = component_model.calc_weight_deltas()
    total_faith = sum(torch.norm(wd).item() for wd in weight_deltas.values())
    logger.info(
        f"Post-init faithfulness (nonzero expected — computational and bias components random): "
        f"{total_faith:.6e}"
    )

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
