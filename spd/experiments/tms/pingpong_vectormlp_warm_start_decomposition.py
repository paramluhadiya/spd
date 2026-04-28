"""Continue an existing PingPong SPD run under a different training regime.

Use case: warm-start from a converged β=0 SPD checkpoint (control run with no
log-sparsity pressure) and continue with BetaInfImportanceMinimalityLoss to
test whether circuit specificity emerges purely under sparsity pressure.

The warm-start path lives in `task_config.warm_start_path` (e.g.
`wandb:.../<run_id>` or a local final-checkpoint dir). All of the V, U, and
CI-fn weights from that run are loaded into a fresh ComponentModel built from
the new config; training then proceeds from there.

The new config controls *training* parameters only (loss, LR, steps). It must
agree with the warm-start checkpoint on *structural* parameters (module_info,
ci_fn_type, ci_fn_hidden_dims, sigmoid_type, pretrained_model_path), otherwise
state_dict load would fail.

Usage (direct):
    python -m spd.experiments.tms.pingpong_vectormlp_warm_start_decomposition \\
        --config_path spd/experiments/tms/pingpong_vectormlp_warm_start_64-8_config.yaml

Usage (via spd-run):
    spd-run --experiments pingpong_vectormlp_warm_start_64-8
"""

import json
from pathlib import Path

import fire
import torch
import wandb

from spd.configs import Config, PingPongTaskConfig
from spd.experiments.tms.bss_models import PingPongModel
from spd.experiments.tms.pingpong_decomposition import PingPongDataset
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb


def _assert_structural_match(new: Config, prev: Config) -> None:
    """Fail fast if anything that determines tensor shapes differs between configs."""
    assert new.pretrained_model_path == prev.pretrained_model_path, (
        f"pretrained_model_path mismatch: {new.pretrained_model_path} vs {prev.pretrained_model_path}"
    )
    assert new.pretrained_model_class == prev.pretrained_model_class, (
        f"pretrained_model_class mismatch"
    )
    assert new.ci_fn_type == prev.ci_fn_type, f"ci_fn_type mismatch: {new.ci_fn_type} vs {prev.ci_fn_type}"
    assert new.ci_fn_hidden_dims == prev.ci_fn_hidden_dims, (
        f"ci_fn_hidden_dims mismatch: {new.ci_fn_hidden_dims} vs {prev.ci_fn_hidden_dims}"
    )
    assert new.sigmoid_type == prev.sigmoid_type, (
        f"sigmoid_type mismatch: {new.sigmoid_type} vs {prev.sigmoid_type}"
    )
    assert new.use_delta_component == prev.use_delta_component, "use_delta_component mismatch"
    new_modules = [(m.module_pattern, m.C) for m in new.module_info]
    prev_modules = [(m.module_pattern, m.C) for m in prev.module_info]
    assert new_modules == prev_modules, (
        f"module_info mismatch: {new_modules} vs {prev_modules}"
    )


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

    task_config = config.task_config
    assert isinstance(task_config, PingPongTaskConfig)
    warm_start_path = task_config.warm_start_path
    assert warm_start_path is not None, (
        "task_config.warm_start_path is required (path to prior SPD run)"
    )

    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")
    set_seed(config.seed)

    out_dir, run_id, tags = setup_decomposition_run(
        experiment_tag="pingpong-vectormlp-warm-start", evals_id=evals_id, sweep_id=sweep_id
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

    logger.info(f"Loading warm-start SPD checkpoint from {warm_start_path}")
    warm_start_run_info = SPDRunInfo.from_path(warm_start_path)
    _assert_structural_match(config, warm_start_run_info.config)

    component_model = ComponentModel.from_run_info(warm_start_run_info)
    target_model = component_model.target_model
    assert isinstance(target_model, PingPongModel)
    target_model.requires_grad_(False)
    component_model.to(device)

    logger.info(
        f"Loaded warm-start ComponentModel: D={target_model.D}, d={target_model.d}, "
        f"T={target_model.T}, n_layers={target_model.n_layers}"
    )

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=target_model.config,
        task_name=task_config.task_name,
    )

    weight_deltas = component_model.calc_weight_deltas()
    total_faith = sum(torch.norm(wd).item() for wd in weight_deltas.values())
    logger.info(f"Post-warm-start faithfulness norm: {total_faith:.6e}")

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
