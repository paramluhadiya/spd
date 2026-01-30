"""Run SPD on a Block-Structured Superposition (BSS) model.

The BSS model has T = (D/d)^2 circuits, each with its own (d x d) weight matrix
and (d,) bias. The model demonstrates computation in superposition where exactly
one circuit is active per forward pass.
"""

import json
from pathlib import Path

import fire
import torch
import wandb
from torch import Tensor
from torch.utils.data import Dataset

from spd.configs import BSSTaskConfig, Config
from spd.experiments.tms.bss_models import BSSModel, BSSTargetRunInfo
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb


class BSSDataset(Dataset[Tensor]):
    """Dataset for BSS model that generates combined input tensors.

    For each sample:
    - Randomly selects one of T circuits
    - Generates a random input for that circuit's block
    - Returns a combined tensor of shape (batch, D+1) where:
      - [:, :D] is x_block (input in block coordinates)
      - [:, D] is the active circuit index (as float for tensor compatibility)
    """

    def __init__(
        self,
        D: int,
        d: int,
        device: str,
        f: dict[int, int],
        value_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.D = D
        self.d = d
        self.T = (D // d) ** 2
        self.device = device
        self.f = f
        self.value_range = value_range

    def __len__(self) -> int:
        return 2**31

    def generate_batch(self, batch_size: int) -> Tensor:
        """Generate a batch of combined (x_block, active_circuit) tensors."""
        min_val, max_val = self.value_range

        # Randomly select circuits for each sample
        active_circuits = torch.randint(0, self.T, (batch_size,), device=self.device)

        # Initialize combined tensor: x_block (D) + circuit_id (1)
        combined = torch.zeros(batch_size, self.D + 1, device=self.device)

        # For each circuit, fill in the appropriate block with random values
        for b in range(batch_size):
            circuit = int(active_circuits[b].item())
            block = self.f[circuit]
            block_start = self.d * block
            block_end = block_start + self.d

            # Generate random values for this block
            values = torch.rand(self.d, device=self.device) * (max_val - min_val) + min_val
            combined[b, block_start:block_end] = values

            # Store circuit ID in last position
            combined[b, self.D] = float(circuit)

        return combined


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
        experiment_tag="bss", evals_id=evals_id, sweep_id=sweep_id
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
    assert isinstance(task_config, BSSTaskConfig)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_run_info = BSSTargetRunInfo.from_path(config.pretrained_model_path)
    target_model = BSSModel.from_run_info(target_run_info)
    target_model = target_model.to(device)
    target_model.eval()

    logger.info(f"Loaded BSS model: D={target_model.D}, d={target_model.d}, T={target_model.T}")

    save_pre_run_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        spd_config=config,
        sweep_params=sweep_params,
        target_model=target_model,
        train_config=target_model.config,
        task_name=config.task_config.task_name,
    )

    dataset = BSSDataset(
        D=target_model.D,
        d=target_model.d,
        device=device,
        f=target_model.f,
        value_range=(0.0, 1.0),
    )
    train_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.microbatch_size, shuffle=False
    )
    eval_loader = DatasetGeneratedDataLoader(
        dataset, batch_size=config.eval_batch_size, shuffle=False
    )

    # No tied weights for BSS
    tied_weights = None

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=tied_weights,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
