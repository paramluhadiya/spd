"""Run SPD on a PingPong computation in superposition model.

The PingPong model has T = (D/d)^2 circuits, each identified by (i, j) block indices.
Each circuit is a 3-layer MLP that ping-pongs between blocks i and j.
"""

import json
from pathlib import Path

import fire
import torch
import wandb
from torch import Tensor
from torch.utils.data import Dataset

from spd.configs import Config, PingPongTaskConfig
from spd.experiments.tms.bss_models import PingPongModel, PingPongTargetRunInfo
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import save_pre_run_info, set_seed
from spd.utils.run_utils import setup_decomposition_run
from spd.utils.wandb_utils import init_wandb


class PingPongDataset(Dataset[Tensor]):
    """Dataset for PingPong model that generates input tensors.

    For each sample:
    - Randomly selects one of T circuits (identified by i, j)
    - Generates a random input for block i
    - Returns tensor of shape (batch, D + 2*num_blocks) where:
      - [:, :D] is x_block (input in block coordinates, only block i is non-zero)
      - [:, D:D+num_blocks] is one_hot_i
      - [:, D+num_blocks:] is one_hot_j
    """

    def __init__(
        self,
        D: int,
        d: int,
        num_blocks: int,
        device: str,
        value_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.D = D
        self.d = d
        self.num_blocks = num_blocks
        self.T = num_blocks**2
        self.device = device
        self.value_range = value_range
        self.input_dim = D + 2 * num_blocks

    def __len__(self) -> int:
        return 2**31

    def generate_batch(self, batch_size: int) -> Tensor:
        """Generate a batch of PingPong input tensors."""
        min_val, max_val = self.value_range
        D, d, num_blocks = self.D, self.d, self.num_blocks

        # Randomly select circuits (i, j pairs)
        circuit_ids = torch.randint(0, self.T, (batch_size,), device=self.device)
        i_indices = circuit_ids // num_blocks
        j_indices = circuit_ids % num_blocks

        # Initialize full input tensor
        x = torch.zeros(batch_size, self.input_dim, device=self.device)

        # Fill x_block: put random values in block i for each sample (vectorized)
        values = torch.rand(batch_size, d, device=self.device) * (max_val - min_val) + min_val
        block_starts = d * i_indices  # (batch_size,)
        col_indices = block_starts.unsqueeze(1) + torch.arange(d, device=self.device)  # (batch_size, d)
        x.scatter_(dim=1, index=col_indices, src=values)

        # Fill one_hot_i and one_hot_j using advanced indexing
        batch_indices = torch.arange(batch_size, device=self.device)
        x[batch_indices, D + i_indices] = 1.0
        x[batch_indices, D + num_blocks + j_indices] = 1.0

        return x


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
        experiment_tag="pingpong", evals_id=evals_id, sweep_id=sweep_id
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
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
