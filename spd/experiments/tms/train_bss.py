"""Initialize and save a Block-Structured Superposition (BSS) model.

Note: BSS models are constructed (not trained via gradient descent),
so this script initializes a model with random circuits and verifies
the construction works correctly.
"""

import wandb

from spd.experiments.tms.bss_configs import BSSModelConfig, BSSTrainConfig
from spd.experiments.tms.bss_models import BSSModel
from spd.log import logger
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import ExecutionStamp, save_file


def run_train(config: BSSTrainConfig, device: str) -> None:
    """Initialize and save a BSS model."""
    model = BSSModel(config=config.bss_model_config)
    model.to(device)

    model_cfg = config.bss_model_config
    run_name = f"bss_D{model_cfg.D}_d{model_cfg.d}_seed{config.seed}"

    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = [f"bss_{model_cfg.D}-{model_cfg.d}"]
        wandb.init(
            id=execution_stamp.run_id,
            project=config.wandb_project,
            name=run_name,
            tags=tags,
        )

    # Save config
    config_path = out_dir / "bss_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    # Verify construction
    logger.info("Verifying BSS construction...")
    logger.info(f"Model has {model.T} circuits, D={model.D}, d={model.d}")
    logger.info(f"Number of blocks: {model.num_blocks}")

    max_error = model.verify_all_circuits()
    logger.info(f"All {model.T} circuits verified! Max error: {max_error:.2e}")

    # Log some statistics
    logger.values(
        msg="BSS Model Statistics",
        data={
            "D (network width)": model.D,
            "d (block width)": model.d,
            "T (num circuits)": model.T,
            "num_blocks": model.num_blocks,
            "B (suppression)": model_cfg.B,
            "max_verification_error": f"{max_error:.2e}",
        },
    )

    # Save model
    model_path = out_dir / "bss.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    device = get_device()

    # BSS 64-8: 64 network width, 8 block width = 64 circuits
    config = BSSTrainConfig(
        wandb_project="spd",
        bss_model_config=BSSModelConfig(
            D=64,
            d=8,
            B=1e6,
            device=device,
        ),
        seed=0,
    )

    set_seed(config.seed)
    run_train(config, device)
