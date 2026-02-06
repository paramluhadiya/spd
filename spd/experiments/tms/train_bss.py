"""Initialize and save a Block-Structured Superposition (BSS) or PingPong model.

Note: These models are constructed (not trained via gradient descent),
so this script initializes a model with random circuits and verifies
the construction works correctly.
"""

import wandb

from spd.experiments.tms.bss_configs import BSSModelConfig, BSSTrainConfig
from spd.experiments.tms.bss_models import BSSModel, PingPongModel
from spd.log import logger
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import ExecutionStamp, save_file


def run_train(config: BSSTrainConfig, device: str) -> None:
    """Initialize and save a BSS or PingPong model."""
    model_cfg = config.bss_model_config
    model_type = model_cfg.model_type

    if model_type == "pingpong":
        model = PingPongModel(config=model_cfg)
    else:
        model = BSSModel(config=model_cfg)
    model.to(device)

    run_name = f"{model_type}_D{model_cfg.D}_d{model_cfg.d}_seed{config.seed}"

    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = [f"{model_type}_{model_cfg.D}-{model_cfg.d}"]
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
    logger.info(f"Verifying {model_type} construction...")
    logger.info(f"Model has {model.T} circuits, D={model.D}, d={model.d}")
    logger.info(f"Number of blocks: {model.num_blocks}")

    max_error = model.verify_all_circuits()
    logger.info(f"All {model.T} circuits verified! Max error: {max_error:.2e}")

    # Log some statistics
    stats = {
        "model_type": model_type,
        "D (network width)": model.D,
        "d (block width)": model.d,
        "T (num circuits)": model.T,
        "num_blocks": model.num_blocks,
        "B (suppression)": model_cfg.B,
        "max_verification_error": f"{max_error:.2e}",
    }
    if model_type == "pingpong":
        stats["n_layers"] = model_cfg.n_layers
        stats["input_dim"] = model_cfg.input_dim
    logger.values(msg=f"{model_type.upper()} Model Statistics", data=stats)

    # Save model
    checkpoint_name = "pingpong.pth" if model_type == "pingpong" else "bss.pth"
    model_path = out_dir / checkpoint_name
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["bss", "pingpong"], default="bss")
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    device = get_device()

    config = BSSTrainConfig(
        wandb_project=args.wandb_project,
        bss_model_config=BSSModelConfig(
            D=args.D,
            d=args.d,
            B=10.0,  # Small enough to not dominate SPD losses, large enough to suppress after ReLU
            device=device,
            n_layers=args.n_layers,
            model_type=args.model_type,
        ),
        seed=args.seed,
    )

    set_seed(config.seed)
    run_train(config, device)
