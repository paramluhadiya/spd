"""Submit long runs for the top Optuna trial configs.

Takes the best HP configs from an Optuna probe sweep and runs them for more steps.

Usage:
    python scripts/submit_top_optuna_runs.py                    # 200k steps, top 5
    python scripts/submit_top_optuna_runs.py --steps 100000             # 100k steps
    python scripts/submit_top_optuna_runs.py --n_top 3                  # only top 3
    python scripts/submit_top_optuna_runs.py --pgd_coeff_override 250   # override PGD coeff
"""

import argparse
import json
import sys

from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT, SPD_OUT_DIR
from spd.utils.slurm import SlurmConfig, generate_script, submit_slurm_job

# Top Optuna trial configs from pingpong_ideal_idx_random_comp sweep
# Ranked by L0 distance at 25k steps
TOP_CONFIGS = [
    {"lr": 0.000705359701911008, "imp_coeff": 0.0004377856582167776, "beta": 0.1764765633645523, "pgd_coeff": 23.344071640155548, "label": "t2"},
    {"lr": 0.0006508592064132907, "imp_coeff": 0.00023619466268288205, "beta": 0.21285136992277048, "pgd_coeff": 16.566128663446236, "label": "t1"},
    {"lr": 0.000612977254640629, "imp_coeff": 0.0001618953410257415, "beta": 0.22028839910593928, "pgd_coeff": 15.24932195105984, "label": "t20"},
    {"lr": 0.0004341069584228468, "imp_coeff": 0.00025960083734623854, "beta": 0.24170334224725792, "pgd_coeff": 28.100016681316877, "label": "t14"},
    {"lr": 0.0005795776820107968, "imp_coeff": 0.000342995883107671, "beta": 0.2238393315089323, "pgd_coeff": 33.36467480278006, "label": "t23"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, default="pingpong_probe_64-8")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--n_top", type=int, default=5)
    parser.add_argument("--pgd_coeff_override", type=float, default=None,
                        help="Override PGD coeff for all runs (e.g. 250)")
    parser.add_argument("--partition", type=str, default="compute")
    parser.add_argument("--time", type=str, default="48:00:00")
    # Init config
    parser.add_argument("--init_comp", type=str, default="random",
                        choices=["ideal", "random"])
    parser.add_argument("--init_idx", type=str, default="ideal",
                        choices=["ideal", "random"])
    parser.add_argument("--init_comp_ci", type=str, default="random",
                        choices=["ideal", "random"])
    parser.add_argument("--init_idx_ci", type=str, default="random",
                        choices=["ideal", "random"])
    args = parser.parse_args()

    assert args.experiment in EXPERIMENT_REGISTRY
    exp_config = EXPERIMENT_REGISTRY[args.experiment]
    script_path = REPO_ROOT / exp_config.decomp_script
    base_config_path = REPO_ROOT / exp_config.config_path

    from spd.configs import Config
    from spd.utils.run_utils import apply_nested_updates

    base_config = Config.from_file(base_config_path)
    configs_to_run = TOP_CONFIGS[: args.n_top]

    pgd_suffix = f"_pgd{int(args.pgd_coeff_override)}" if args.pgd_coeff_override else ""
    wandb_group = f"pingpong_long_{args.steps // 1000}k{pgd_suffix}"
    print(f"Submitting {len(configs_to_run)} runs @ {args.steps} steps")
    if args.pgd_coeff_override:
        print(f"PGD coeff override: {args.pgd_coeff_override}")
    print(f"WandB group: {wandb_group}")

    for cfg in configs_to_run:
        pgd_coeff = args.pgd_coeff_override if args.pgd_coeff_override else cfg["pgd_coeff"]
        overrides = {
            "steps": args.steps,
            "lr_schedule.start_val": cfg["lr"],
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": cfg["imp_coeff"],
            "loss_metric_configs.ImportanceMinimalityLoss.p_anneal_end_frac": 1.0,
            "loss_metric_configs.ImportanceMinimalityLoss.beta": cfg["beta"],
            "loss_metric_configs.PGDReconLoss.coeff": pgd_coeff,
            "loss_metric_configs.PGDReconSubsetLoss.coeff": pgd_coeff,
            "task_config.init_computational_components": args.init_comp,
            "task_config.init_indexing_components": args.init_idx,
            "task_config.init_computational_ci": args.init_comp_ci,
            "task_config.init_indexing_ci": args.init_idx_ci,
            "save_freq": 50_000,
            "save_checkpoints_to_wandb": True,
        }

        config_dict = base_config.model_dump(mode="json")
        config_dict = apply_nested_updates(config_dict, overrides)
        label = cfg["label"]
        config_dict["wandb_run_name"] = (
            f"long-{label}-lr{cfg['lr']:.0e}-imp{cfg['imp_coeff']:.0e}"
            f"-b{cfg['beta']:.2f}-pgd{pgd_coeff:.0f}"
        )

        config_json = "json:" + json.dumps(config_dict)

        slurm_config = SlurmConfig(
            job_name=f"spd-long-{label}",
            partition=args.partition,
            n_gpus=1,
            time=args.time,
        )

        command = (
            f"WANDB_RUN_GROUP={wandb_group} "
            f"python {script_path} --config_json '{config_json}'"
        )

        script_content = generate_script(slurm_config, command)
        result = submit_slurm_job(
            script_content,
            script_name_prefix=f"long_run_{label}",
        )

        print(
            f"  {label}: lr={cfg['lr']:.0e}, imp={cfg['imp_coeff']:.0e}, "
            f"beta={cfg['beta']:.3f}, pgd={pgd_coeff:.0f}  "
            f"→ Job {result.job_id}"
        )

    print(f"\nAll {len(configs_to_run)} jobs submitted.")


if __name__ == "__main__":
    main()
