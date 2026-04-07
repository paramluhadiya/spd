"""Submit Optuna sweep to SLURM.

Launches N worker nodes, each running the Optuna sweep script with 8 GPUs.
All workers share the same Optuna study via SQLite on the shared filesystem.

Usage:
    python scripts/submit_optuna_sweep.py                    # 1 node (8 GPUs)
    python scripts/submit_optuna_sweep.py --n_nodes 2        # 2 nodes (16 GPUs)
    python scripts/submit_optuna_sweep.py --n_trials 100     # more trials
    python scripts/submit_optuna_sweep.py --init_comp random --init_idx ideal \
        --init_comp_ci random --init_idx_ci random
"""

import argparse

from spd.settings import REPO_ROOT, SPD_OUT_DIR
from spd.utils.slurm import SlurmConfig, generate_script, submit_slurm_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--n_trials_per_node", type=int, default=25)
    parser.add_argument("--experiment", type=str, default="pingpong_probe_64-8")
    parser.add_argument("--study_name", type=str, default="pingpong_random_init")
    parser.add_argument("--partition", type=str, default="compute")
    parser.add_argument("--time", type=str, default="12:00:00")
    # Init config passthrough
    parser.add_argument("--init_comp", type=str, default="random",
                        choices=["ideal", "random"])
    parser.add_argument("--init_idx", type=str, default="random",
                        choices=["ideal", "random"])
    parser.add_argument("--init_comp_ci", type=str, default="random",
                        choices=["ideal", "random"])
    parser.add_argument("--init_idx_ci", type=str, default="random",
                        choices=["ideal", "random"])
    args = parser.parse_args()

    db_path = SPD_OUT_DIR / "optuna_studies.db"
    n_gpus_per_node = 8

    total_gpus = args.n_nodes * n_gpus_per_node
    total_trials = args.n_nodes * args.n_trials_per_node
    init_desc = f"comp={args.init_comp}, idx={args.init_idx}, comp_ci={args.init_comp_ci}, idx_ci={args.init_idx_ci}"
    print(f"Submitting {args.n_nodes} node(s) × {n_gpus_per_node} GPUs = {total_gpus} GPUs")
    print(f"Total trials: {total_trials} ({args.n_trials_per_node} per node)")
    print(f"Init: {init_desc}")
    print(f"Study DB: {db_path}")

    for node_idx in range(args.n_nodes):
        config = SlurmConfig(
            job_name=f"spd-optuna-{node_idx}",
            partition=args.partition,
            n_gpus=n_gpus_per_node,
            time=args.time,
        )

        command = (
            f"python {REPO_ROOT}/scripts/optuna_sweep.py "
            f"--experiment {args.experiment} "
            f"--n_gpus {n_gpus_per_node} "
            f"--n_trials {args.n_trials_per_node} "
            f"--study_name {args.study_name} "
            f"--init_comp {args.init_comp} "
            f"--init_idx {args.init_idx} "
            f"--init_comp_ci {args.init_comp_ci} "
            f"--init_idx_ci {args.init_idx_ci} "
            f"--resume"
        )

        script_content = generate_script(config, command)

        result = submit_slurm_job(
            script_content,
            script_name_prefix=f"optuna_sweep_{args.study_name}",
        )

        print(f"  Node {node_idx}: Job {result.job_id}, logs: {result.log_pattern}")

    print(f"\nAll {args.n_nodes} job(s) submitted.")


if __name__ == "__main__":
    main()
