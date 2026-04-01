"""Dynamic hyperparameter sweep using Optuna with Bayesian optimization and pruning.

Runs trials across multiple GPUs, monitors metrics via JSONL files,
and prunes clearly bad runs early. Uses MOTPESampler for multi-objective
Bayesian optimization.

Usage:
    python scripts/optuna_sweep.py --n_gpus 8 --n_trials 50
    python scripts/optuna_sweep.py --n_gpus 8 --n_trials 50 --resume  # resume existing study
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue

import optuna

from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT, SPD_OUT_DIR

METRICS_DIR = Path("/tmp/spd_optuna_metrics")

# Metric keys in the JSONL files (without eval/ prefix)
L0_KEYS = [f"l0/0.1_model.{i}" for i in [0, 2, 4]]
FAITH_KEY = "loss/FaithfulnessLoss"
PGD_KEY = "loss/PGDReconLoss"

# Ideal targets
IDEAL_L0 = {"l0/0.1_model.0": 9.5, "l0/0.1_model.2": 6.0, "l0/0.1_model.4": 5.8}

# Pruning thresholds
PRUNE_AFTER_STEPS = 8000
PRUNE_PGD_THRESHOLD = 2.0
PRUNE_FAITH_THRESHOLD = 0.1

POLL_INTERVAL_SECONDS = 30


def read_metrics_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line]


def compute_l0_distance(metrics: dict) -> float:
    total = 0.0
    for key, target in IDEAL_L0.items():
        val = metrics.get(key)
        if val is not None:
            total += abs(float(val) - target)
    return total


def create_objective(experiment: str, gpu_queue: Queue):
    exp_config = EXPERIMENT_REGISTRY[experiment]
    script_path = REPO_ROOT / exp_config.decomp_script
    base_config_path = REPO_ROOT / exp_config.config_path

    from spd.configs import Config
    from spd.utils.run_utils import apply_nested_updates

    base_config = Config.from_file(base_config_path)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # --- Suggest hyperparameters ---
        lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
        imp_coeff = trial.suggest_float("imp_coeff", 5e-5, 1e-3, log=True)
        p_anneal_end_frac = trial.suggest_float("p_anneal_end_frac", 0.2, 1.0)
        beta = trial.suggest_float("beta", 0.05, 0.5)
        pgd_coeff = trial.suggest_float("pgd_coeff", 10.0, 50.0)

        # --- Build config ---
        overrides = {
            "lr_schedule.start_val": lr,
            "loss_metric_configs.ImportanceMinimalityLoss.coeff": imp_coeff,
            "loss_metric_configs.ImportanceMinimalityLoss.p_anneal_end_frac": p_anneal_end_frac,
            "loss_metric_configs.ImportanceMinimalityLoss.beta": beta,
            "loss_metric_configs.PGDReconLoss.coeff": pgd_coeff,
            "loss_metric_configs.PGDReconSubsetLoss.coeff": pgd_coeff,
            "task_config.init_computational_components": "random",
            "task_config.init_indexing_components": "random",
            "task_config.init_computational_ci": "random",
            "task_config.init_indexing_ci": "random",
        }

        config_dict = base_config.model_dump(mode="json")
        config_dict = apply_nested_updates(config_dict, overrides)
        config_dict["wandb_run_name"] = (
            f"optuna-t{trial.number}-lr{lr:.0e}-imp{imp_coeff:.0e}-ann{p_anneal_end_frac:.1f}"
        )

        # --- GPU assignment via queue ---
        gpu_id = gpu_queue.get()
        metrics_path = METRICS_DIR / f"trial_{trial.number}.jsonl"
        if metrics_path.exists():
            metrics_path.unlink()

        try:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "OPTUNA_METRICS_PATH": str(metrics_path),
            }

            config_json = "json:" + json.dumps(config_dict)
            proc = subprocess.Popen(
                [sys.executable, str(script_path), "--config_json", config_json],
                env=env,
            )

            # --- Monitor and prune ---
            last_step = 0
            while proc.poll() is None:
                time.sleep(POLL_INTERVAL_SECONDS)

                all_metrics = read_metrics_file(metrics_path)
                if not all_metrics:
                    continue

                latest = all_metrics[-1]
                step = latest.get("step", 0)
                if step <= last_step:
                    continue
                last_step = step

                pgd = latest.get(PGD_KEY, 999.0)
                faith = latest.get(FAITH_KEY, 999.0)

                # Report composite metric for Optuna's pruner
                l0_dist = compute_l0_distance(latest)
                trial.report(l0_dist + 10 * pgd, step=step)

                # Hard prune: clearly bad runs
                if step >= PRUNE_AFTER_STEPS:
                    if pgd > PRUNE_PGD_THRESHOLD:
                        print(f"  Trial {trial.number}: PRUNED (PGD={pgd:.3f} > {PRUNE_PGD_THRESHOLD} at step {step})")
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=30)
                        raise optuna.TrialPruned()
                    if faith > PRUNE_FAITH_THRESHOLD:
                        print(f"  Trial {trial.number}: PRUNED (faith={faith:.3e} > {PRUNE_FAITH_THRESHOLD} at step {step})")
                        proc.send_signal(signal.SIGTERM)
                        proc.wait(timeout=30)
                        raise optuna.TrialPruned()

                # Optuna's statistical pruning
                if trial.should_prune():
                    print(f"  Trial {trial.number}: PRUNED by Optuna at step {step}")
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=30)
                    raise optuna.TrialPruned()

            if proc.returncode != 0:
                raise optuna.TrialPruned()

            # --- Final metrics ---
            final_metrics = read_metrics_file(metrics_path)
            assert final_metrics, f"No metrics for trial {trial.number}"
            final = final_metrics[-1]

            l0_dist = compute_l0_distance(final)
            pgd = final.get(PGD_KEY, float("inf"))

            print(f"  Trial {trial.number}: DONE (L0_dist={l0_dist:.2f}, PGD={pgd:.4f})")
            return l0_dist, float(pgd)

        finally:
            gpu_queue.put(gpu_id)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, default="pingpong_probe_64-8")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default="pingpong_random_init")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    assert args.experiment in EXPERIMENT_REGISTRY

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # GPU queue for round-robin assignment
    gpu_queue: Queue = Queue()
    for i in range(args.n_gpus):
        gpu_queue.put(i)

    db_path = SPD_OUT_DIR / "optuna_studies.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["minimize", "minimize"],  # l0_distance, pgd_loss
        sampler=optuna.samplers.TPESampler(n_startup_trials=8),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=4000,
            max_resource=25000,
            reduction_factor=3,
        ),
        load_if_exists=args.resume,
    )

    objective = create_objective(args.experiment, gpu_queue)

    print(f"Starting Optuna sweep: {args.n_trials} trials across {args.n_gpus} GPUs")
    print(f"Study: {args.study_name} (storage: {db_path})")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_gpus,
        show_progress_bar=True,
    )

    # Print Pareto front
    print("\n=== Pareto-optimal trials ===")
    for trial in study.best_trials:
        assert trial.values is not None
        print(
            f"  Trial {trial.number}: L0_dist={trial.values[0]:.2f}, PGD={trial.values[1]:.4f}"
        )
        print(f"    Params: {trial.params}")

    # Print top 5 by L0 distance
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.values[0] if t.values else float("inf"))
    print(f"\n=== Top 5 by L0 distance ({len(completed)} completed) ===")
    for t in completed[:5]:
        assert t.values is not None
        print(
            f"  Trial {t.number}: L0_dist={t.values[0]:.2f}, PGD={t.values[1]:.4f}, "
            f"lr={t.params['lr']:.0e}, imp={t.params['imp_coeff']:.0e}, "
            f"anneal={t.params['p_anneal_end_frac']:.2f}, beta={t.params['beta']:.3f}"
        )

    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nSummary: {len(completed)} completed, {len(pruned)} pruned")


if __name__ == "__main__":
    main()
