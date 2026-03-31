"""Run a sweep locally across multiple GPUs without SLURM.

Usage:
    python scripts/run_local_sweep.py pingpong_probe_64-8 \
        spd/experiments/tms/pingpong_stage2_sweep_params.yaml \
        --n_gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from spd.configs import Config
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.utils.run_utils import apply_nested_updates, generate_grid_combinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=str)
    parser.add_argument("sweep_params_path", type=str)
    parser.add_argument("--n_gpus", type=int, default=8)
    args = parser.parse_args()

    assert args.experiment in EXPERIMENT_REGISTRY, f"Unknown experiment: {args.experiment}"
    exp_config = EXPERIMENT_REGISTRY[args.experiment]

    with open(args.sweep_params_path) as f:
        sweep_params = yaml.safe_load(f)

    # Merge global + experiment-specific params
    params = dict(sweep_params.get("global", {}))
    if args.experiment in sweep_params:
        params.update(sweep_params[args.experiment])

    combinations = generate_grid_combinations(params)
    print(f"Generated {len(combinations)} configs across {args.n_gpus} GPUs")

    base_config = Config.from_file(exp_config.config_path)
    script_path = REPO_ROOT / exp_config.decomp_script

    # Write each config to a temp file to avoid shell quoting issues
    config_files: list[tuple[Path, dict[str, str], int]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="spd_sweep_"))

    for i, param_combo in enumerate(combinations):
        config_dict = base_config.model_dump(mode="json")
        config_dict = apply_nested_updates(config_dict, param_combo)

        name_parts = []
        for k, v in param_combo.items():
            short_key = k.split(".")[-1]
            name_parts.append(f"{short_key}-{v}")
        config_dict["wandb_run_name"] = f"{args.experiment}-{'-'.join(name_parts)}"

        config_path = tmp_dir / f"config_{i:03d}.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

        gpu_id = i % args.n_gpus
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
        config_files.append((config_path, env, i))

    # Run in batches of n_gpus
    batch_size = args.n_gpus
    for batch_start in range(0, len(config_files), batch_size):
        batch = config_files[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(config_files))
        print(f"\n--- Batch {batch_start // batch_size + 1}: "
              f"runs {batch_start + 1}-{batch_end} of {len(config_files)} ---")

        procs: list[subprocess.Popen[bytes]] = []
        for config_path, env, idx in batch:
            full_env = {**os.environ, **env}
            cmd = [
                sys.executable,
                str(script_path),
                "--config_json",
                "json:" + config_path.read_text(),
            ]
            proc = subprocess.Popen(cmd, env=full_env)
            procs.append(proc)

        for proc in procs:
            proc.wait()
            if proc.returncode != 0:
                print(f"WARNING: Process {proc.pid} exited with code {proc.returncode}")

    print(f"\nAll runs complete. Temp configs in: {tmp_dir}")


if __name__ == "__main__":
    main()
