"""Score PingPong sweep runs by L0 proximity and trajectory.

Pulls metrics from WandB, ranks configs by how well they approach the ideal
L0 targets while maintaining low faithfulness and PGD reconstruction loss.

Usage:
    python scripts/score_pingpong_sweep.py <run_id>
    python scripts/score_pingpong_sweep.py <run_id> --project paramluhadiya/spd
"""

import argparse

import numpy as np
import wandb

IDEAL_L0 = {"model.0": 9.5, "model.2": 6.0, "model.4": 5.8}
L0_KEYS = [f"eval/l0/0.1_model.{i}" for i in [0, 2, 4]]
LAYER_NAMES = ["model.0", "model.2", "model.4"]

FAITH_KEY = "eval/loss/FaithfulnessLoss"
PGD_KEY = "eval/loss/PGDReconLoss"

# Disqualification thresholds
MAX_FAITH = 1e-3
MAX_PGD = 0.001

# Scoring weights
SLOPE_WEIGHT = 5.0  # How much to reward negative L0 slope


def fetch_run_metrics(run: wandb.apis.public.Run) -> dict | None:
    """Fetch L0, faithfulness, and PGD history from a run."""
    history = run.scan_history(keys=L0_KEYS + [FAITH_KEY, PGD_KEY])
    rows = list(history)
    if not rows:
        return None

    l0_series = {name: [] for name in LAYER_NAMES}
    faith_vals: list[float] = []
    pgd_vals: list[float] = []

    for row in rows:
        for i, name in enumerate(LAYER_NAMES):
            key = L0_KEYS[i]
            if key in row and row[key] is not None:
                l0_series[name].append(float(row[key]))
        if FAITH_KEY in row and row[FAITH_KEY] is not None:
            faith_vals.append(float(row[FAITH_KEY]))
        if PGD_KEY in row and row[PGD_KEY] is not None:
            pgd_vals.append(float(row[PGD_KEY]))

    if not faith_vals or not any(l0_series.values()):
        return None

    return {
        "l0_series": l0_series,
        "final_faith": faith_vals[-1] if faith_vals else float("inf"),
        "final_pgd": pgd_vals[-1] if pgd_vals else float("inf"),
    }


def compute_l0_distance(l0_series: dict[str, list[float]]) -> float:
    """Sum of absolute differences between final L0 and ideal targets."""
    total = 0.0
    for name in LAYER_NAMES:
        vals = l0_series[name]
        if vals:
            total += abs(vals[-1] - IDEAL_L0[name])
    return total


def compute_l0_slope(l0_series: dict[str, list[float]]) -> float:
    """Average L0 slope (linear regression on second half). More negative = better."""
    slopes = []
    for name in LAYER_NAMES:
        vals = l0_series[name]
        if len(vals) < 4:
            continue
        half = len(vals) // 2
        second_half = vals[half:]
        x = np.arange(len(second_half), dtype=np.float64)
        y = np.array(second_half, dtype=np.float64)
        if len(x) < 2:
            continue
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return float(np.mean(slopes)) if slopes else 0.0


def extract_sweep_params(run: wandb.apis.public.Run) -> dict[str, str]:
    """Extract swept HP values from run config for display."""
    config = run.config
    params = {}

    lr = config.get("lr_schedule", {}).get("start_val")
    if lr is not None:
        params["lr"] = f"{lr:.0e}"

    for loss_cfg in config.get("loss_metric_configs", []):
        if loss_cfg.get("classname") == "ImportanceMinimalityLoss":
            coeff = loss_cfg.get("coeff")
            if coeff is not None:
                params["imp_coeff"] = f"{coeff:.0e}"
            anneal = loss_cfg.get("p_anneal_end_frac")
            if anneal is not None:
                params["anneal"] = f"{anneal}"
            break

    task_cfg = config.get("task_config", {})
    for key in ["init_computational_ci", "init_indexing_ci",
                "init_computational_components", "init_indexing_components"]:
        val = task_cfg.get(key)
        if val is not None:
            params[key] = val

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=str, help="spd-run run ID (e.g. run_20260330_203919)")
    parser.add_argument("--project", type=str, default="paramluhadiya/spd")
    args = parser.parse_args()

    api = wandb.Api()
    all_runs = api.runs(args.project, per_page=100, order="-created_at")
    runs = [
        r for r in all_runs
        if r.state == "finished" and r.name.startswith("pingpong_probe_64-8-")
    ]
    print(f"Found {len(runs)} finished pingpong_probe runs")

    results: list[dict] = []

    for run in runs:
        metrics = fetch_run_metrics(run)
        if metrics is None:
            continue

        l0_dist = compute_l0_distance(metrics["l0_series"])
        l0_slope = compute_l0_slope(metrics["l0_series"])
        faith = metrics["final_faith"]
        pgd = metrics["final_pgd"]

        # Disqualification
        disqualified = False
        reason = ""
        if faith > MAX_FAITH:
            disqualified = True
            reason = f"faith={faith:.2e}"
        if pgd > MAX_PGD:
            disqualified = True
            reason += f" pgd={pgd:.4f}"

        score = float("inf") if disqualified else l0_dist - SLOPE_WEIGHT * l0_slope

        final_l0 = {}
        for name in LAYER_NAMES:
            vals = metrics["l0_series"][name]
            final_l0[name] = vals[-1] if vals else float("nan")

        results.append({
            "run_id": run.id,
            "run_name": run.name,
            "score": score,
            "l0_dist": l0_dist,
            "l0_slope": l0_slope,
            "faith": faith,
            "pgd": pgd,
            "final_l0": final_l0,
            "disqualified": disqualified,
            "reason": reason,
            "params": extract_sweep_params(run),
        })

    # Sort by score (lower = better)
    results.sort(key=lambda r: r["score"])

    # Print table
    print(f"\n{'Rank':>4s}  {'Run':>12s}  {'Score':>8s}  {'L0 dist':>7s}  {'Slope':>7s}  "
          f"{'Faith':>9s}  {'PGD':>9s}  "
          f"{'L0 m.0':>6s}  {'L0 m.2':>6s}  {'L0 m.4':>6s}  "
          f"{'lr':>6s}  {'imp':>6s}  {'anneal':>6s}  {'Status'}")
    print("-" * 140)

    for rank, r in enumerate(results, 1):
        status = f"DQ ({r['reason']})" if r["disqualified"] else "OK"
        score_str = "inf" if r["score"] == float("inf") else f"{r['score']:.2f}"
        p = r["params"]
        print(f"{rank:>4d}  {r['run_id']:>12s}  {score_str:>8s}  {r['l0_dist']:>7.2f}  "
              f"{r['l0_slope']:>7.4f}  {r['faith']:>9.2e}  {r['pgd']:>9.4f}  "
              f"{r['final_l0']['model.0']:>6.1f}  {r['final_l0']['model.2']:>6.1f}  "
              f"{r['final_l0']['model.4']:>6.1f}  "
              f"{p.get('lr', '?'):>6s}  {p.get('imp_coeff', '?'):>6s}  "
              f"{p.get('anneal', '?'):>6s}  {status}")

    # Print ideal targets for reference
    print(f"\nIdeal L0 targets: model.0={IDEAL_L0['model.0']}, "
          f"model.2={IDEAL_L0['model.2']}, model.4={IDEAL_L0['model.4']}")
    print(f"Disqualification: faith > {MAX_FAITH}, PGD > {MAX_PGD}")

    qualified = [r for r in results if not r["disqualified"]]
    print(f"\nQualified: {len(qualified)}/{len(results)} runs")

    if qualified:
        print("\nTop 5 configs:")
        for r in qualified[:5]:
            p = r["params"]
            print(f"  {r['run_name']}: score={r['score']:.2f}, "
                  f"lr={p.get('lr')}, imp={p.get('imp_coeff')}, anneal={p.get('anneal')}")


if __name__ == "__main__":
    main()
