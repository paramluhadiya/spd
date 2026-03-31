"""Score PingPong sweep runs by L0 proximity to ideal targets.

Uses run.summary for instant metric access (no history download).

Usage:
    python scripts/score_pingpong_sweep.py
    python scripts/score_pingpong_sweep.py --project paramluhadiya/spd
    python scripts/score_pingpong_sweep.py --name-prefix "pingpong_probe_64-8-"
"""

import argparse

import wandb

IDEAL_L0 = {"model.0": 9.5, "model.2": 6.0, "model.4": 5.8}
L0_KEYS = {f"eval/l0/0.1_model.{i}": f"model.{i}" for i in [0, 2, 4]}
FAITH_KEY = "eval/loss/FaithfulnessLoss"
PGD_KEY = "eval/loss/PGDReconLoss"

MAX_FAITH = 1e-3
MAX_PGD = 0.001


def extract_sweep_params(run: wandb.apis.public.Run) -> dict[str, str]:
    config = run.config
    params: dict[str, str] = {}

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
    comp_init = task_cfg.get("init_computational_components", "?")
    ci_init = task_cfg.get("init_computational_ci", "?")
    params["comp"] = comp_init
    params["ci"] = ci_init

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=str, default="paramluhadiya/spd")
    parser.add_argument("--name-prefix", type=str, default="pingpong_probe_64-8-")
    args = parser.parse_args()

    api = wandb.Api()
    print("Fetching runs...")
    all_runs = api.runs(args.project, per_page=100, order="-created_at")

    results: list[dict] = []

    for run in all_runs:
        if run.state != "finished" or not run.name.startswith(args.name_prefix):
            continue

        summary = run.summary

        # Extract final L0 values from summary
        final_l0: dict[str, float] = {}
        for wb_key, layer_name in L0_KEYS.items():
            val = summary.get(wb_key)
            if val is not None:
                final_l0[layer_name] = float(val)

        if len(final_l0) < 3:
            continue

        faith = float(summary.get(FAITH_KEY, float("inf")))
        pgd = float(summary.get(PGD_KEY, float("inf")))

        # L0 distance from ideal
        l0_dist = sum(abs(final_l0[name] - IDEAL_L0[name]) for name in IDEAL_L0)

        # Disqualification
        disqualified = False
        reason = ""
        if faith > MAX_FAITH:
            disqualified = True
            reason = f"faith={faith:.2e}"
        if pgd > MAX_PGD:
            disqualified = True
            reason += f" pgd={pgd:.4f}"

        score = float("inf") if disqualified else l0_dist

        results.append({
            "run_id": run.id,
            "run_name": run.name,
            "score": score,
            "l0_dist": l0_dist,
            "faith": faith,
            "pgd": pgd,
            "final_l0": final_l0,
            "disqualified": disqualified,
            "reason": reason,
            "params": extract_sweep_params(run),
        })

    print(f"Found {len(results)} finished runs matching '{args.name_prefix}'")

    results.sort(key=lambda r: r["score"])

    print(f"\n{'Rank':>4s}  {'Run':>12s}  {'Score':>8s}  "
          f"{'Faith':>9s}  {'PGD':>9s}  "
          f"{'L0 m.0':>6s}  {'L0 m.2':>6s}  {'L0 m.4':>6s}  "
          f"{'lr':>6s}  {'imp':>6s}  {'anneal':>6s}  "
          f"{'comp':>6s}  {'ci':>6s}  {'Status'}")
    print("-" * 140)

    for rank, r in enumerate(results, 1):
        status = f"DQ ({r['reason']})" if r["disqualified"] else "OK"
        score_str = "inf" if r["score"] == float("inf") else f"{r['score']:.2f}"
        p = r["params"]
        print(f"{rank:>4d}  {r['run_id']:>12s}  {score_str:>8s}  "
              f"{r['faith']:>9.2e}  {r['pgd']:>9.4f}  "
              f"{r['final_l0']['model.0']:>6.1f}  {r['final_l0']['model.2']:>6.1f}  "
              f"{r['final_l0']['model.4']:>6.1f}  "
              f"{p.get('lr', '?'):>6s}  {p.get('imp_coeff', '?'):>6s}  "
              f"{p.get('anneal', '?'):>6s}  "
              f"{p.get('comp', '?'):>6s}  {p.get('ci', '?'):>6s}  {status}")

    print(f"\nIdeal L0 targets: model.0={IDEAL_L0['model.0']}, "
          f"model.2={IDEAL_L0['model.2']}, model.4={IDEAL_L0['model.4']}")

    qualified = [r for r in results if not r["disqualified"]]
    print(f"Qualified: {len(qualified)}/{len(results)} runs")

    if qualified:
        print("\nTop 5 configs:")
        for r in qualified[:5]:
            p = r["params"]
            print(f"  {r['run_name']}: score={r['score']:.2f}, "
                  f"lr={p.get('lr')}, imp={p.get('imp_coeff')}, anneal={p.get('anneal')}, "
                  f"comp={p.get('comp')}, ci={p.get('ci')}")


if __name__ == "__main__":
    main()
