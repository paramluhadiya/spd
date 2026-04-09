"""Analyze per-circuit component evolution across PingPong training checkpoints.

For each circuit (i,j) and each layer, shows:
- Which computational components are active vs the ground truth expected set
- Indexing component (one_hot_i, one_hot_j) CI values
- How all of these evolve across training steps

Ground truth for circuit (i,j):
  Layer 0 (model.0): block i comp (i*d..i*d+7) + one_hot_i[i] + one_hot_j[j] = 10
  Layer 1 (model.2): block j comp (j*d..j*d+7) + one_hot_i[i] + one_hot_j[j] = 10
  Layer 2 (model.4): block i comp (i*d..i*d+7) + one_hot_i[i] + one_hot_j[j] = 10

Usage:
    python scripts/analyze_component_evolution.py <run_dir> [--steps 10000,50000,100000]
"""

import argparse
from pathlib import Path
from typing import Any

import torch

from spd.models.component_model import ComponentModel, SPDRunInfo

N_COMPUTATIONAL = 64
N_ONE_HOT_I = 8
N_ONE_HOT_J = 8
N_TRUE = N_COMPUTATIONAL + N_ONE_HOT_I + N_ONE_HOT_J  # 80
D = 64
d = 8
NUM_BLOCKS = 8


def load_model_at_step(run_dir: Path, step: int) -> ComponentModel:
    checkpoint_path = run_dir / f"model_{step}.pth"
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    run_info = SPDRunInfo.from_path(str(checkpoint_path))
    return ComponentModel.from_run_info(run_info)


def make_circuit_batch(
    i: int, j: int, n_samples: int, device: str
) -> torch.Tensor:
    input_dim = D + 2 * NUM_BLOCKS
    x = torch.zeros(n_samples, input_dim, device=device)
    x[:, d * i : d * (i + 1)] = torch.randn(n_samples, d, device=device).abs()
    x[:, D + i] = 1.0
    x[:, D + NUM_BLOCKS + j] = 1.0
    return x


def expected_comp_block(layer_idx: int, i: int, j: int) -> range:
    """Return the expected active computational block for this layer and circuit."""
    # Layer 0: i→j, Layer 1: j→i, Layer 2: i→j
    # Active input block: layer 0 & 2 use block i, layer 1 uses block j
    block = i if layer_idx in (0, 2) else j
    return range(block * d, (block + 1) * d)


LAYER_NAMES = ["model.0", "model.2", "model.4"]
LAYER_INDICES = {name: idx for idx, name in enumerate(LAYER_NAMES)}


@torch.no_grad()
def analyze_circuit(
    model: ComponentModel,
    i: int,
    j: int,
    device: str,
    ci_threshold: float,
    n_samples: int = 1024,
) -> dict[str, Any]:
    """Analyze one circuit across all layers."""
    batch = make_circuit_batch(i, j, n_samples, device)
    output = model(batch, cache_type="input")
    ci = model.calc_causal_importances(output.cache, sampling="continuous")

    result: dict[str, Any] = {}
    for name in LAYER_NAMES:
        layer_ci = ci.upper_leaky[name].mean(dim=0).cpu()  # (C,)
        layer_idx = LAYER_INDICES[name]
        expected_block = expected_comp_block(layer_idx, i, j)

        # All active components
        active_mask = layer_ci > ci_threshold
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()
        # Categorize active components
        active_comp = [(idx, layer_ci[idx].item()) for idx in active_indices if idx < N_COMPUTATIONAL]
        active_idx_comps = [(idx, layer_ci[idx].item()) for idx in active_indices
                           if N_COMPUTATIONAL <= idx < N_TRUE]
        active_extra = [(idx, layer_ci[idx].item()) for idx in active_indices if idx >= N_TRUE]

        # Expected components
        expected_comp_indices = list(expected_block)
        expected_idx_i = N_COMPUTATIONAL + i
        expected_idx_j = N_COMPUTATIONAL + N_ONE_HOT_I + j

        # Check which expected comp components are found/missing
        found_expected = [(idx, layer_ci[idx].item()) for idx in expected_comp_indices
                          if layer_ci[idx].item() > ci_threshold]
        missing_expected = [(idx, layer_ci[idx].item()) for idx in expected_comp_indices
                            if layer_ci[idx].item() <= ci_threshold]
        # Unexpected computational components (active but not in expected block)
        unexpected_comp = [(idx, ci_val) for idx, ci_val in active_comp
                           if idx not in expected_comp_indices]

        result[name] = {
            "n_active_total": len(active_indices),
            "n_active_comp": len(active_comp),
            "n_active_indexing": len(active_idx_comps),
            "n_active_extra": len(active_extra),
            # Expected computational: 8 from the correct block
            "expected_found": found_expected,
            "expected_missing": missing_expected,
            "unexpected_comp": unexpected_comp,
            # Indexing CI for the correct i and j
            "one_hot_i_ci": layer_ci[expected_idx_i].item(),
            "one_hot_j_ci": layer_ci[expected_idx_j].item(),
            # All indexing CIs (to see if wrong ones activate)
            "all_ohi_ci": [layer_ci[N_COMPUTATIONAL + k].item() for k in range(N_ONE_HOT_I)],
            "all_ohj_ci": [layer_ci[N_COMPUTATIONAL + N_ONE_HOT_I + k].item()
                           for k in range(N_ONE_HOT_J)],
            # Active indexing detail
            "active_indexing": active_idx_comps,
            "active_extra": active_extra,
        }
    return result


def print_circuit_evolution(
    i: int,
    j: int,
    all_results: dict[int, dict[str, Any]],
) -> None:
    steps = sorted(all_results.keys())

    print(f"\n{'=' * 100}")
    print(f"CIRCUIT ({i},{j})")
    print(f"{'=' * 100}")

    for name in LAYER_NAMES:
        layer_idx = LAYER_INDICES[name]
        expected_block_idx = i if layer_idx in (0, 2) else j
        expected_range = expected_comp_block(layer_idx, i, j)

        print(f"\n  {name} (expected: block {expected_block_idx} = "
              f"c{expected_range.start}..c{expected_range.stop - 1}, "
              f"ohi[{i}]=c{N_COMPUTATIONAL + i}, ohj[{j}]=c{N_COMPUTATIONAL + N_ONE_HOT_I + j})")
        print(f"  {'─' * 90}")

        # Table header
        print(f"  {'Step':>7s} | {'Total':>5s} | {'Comp':>4s} | {'Idx':>3s} | "
              f"{'Found/8':>7s} | {'ohi_ci':>6s} | {'ohj_ci':>6s} | Details")
        print(f"  {'─' * 90}")

        for step in steps:
            r = all_results[step][name]
            n_found = len(r["expected_found"])

            # Build detail string
            details: list[str] = []
            if r["expected_missing"]:
                miss_str = ", ".join(f"c{idx}({ci:.2f})" for idx, ci in r["expected_missing"])
                details.append(f"MISSING: {miss_str}")
            if r["unexpected_comp"]:
                unexp_str = ", ".join(f"c{idx}({ci:.2f})" for idx, ci in r["unexpected_comp"])
                details.append(f"UNEXPECTED: {unexp_str}")
            if r["active_extra"]:
                extra_str = ", ".join(f"c{idx}({ci:.2f})" for idx, ci in r["active_extra"])
                details.append(f"EXTRA>80: {extra_str}")

            # Check for wrong indexing components
            wrong_idx = [(idx, ci) for idx, ci in r["active_indexing"]
                         if idx != N_COMPUTATIONAL + i
                         and idx != N_COMPUTATIONAL + N_ONE_HOT_I + j]
            if wrong_idx:
                wrong_str = ", ".join(f"c{idx}({ci:.2f})" for idx, ci in wrong_idx)
                details.append(f"WRONG_IDX: {wrong_str}")

            detail_str = " | ".join(details) if details else "OK"

            print(f"  {step:>7d} | {r['n_active_total']:>5d} | {r['n_active_comp']:>4d} | "
                  f"{r['n_active_indexing']:>3d} | "
                  f"{n_found:>3d}/8   | {r['one_hot_i_ci']:>6.3f} | {r['one_hot_j_ci']:>6.3f} | "
                  f"{detail_str}")

        # Show CI evolution for the 8 expected computational components
        print(f"\n  CI for expected comp block {expected_block_idx} "
              f"(c{expected_range.start}..c{expected_range.stop - 1}):")
        header = "  " + f"{'Step':>7s} | " + " ".join(
            f"c{k:>2d}" for k in expected_range
        )
        print(header)
        for step in steps:
            r = all_results[step][name]
            found_dict = {idx: ci for idx, ci in r["expected_found"]}
            missing_dict = {idx: ci for idx, ci in r["expected_missing"]}
            vals = []
            for k in expected_range:
                ci_val = found_dict.get(k, missing_dict.get(k, 0.0))
                vals.append(f"{ci_val:>4.2f}")
            print(f"  {step:>7d} | " + " ".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ci_threshold", type=float, default=0.5)
    parser.add_argument("--steps", type=str, default=None)
    parser.add_argument("--circuits", type=str, default=None,
                        help="Comma-separated circuits like '0-0,1-1,2-3' (default: all)")
    parser.add_argument("--n_samples", type=int, default=1024)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    assert run_dir.exists()

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = sorted(
            int(f.stem.split("_")[1]) for f in run_dir.glob("model_*.pth")
        )

    if args.circuits:
        circuits = [tuple(int(x) for x in c.split("-")) for c in args.circuits.split(",")]
    else:
        circuits = [(i, j) for i in range(NUM_BLOCKS) for j in range(NUM_BLOCKS)]

    print(f"Steps: {steps}")
    print(f"Circuits: {len(circuits)}")
    print(f"CI threshold: {args.ci_threshold}")

    for i, j in circuits:
        # Collect results across all steps for this circuit
        circuit_results: dict[int, dict[str, Any]] = {}
        for step in steps:
            model = load_model_at_step(run_dir, step)
            model.to(args.device)
            model.eval()
            circuit_results[step] = analyze_circuit(
                model, i, j, args.device, args.ci_threshold, args.n_samples
            )

        print_circuit_evolution(i, j, circuit_results)


if __name__ == "__main__":
    main()
