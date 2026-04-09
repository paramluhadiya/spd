# PingPong 64-8 Hyperparameter Tuning Strategy

## Context

We have verified an ideal decomposition for PingPong 64-8 (D=64, d=8, T=64 circuits, 3 layers).
The ideal solution was confirmed stable by initializing from it and running SPD — the decomposition
stays at the ideal (WandB: `paramluhadiya/spd/s-1c8b8e5d`).

**The problem:** using the ideal run's config from random init does not converge. Runs are too slow
and fail to fully decompose. We need to find hyperparameters that reach the ideal from random init.

## Ideal Solution Targets

From the stable ideal-init run (`paramluhadiya/spd/s-1c8b8e5d`):

| Metric | Ideal Value | Notes |
|--------|-------------|-------|
| `eval/l0/0.1_model.0` | ~9.5 | **Primary signal** |
| `eval/l0/0.1_model.2` | ~6.0 | **Primary signal** |
| `eval/l0/0.1_model.4` | ~5.8 | **Primary signal** |
| `eval/loss/FaithfulnessLoss` | ~9e-9 | Should be near zero early |
| `eval/loss/PGDReconLoss` | ~7e-7 | Not useful as early signal |
| `eval/loss/PGDReconSubsetLoss` | ~5e-7 | Not useful as early signal |
| `eval/loss/ImportanceMinimalityLoss` | ~66 | Reflects sparsity state |

### Ideal run config (key fields)

- `lr_schedule.start_val`: 1e-5 (too low for convergence from random init)
- `loss.Faith.coeff`: 5
- `loss.ImpMin.coeff`: 5e-5, beta=0.16, pnorm=2, p_anneal_end_frac=1, p_anneal_final_p=0.4
- `loss.PGDRecon.coeff`: 25, n_steps=4, step_size=0.5
- `loss.PGDReconSub.coeff`: 25, n_steps=4, step_size=0.5
- `steps`: 100000
- `batch_size`: 4096

## Strategy: Short Probe + Score + Extend

Full runs are O(100k) steps, making large grid searches prohibitively expensive. Instead:

### Phase 1: Short probe runs (~20-30k steps)

Run a focused sweep varying parameters that control convergence dynamics. Keep runs short
to quickly evaluate many configs.

### Phase 2: Score by L0 reduction

**L0 is the primary early signal.** PGD recon loss is misleading early on — most runs achieve
similar PGD values initially regardless of whether they eventually converge. The differentiator
is whether a config can quickly cull unimportant components, driving L0 down toward the ~10
ballpark.

Scoring criteria (in priority order):
1. **L0 trajectory slope** — is L0 decreasing? Faster decrease = better config.
2. **L0 proximity to target** — how close are the L0 values to [9.5, 6.0, 5.8] at the end of the probe?
3. **Faithfulness sanity check** — faithfulness should be near zero; if not, the config is broken.

### Phase 3: Extend top candidates

Take the top 3-5 configs from Phase 2 and run them to 200k-1M steps to verify actual convergence.

## Sweep Parameters

The sweep should focus on parameters that control how aggressively the optimization culls components.

### Primary axes

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `lr_schedule.start_val` | [5e-5, 1e-4, 3e-4, 5e-4] | Most impactful for convergence speed. Ideal run used 1e-5 (too low from random init). |
| `loss.ImpMin.coeff` | [5e-5, 1e-4, 2e-4] | Controls sparsity pressure. Higher = more aggressive culling. |
| `p_anneal_end_frac` | [0.3, 0.5, 1.0] | When sparsity fully kicks in. Lower = earlier full pressure. |

**Grid size:** 4 x 3 x 3 = 36 configs. Fits in a sweep with 8 GPU agents.

### Fixed parameters (from current config)

These should stay fixed during the probe sweep:

```yaml
loss.Faith.coeff: 5.0
loss.PGDRecon.coeff: 25
loss.PGDReconSub.coeff: 25
loss.ImpMin.beta: 0.16
loss.ImpMin.pnorm: 2.0
loss.ImpMin.p_anneal_start_frac: 0.0
loss.ImpMin.p_anneal_final_p: 0.4
batch_size: 4096
n_mask_samples: 1
```

### Secondary axes (for Phase 3 refinement if needed)

- `loss.PGDRecon.coeff` / `loss.PGDReconSub.coeff` ratio
- `loss.ImpMin.beta` — controls binary vs continuous sparsity pressure
- `faithfulness_warmup_steps` — longer warmup before sparsity pressure

## Scoring Script

After probe runs complete, pull metrics from WandB and rank configs:

```python
# Pseudocode for scoring
ideal_l0 = {"model.0": 9.5, "model.2": 6.0, "model.4": 5.8}

for run in sweep_runs:
    # 1. L0 proximity score (lower = better)
    final_l0 = get_final_l0(run)
    l0_distance = sum(abs(final_l0[k] - ideal_l0[k]) for k in ideal_l0)

    # 2. L0 trajectory slope (more negative = better)
    l0_history = get_l0_history(run)
    l0_slope = compute_slope(l0_history)  # linear regression on last half

    # 3. Faithfulness sanity check
    faith = get_final_metric(run, "eval/loss/FaithfulnessLoss")
    if faith > 1e-3:
        score = float('inf')  # disqualify
    else:
        score = l0_distance - weight * l0_slope  # combine proximity and trajectory

    run.score = score
```

## Implementation Checklist

- [ ] Create `sweep_params.yaml` for Phase 1 probe sweep (36 configs, 20-30k steps)
- [ ] Update pingpong config for probe run length (~25k steps)
- [ ] Run Phase 1 sweep: `spd-run --experiments pingpong --sweep <params.yaml> --n_agents 8`
- [ ] Write scoring script to rank probe runs by L0 reduction
- [ ] Run scoring script, identify top 3-5 configs
- [ ] Create configs for Phase 3 long runs (200k-1M steps) with top configs
- [ ] Run Phase 3 and evaluate convergence to ideal

## Notes

- Max 8 GPUs at a time (cluster policy)
- The ideal L0 values (~10 per layer) tell us the target is sparse — most of the 600 components
  per layer should be culled. Configs that can't start culling within 20-30k steps are unlikely
  to converge even given more time.
- PGDReconLayerwiseLoss was tested but commented out in the current config. Could be revisited
  as a secondary axis if initial sweep doesn't find good configs.
