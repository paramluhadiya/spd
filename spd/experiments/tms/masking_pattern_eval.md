# MaskingPatternEval

Evaluates whether the SPD decomposition learns components that capture the masking/suppression mechanism of the PingPong model.

## Background

In PingPong, computation bounces between memory blocks i and j. At each layer, a large negative bias suppresses all blocks except the active one:

- **Even layers (0, 2)**: `one_hot_j` controls masking — only block j survives
- **Odd layers (1)**: `one_hot_i` controls masking — only block i survives

The masking is implemented via `W_jc` / `W_ic` matrices that fill all blocks with `-B` then zero out the active block (see `bss_models.py:_build_bias_and_mask_blocks`).

## What This Eval Checks

For each input with indices (i, j), and for each layer:

1. **Find firing components**: Select components with CI > `ci_threshold` for this specific input
2. **Find aligned components**: Among firing components, find those whose V has cosine similarity > `cos_sim_threshold` with the relevant one-hot vector (one_hot_j for even layers, one_hot_i for odd)
3. **Build virtual masking component**: Sum the U vectors of all firing + aligned components
4. **Evaluate the pattern**: Check if the virtual U has the correct structure — the active block should have the highest mean value, all other blocks should be suppressed (large negative)

## Metrics

| Metric | Description |
|--------|-------------|
| `coverage` | Fraction of inputs where at least one firing + aligned component was found |
| `pattern_accuracy` | Of inputs with found components, fraction where argmax of block means is the correct active block |
| `suppression_strength` | Mean magnitude of negative values in suppressed blocks (higher = stronger masking) |
| `margin` | Mean(active block) - Mean(suppressed blocks) (higher = cleaner separation) |

All metrics are reported per-layer (`*_model.0`, `*_model.2`, `*_model.4`) and averaged across the model (`*_model`).

## Config

```yaml
eval_metric_configs:
  - classname: "MaskingPatternEval"
    D: 64                    # Network width
    d: 8                     # Block width
    cos_sim_threshold: 0.9   # Min cosine similarity for V alignment
    ci_threshold: 0.1        # Min CI for component to be considered firing
```

## Interpretation

- **coverage ~ 1.0**: The decomposition reliably finds masking-like components for each input
- **pattern_accuracy ~ 1.0**: Those components have the correct suppression pattern
- **suppression_strength >> 0**: The bias magnitudes are large enough to kill pre-activations
- **margin >> 0**: Clean separation between active and suppressed blocks

Low coverage means the decomposition hasn't learned distinct masking components. Low pattern_accuracy means components fire but produce wrong suppression patterns.

## Files

- Metric implementation: `spd/metrics/masking_pattern_eval.py`
- Config class: `MaskingPatternEvalConfig` in `spd/configs.py`
- Eval registration: `spd/eval.py`
