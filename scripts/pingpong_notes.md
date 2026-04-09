# PingPong Model & SPD Decomposition Notes

## The PingPong Model (Alex Gibson, Dec 2025)

**Source**: "Ping pong computation in superposition" — LessWrong post by Alex Gibson. PDF at `spd/experiments/tms/Ping pong computation in superposition — LessWrong.pdf`.

### Core Idea

A construction that embeds T = (D/d)² circuits into a single ReLU network of width D, with **zero error** when exactly one circuit is active per forward pass (z=1).

### How It Works

1. Divide width D into D/d contiguous **memory blocks** of size d
2. Each circuit is identified by an ordered pair of blocks **(i, j)**
3. Input: vector x ∈ ℝ^d loaded into block i, plus one-hot encodings of i and j
4. Computation **ping-pongs** between blocks i and j across layers:
   - Layer 0: i → j (read from block i, write to block j)
   - Layer 1: j → i (read from block j, write to block i)
   - Layer 2: i → j (read from block i, write to block j)
5. At each layer, a **massive negative bias (-B)** from the routing one-hot suppresses all blocks except the destination block — only the target block survives ReLU
6. The source one-hot provides per-circuit **biases** to the destination block
7. Each transition freely specifies a d×d weight matrix and d-dim bias — effectively a width-d MLP layer

### Implementation (D=64, d=8)

- **T = 64 circuits** (8 blocks × 8 blocks)
- **3 layers** (n_layers=3), input dim = 80 (64 computing + 8 one_hot_i + 8 one_hot_j)
- Weight matrices are (80×80) with block structure:

```
[c→c    i→c    j→c ]    c: computing block (64 dims)
[c→i    i→i    j→i ]    i: one_hot_i block (8 dims)
[c→j    i→j    j→j ]    j: one_hot_j block (8 dims)
```

Where:
- **c→c** (64×64): embedded circuit weights (all T circuits superimposed)
- **i→c / j→c**: bias injection + suppression masks
- **i→i / j→j**: identity (preserve one-hots through layers)
- All other blocks: zeros

### Suppression Mechanism

- Even layers (0, 2): `one_hot_j` controls masking — fills all blocks with -B, zeros out block j
- Odd layers (1): `one_hot_i` controls masking — fills all blocks with -B, zeros out block i
- B is large enough that non-target blocks are zeroed after ReLU

### Key Code: `spd/experiments/tms/bss_models.py`

- `BSSModel`: Original single-layer BSS construction
- `PingPongModel`: Multi-layer ping-pong construction (lines 428-697)
  - `_build_cc_block()`: Embeds all circuit weights into the c→c block
  - `_build_bias_and_mask_blocks()`: Builds suppression and bias blocks
  - `verify_circuit()` / `verify_all_circuits()`: Validates network output matches direct computation

---

## SPD Decomposition of PingPong

Three decomposition variants exist, with increasing sophistication:

### 1. Standard Decomposition (`pingpong_decomposition.py`)

- **Random initialization**, C=1200 components per layer
- No ground-truth knowledge used
- Config: `pingpong_64-8_config.yaml`

### 2. Ideal Init — 80 Components (`pingpong_ideal_init_decomposition.py`)

- **80 true components** per layer: 64 computational (one per neuron) + 16 indexing (8 ohi + 8 ohj)
- V[:,k] = e_k (standard basis), U[k,:] = W^T[k,:] (column of target weight)
- CI uses **per-input-dimension** detection: GELU finite-difference trick
  - `CI_k(x) ≈ 1 when x_k > 0, ≈ 0 when x_k = 0`
- Remaining C-80 components initialized small with CI biased off
- Config: `pingpong_ideal_init_64-8_config.yaml`

### 3. Per-Circuit Ideal Init — 528 Components (`pingpong_percircuit_ideal_init_decomposition.py`)

**Most sophisticated variant.** This is the current focus of experiments.

- **528 true components**: 512 computational + 16 indexing
- **512 computational**: one per (src_block, route_block, neuron) triplet
  - V picks up a single neuron's input dim
  - U maps only to the route block's output dims (sparse!)
  - Index: `src * (NUM_BLOCKS * d) + route * d + neuron`
- **16 indexing**: 8 one_hot_i + 8 one_hot_j (same as variant 2)
- **CI uses AND logic**: `CI_k = AND(block_src active, routing_onehot = 1)`
  - Implemented via 3-layer MLP (2 hidden + output) with GELU finite-difference
  - Layer 0: detect block sums + ohi/ohj from input (H1=48 ideal dims)
  - Layer 1: compute AND(block_src, routing) for 64 pairs + pass-through indexing (H2=160 ideal dims)
  - Layer 2: map AND results to 528 component CIs
- **Routing varies by layer**:
  - Layers 0, 2 (even): src=i, route by ohj[j]
  - Layer 1 (odd): src=j, route by ohi[i]
- Config: `pingpong_percircuit_ideal_init_64-8_config.yaml`
  - C=600 per layer (528 true + 72 extra)
  - ci_fn_hidden_dims: [256, 256]
  - beta=0.16, lr=1e-4, 100k steps
  - Loss: faithfulness(5.0) + importance_minimality(1e-4) + PGDReconSubset(25) + PGDRecon(25)

### Ground Truth: What Should Be Active Per Circuit

For circuit (i, j):
- **Layer 0** (model.0): 8 comp from block i + ohi[i] + ohj[j] = 10 components
- **Layer 1** (model.2): 8 comp from block j + ohi[i] + ohj[j] = 10 components
- **Layer 2** (model.4): 8 comp from block i + ohi[i] + ohj[j] = 10 components

In the 528-component decomposition:
- Layer 0: 8 components with src=i, route=j + ohi[i] + ohj[j]
- Layer 1: 8 components with src=j, route=i + ohi[i] + ohj[j]
- Layer 2: 8 components with src=i, route=j + ohi[i] + ohj[j]

---

## Eval Scripts

All in `scripts/`. These evaluate trained decompositions.

### `eval_masking_pattern.py`
- Runs `MaskingPatternEval` metric on a pretrained model
- Checks if learned components capture the suppression mechanism
- Metrics: coverage, pattern_accuracy, suppression_strength, margin
- See `spd/experiments/tms/masking_pattern_eval.md` for full docs

### `analyze_component_evolution.py`
- Tracks component evolution across training checkpoints
- For each circuit (i,j) and layer, shows which components are active vs ground truth
- Reports: found/missing expected components, unexpected activations, indexing CI values
- Usage: `python scripts/analyze_component_evolution.py <run_dir> [--steps ...] [--circuits ...]`

### `circuit_weight_analysis.py`
- Analyzes effective weight matrix W_eff = V @ diag(CI) @ U vs target W
- Per-column error breakdown for circuit-relevant input dimensions
- Shows CI values for ground-truth components and all active components
- Usage: `python scripts/circuit_weight_analysis.py <checkpoint> --circuit i,j --layer model.0`

### `test_bias_dominance_hypothesis.py`
- Investigates per-sample CI behavior for specific circuits
- **Q1**: Which components are active per sample? Activation frequency analysis
- **Q2**: Hidden activation reconstruction MSE on active subspace (destination block)
- **Q3**: Is there a fixed set of computational components generally active?
- Tests whether decomposition learns per-neuron components vs SVD-like directions

### `pingpong_ci_per_circuit.py`
- Histogram of CI activation counts per circuit type (i,j)
- Per-layer + total bar plots with error bars
- Usage: `python scripts/pingpong_ci_per_circuit.py <model_path> [--threshold 0.9]`

### `track_v_distribution.py`
- Tracks how V vectors evolve from one-hot (neuron basis) to distributed directions
- Measures L0.1 (number of active dimensions) and singular value spectrum
- Subspace alignment: do learned V vectors match target SVD right singular vectors?
- Hardcoded to a specific wandb run's checkpoints

### `view_active_components.py`
- Shows active components for specific circuits with detailed V/U vector breakdowns
- Displays per-block norms and means to understand component structure
- Compares target, decomposed, and CI-masked post-layer-0 activations

### `plot_component_activations.py`
- Scatter plots of component activations for high-CI datapoints
- Uses harvest data (requires `spd-harvest` to have been run)
- Orders by median activation or firing frequency
- Usage: `python scripts/plot_component_activations.py <run_id> [--ci-threshold 0.1]`

---

## Key Config Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| D | 64 | Network width |
| d | 8 | Block size |
| num_blocks | 8 | D/d memory blocks |
| T | 64 | Number of circuits (num_blocks²) |
| n_layers | 3 | Ping-pong layers |
| B | large | Suppression bias strength |
| C | 600 | Components per layer (528 true + 72 extra) |
| ci_fn_hidden_dims | [256, 256] | CI MLP hidden layer sizes |
| beta | 0.16 | Importance minimality temperature |
| lr | 1e-4 | Learning rate |
| steps | 100k | Training steps |

## Recent Experiment History

- Latest commit (847e529f, Mar 16): "ideal init pingpong with 528 circuits. Now with nonzero beta, balanced, lower learning rate"
- Previous work explored 80-component decomposition, then moved to 528 per-circuit
- Multiple eval scripts written to diagnose decomposition quality
- Key questions being investigated:
  - Do computational components learn per-neuron or SVD-like directions?
  - Does the CI function correctly implement AND logic after training?
  - How does the V distribution evolve from initialization?

## WandB Paths Referenced

- `wandb:paramluhadiya/spd/runs/s-898e5c08` — used in eval_masking_pattern.py and view_active_components.py
- Pretrained target model: `/home/pluhadiya/spd_out/train/t-a4d0f086/pingpong.pth`
