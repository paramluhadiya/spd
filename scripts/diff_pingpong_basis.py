"""Test the rotated-basis hypothesis for PingPong SPD components (model.0 only).

Checks:
  (1) V block-sparsity on the input side.
  (2) Layer faithfulness: V @ U ≈ W_T.
  (3) Per-block reconstruction with V-norm and CI filtering:
      For each input block b, select components that (a) have V mass in block b
      AND (b) have high CI on inputs where block b is active. Check if the selected
      V_sub @ U_sub ≈ W_T[b_rows, :].

For model.0, the pre_weight_act is just the raw input x, so CI computation is
straightforward: generate PingPong inputs with block b active, feed to CI MLP.

Usage:
    python scripts/diff_pingpong_basis.py --run wandb:paramluhadiya/spd/s-64ee330f
"""

import argparse
from pathlib import Path

import torch

from spd.experiments.tms.pingpong_percircuit_ideal_init_decomposition import (
    D,
    NUM_BLOCKS,
    d,
)
from spd.models.component_model import ComponentModel
from spd.settings import SPD_OUT_DIR

LAYER_NAME = "model.0"
N_SAMPLES_PER_BLOCK = 256


def generate_block_inputs(
    block_b: int,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate PingPong inputs where source block i = block_b.

    For model.0 (even layer), the source block is i. We cycle j through all 8 blocks,
    with random values in block i.

    Returns: (n_samples, 80) tensor.
    """
    input_dim = D + 2 * NUM_BLOCKS  # 80
    x = torch.zeros(n_samples, input_dim, device=device)

    # Random values in block i = block_b
    values = torch.rand(n_samples, d, device=device)
    x[:, block_b * d : (block_b + 1) * d] = values

    # One-hot i = block_b for all samples
    x[:, D + block_b] = 1.0

    # Cycle j through all blocks
    for s in range(n_samples):
        j = s % NUM_BLOCKS
        x[s, D + NUM_BLOCKS + j] = 1.0

    return x


def compute_ci_for_inputs(
    model: ComponentModel,
    layer_name: str,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute CI values for given inputs at a specific layer.

    For model.0 (VectorMLPCiFn), pre_weight_act = x itself.
    Returns: (n_samples, C) tensor of CI values (post-sigmoid, lower-leaky).
    """
    with torch.no_grad():
        ci_outputs = model.calc_causal_importances(
            pre_weight_acts={layer_name: x},
            sampling="continuous",
        )
    return ci_outputs.lower_leaky[layer_name]  # (n_samples, C)


def analyse_model0(model: ComponentModel) -> str:
    components = model.components[LAYER_NAME]
    V = components.V.detach().cpu().float()  # (80, C)
    U = components.U.detach().cpu().float()  # (C, 80)
    W_T = model.target_weight(LAYER_NAME).detach().cpu().float().T  # (80, 80)
    C = V.shape[1]

    device = next(model.parameters()).device
    lines: list[str] = [f"--- {LAYER_NAME} (C={C}) ---"]

    # Layer faithfulness
    Weff = V @ U
    frob_err = (Weff - W_T).norm().item() / W_T.norm().item()
    lines.append(f"  layer faithfulness ||V@U - W_T|| / ||W_T|| = {frob_err:.4e}")

    # V block mass analysis
    V_comp = V[:D, :]  # (64, C)
    V_blocks = V_comp.reshape(NUM_BLOCKS, d, C)
    block_mass = (V_blocks * V_blocks).sum(dim=1)  # (8, C)
    total_comp_mass = block_mass.sum(dim=0)
    idx_mass = (V[D:, :] ** 2).sum(dim=0)
    total_mass = total_comp_mass + idx_mass

    is_comp = total_comp_mass > idx_mass
    V_block_conc = block_mass.max(dim=0).values / total_comp_mass.clamp_min(1e-30)
    best_block = block_mass.argmax(dim=0)
    v_norm = total_mass.sqrt()  # (C,)

    bc_comp = V_block_conc[is_comp]
    lines.append(f"  #computational: {int(is_comp.sum().item())}")
    lines.append("  V block_concentration (computational):")
    lines.append(f"    mean={bc_comp.mean().item():.4f}  "
                 f"#>0.99={int((bc_comp > 0.99).sum().item())}  "
                 f"#>0.9={int((bc_comp > 0.9).sum().item())}  "
                 f"#>0.8={int((bc_comp > 0.8).sum().item())}  "
                 f"#>0.5={int((bc_comp > 0.5).sum().item())}")

    # Compute CI for each input block
    lines.append("")
    lines.append("  Computing CI on synthetic inputs per block...")
    # ci_per_block[b] = mean CI across samples where block b is active, shape (C,)
    ci_per_block: dict[int, torch.Tensor] = {}
    for b in range(NUM_BLOCKS):
        x_b = generate_block_inputs(b, N_SAMPLES_PER_BLOCK, device)
        ci_vals = compute_ci_for_inputs(model, LAYER_NAME, x_b)  # (N, C)
        ci_per_block[b] = ci_vals.mean(dim=0).cpu()  # (C,)

    # Per-block reconstruction with different filtering strategies
    lines.append("")
    lines.append("  === Per-block reconstruction: V-argmax only (no filtering) ===")
    lines.append(f"    {'b':>2s}  {'n':>5s}  {'rank':>4s}  {'recon_err':>12s}")
    for b in range(NUM_BLOCKS):
        mask = is_comp & (best_block == b)
        n_b = int(mask.sum().item())
        V_sub = V[b * d : (b + 1) * d, mask]
        U_sub = U[mask, :]
        target = W_T[b * d : (b + 1) * d, :]
        recon = V_sub @ U_sub
        err = (recon - target).norm().item() / target.norm().item()
        rank = int(torch.linalg.matrix_rank(V_sub).item())
        lines.append(f"    {b:>2d}  {n_b:>5d}  {rank:>4d}  {err:>12.4e}")

    # Filter by V-norm: keep top-k by V-norm per block
    lines.append("")
    lines.append("  === Per-block reconstruction: V-argmax + top-64 by V-norm ===")
    lines.append(f"    {'b':>2s}  {'n':>5s}  {'rank':>4s}  {'recon_err':>12s}")
    for b in range(NUM_BLOCKS):
        mask = is_comp & (best_block == b)
        idxs = mask.nonzero(as_tuple=True)[0]
        norms = v_norm[idxs]
        top_k = min(64, len(idxs))
        top_idxs = idxs[norms.argsort(descending=True)[:top_k]]
        top_mask = torch.zeros(C, dtype=torch.bool)
        top_mask[top_idxs] = True
        V_sub = V[b * d : (b + 1) * d, top_mask]
        U_sub = U[top_mask, :]
        target = W_T[b * d : (b + 1) * d, :]
        recon = V_sub @ U_sub
        err = (recon - target).norm().item() / target.norm().item()
        rank = int(torch.linalg.matrix_rank(V_sub).item())
        lines.append(f"    {b:>2d}  {top_k:>5d}  {rank:>4d}  {err:>12.4e}")

    # Filter by CI: keep components with mean CI > threshold for this block
    for ci_thresh in [0.5, 0.3, 0.1]:
        lines.append("")
        lines.append(f"  === Per-block reconstruction: CI > {ci_thresh} for block inputs ===")
        lines.append(f"    {'b':>2s}  {'n':>5s}  {'rank':>4s}  {'recon_err':>12s}")
        for b in range(NUM_BLOCKS):
            ci_b = ci_per_block[b]  # (C,)
            mask = is_comp & (ci_b > ci_thresh)
            n_b = int(mask.sum().item())
            if n_b == 0:
                lines.append(f"    {b:>2d}  {n_b:>5d}  {'-':>4s}  {'(empty)':>12s}")
                continue
            V_sub = V[b * d : (b + 1) * d, mask]
            U_sub = U[mask, :]
            target = W_T[b * d : (b + 1) * d, :]
            recon = V_sub @ U_sub
            err = (recon - target).norm().item() / target.norm().item()
            rank = int(torch.linalg.matrix_rank(V_sub).item())
            lines.append(f"    {b:>2d}  {n_b:>5d}  {rank:>4d}  {err:>12.4e}")

    # Filter by BOTH: V-argmax to block AND CI > threshold
    for ci_thresh in [0.5, 0.3, 0.1]:
        lines.append("")
        lines.append(f"  === Per-block: V-argmax to block AND CI > {ci_thresh} ===")
        lines.append(f"    {'b':>2s}  {'n':>5s}  {'rank':>4s}  {'recon_err':>12s}")
        for b in range(NUM_BLOCKS):
            ci_b = ci_per_block[b]
            mask = is_comp & (best_block == b) & (ci_b > ci_thresh)
            n_b = int(mask.sum().item())
            if n_b == 0:
                lines.append(f"    {b:>2d}  {n_b:>5d}  {'-':>4s}  {'(empty)':>12s}")
                continue
            V_sub = V[b * d : (b + 1) * d, mask]
            U_sub = U[mask, :]
            target = W_T[b * d : (b + 1) * d, :]
            recon = V_sub @ U_sub
            err = (recon - target).norm().item() / target.norm().item()
            rank = int(torch.linalg.matrix_rank(V_sub).item())
            lines.append(f"    {b:>2d}  {n_b:>5d}  {rank:>4d}  {err:>12.4e}")

    # CI-only (no V-argmax constraint): which comps fire for block b?
    lines.append("")
    lines.append("  === CI distribution: mean CI per component for each block ===")
    lines.append(f"    {'b':>2s}  {'#CI>0.9':>8s}  {'#CI>0.5':>8s}  {'#CI>0.3':>8s}  {'#CI>0.1':>8s}")
    for b in range(NUM_BLOCKS):
        ci_b = ci_per_block[b]
        lines.append(
            f"    {b:>2d}  "
            f"{int((ci_b > 0.9).sum().item()):>8d}  "
            f"{int((ci_b > 0.5).sum().item()):>8d}  "
            f"{int((ci_b > 0.3).sum().item()):>8d}  "
            f"{int((ci_b > 0.1).sum().item()):>8d}"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default="wandb:paramluhadiya/spd/s-64ee330f")
    args = parser.parse_args()

    run_path = args.run
    if not run_path.startswith("wandb:") and not Path(run_path).exists():
        run_path = "wandb:" + run_path

    run_id = run_path.rstrip("/").rsplit("/", 1)[-1]
    out_dir = SPD_OUT_DIR / "component_diff" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {run_path} ...")
    model = ComponentModel.from_pretrained(run_path)
    model.eval()

    header = f"V-basis rotation analysis for {run_path}\n{'=' * 72}\n"
    report = header + analyse_model0(model)
    print(report)
    (out_dir / "basis_rotation.txt").write_text(report)
    print(f"Saved to {out_dir / 'basis_rotation.txt'}")


if __name__ == "__main__":
    main()
