"""Investigate per-sample CI behavior for PingPong circuits.

Three questions:
1. For a given circuit, which components are active (CI > threshold) per sample?
   Are computational components 0-63 ever active? Which ones?

2. Hidden activation reconstruction: measure MSE only on the ACTIVE subspace
   (destination block dims, post-ReLU) rather than all 80 dims.
   If this is high, the model gave up on computational components.
   If low, something interesting is happening.

3. Is there a small fixed set of computational components that are "generally" active
   across all samples of a circuit? This would suggest the decomposition learned
   dominant eigenvalue directions rather than per-neuron components.

Component index convention:
  0-63:  computational (8 per block, block k = components k*8..(k+1)*8-1)
  64-71: one_hot_i (component 64+k = one_hot_i[k])
  72-79: one_hot_j (component 72+k = one_hot_j[k])
  80+:   extra/unused
"""

import torch
import torch.nn.functional as F

from spd.models.component_model import ComponentModel, SPDRunInfo

D = 64
d = 8
NUM_BLOCKS = 8
N_COMPUTATIONAL = 64
N_ONE_HOT_I = 8
N_ONE_HOT_J = 8
N_TRUE = N_COMPUTATIONAL + N_ONE_HOT_I + N_ONE_HOT_J  # 80
INPUT_DIM = D + 2 * NUM_BLOCKS  # 80

C_START, C_END = 0, D
I_START, I_END = D, D + NUM_BLOCKS
J_START, J_END = D + NUM_BLOCKS, D + 2 * NUM_BLOCKS

CI_THRESHOLD = 0.1


def mse(a: torch.Tensor) -> float:
    return (a**2).mean().item()


def per_sample_effective_output(
    h: torch.Tensor,
    ci_vals: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample CI-weighted output: sum_k ci_nk * (h_n · V[:,k]) * U[k,:].

    Args:
        h: (n_samples, d_in)
        ci_vals: (n_samples, C)
        V: (d_in, C)
        U: (C, d_out)

    Returns:
        (n_samples, d_out)
    """
    projections = h @ V  # (n_samples, C)
    weighted = projections * ci_vals  # (n_samples, C)
    return weighted @ U  # (n_samples, d_out)


def component_label(k: int) -> str:
    if k < N_COMPUTATIONAL:
        block = k // d
        neuron = k % d
        return f"comp[block{block},n{neuron}]"
    elif k < N_COMPUTATIONAL + N_ONE_HOT_I:
        return f"ohi[{k - N_COMPUTATIONAL}]"
    elif k < N_TRUE:
        return f"ohj[{k - N_COMPUTATIONAL - N_ONE_HOT_I}]"
    else:
        return f"extra[{k}]"


@torch.no_grad()
def main() -> None:
    run_info = SPDRunInfo.from_path("wandb/s-e2e8fb27/files/model_100000.pth")
    model = ComponentModel.from_run_info(run_info)
    model.eval()

    target_model = model.target_model
    n_samples = 4096

    circuits = [(0, 0), (0, 3), (2, 5), (7, 1)]
    layer_names = ["model.0", "model.2", "model.4"]

    for ci_idx, cj_idx in circuits:
        print(f"\n{'='*70}")
        print(f"Circuit ({ci_idx}, {cj_idx})")
        print(f"{'='*70}")

        # Build full input
        x_full = torch.zeros(n_samples, INPUT_DIM)
        x_full[:, d * ci_idx : d * (ci_idx + 1)] = torch.randn(n_samples, d).abs()
        x_full[:, D + ci_idx] = 1.0
        x_full[:, D + NUM_BLOCKS + cj_idx] = 1.0

        # Compute per-sample CI
        output = model(x_full, cache_type="input")
        ci = model.calc_causal_importances(output.cache, sampling="continuous")

        # Collect layer data
        W_targets: dict[str, torch.Tensor] = {}
        Vs: dict[str, torch.Tensor] = {}
        Us: dict[str, torch.Tensor] = {}
        ci_per_sample: dict[str, torch.Tensor] = {}

        for layer_name in layer_names:
            layer = dict(target_model.named_modules())[layer_name]
            W_targets[layer_name] = layer.weight.detach()
            comp = model.components[layer_name]
            Vs[layer_name] = comp.V.detach()
            Us[layer_name] = comp.U.detach()
            ci_per_sample[layer_name] = ci.lower_leaky[layer_name]  # (n_samples, C)

        # Run layer by layer
        h_target = x_full.clone()
        h_eff = x_full.clone()

        for layer_idx, layer_name in enumerate(layer_names):
            W_t = W_targets[layer_name]
            V = Vs[layer_name]
            U = Us[layer_name]
            ci_vals = ci_per_sample[layer_name]  # (n_samples, C)
            C = ci_vals.shape[1]

            if layer_idx % 2 == 0:
                dst_block = cj_idx
            else:
                dst_block = ci_idx
            ds, de = d * dst_block, d * (dst_block + 1)

            expected_comp_block = ci_idx if layer_idx in (0, 2) else cj_idx

            print(f"\n  Layer {layer_idx} ({layer_name}), dst block {dst_block}, "
                  f"expected comp block {expected_comp_block}")

            # =====================================================
            # Q1: Which components are active per sample?
            # =====================================================
            active_mask = ci_vals > CI_THRESHOLD  # (n_samples, C)

            # For each component: fraction of samples where it's active
            frac_active = active_mask.float().mean(dim=0)  # (C,)

            # Split into categories
            print(f"\n  Q1: Component activation frequency (CI > {CI_THRESHOLD})")

            # Expected computational components for this circuit at this layer
            exp_start = expected_comp_block * d
            exp_end = exp_start + d
            print(f"    Expected comp block {expected_comp_block} (components {exp_start}-{exp_end-1}):")
            for k in range(exp_start, exp_end):
                ci_when_active = ci_vals[:, k][active_mask[:, k]]
                mean_ci_active = ci_when_active.mean().item() if ci_when_active.numel() > 0 else 0
                print(f"      {k:3d}: active in {frac_active[k].item()*100:5.1f}% of samples"
                      f"  (mean CI when active: {mean_ci_active:.3f})")

            # Indexing components
            ohi_k = N_COMPUTATIONAL + ci_idx
            ohj_k = N_COMPUTATIONAL + N_ONE_HOT_I + cj_idx
            print(f"    ohi[{ci_idx}] (comp {ohi_k}): active in {frac_active[ohi_k].item()*100:.1f}%"
                  f"  mean CI={ci_vals[:, ohi_k].mean().item():.4f}")
            print(f"    ohj[{cj_idx}] (comp {ohj_k}): active in {frac_active[ohj_k].item()*100:.1f}%"
                  f"  mean CI={ci_vals[:, ohj_k].mean().item():.4f}")

            # Any OTHER components that are active?
            other_active = []
            for k in range(C):
                if k in range(exp_start, exp_end) or k == ohi_k or k == ohj_k:
                    continue
                if frac_active[k].item() > 0.01:  # active in >1% of samples
                    ci_when_active = ci_vals[:, k][active_mask[:, k]]
                    mean_ci_active = ci_when_active.mean().item() if ci_when_active.numel() > 0 else 0
                    other_active.append((k, frac_active[k].item(), mean_ci_active))

            if other_active:
                other_active.sort(key=lambda x: -x[1])
                print(f"    Other components active in >1% of samples:")
                for k, frac, mean_ci in other_active[:10]:
                    print(f"      {k:3d} ({component_label(k):>25s}): {frac*100:5.1f}%"
                          f"  (mean CI when active: {mean_ci:.3f})")
            else:
                print(f"    No other components active in >1% of samples")

            # =====================================================
            # Q2: Active subspace reconstruction
            # =====================================================
            post_target = F.relu(h_target @ W_t.T)
            post_eff = F.relu(per_sample_effective_output(h_eff, ci_vals, V, U))

            # Destination block only (the active subspace after suppression)
            dst_target = post_target[:, ds:de]
            dst_eff = post_eff[:, ds:de]

            dst_energy = mse(dst_target)
            dst_err = mse(dst_target - dst_eff)

            # For comparison: full 80-dim reconstruction
            full_energy = mse(post_target)
            full_err = mse(post_target - post_eff)

            print(f"\n  Q2: Reconstruction quality")
            print(f"    Full output (all {INPUT_DIM} dims):")
            print(f"      energy={full_energy:.6f}  MSE={full_err:.6f}"
                  f"  rel={full_err/full_energy*100 if full_energy > 1e-10 else 0:.1f}%")
            print(f"    Active subspace (dst block {dst_block}, dims {ds}-{de-1}):")
            print(f"      energy={dst_energy:.6f}  MSE={dst_err:.6f}"
                  f"  rel={dst_err/dst_energy*100 if dst_energy > 1e-10 else 0:.1f}%")

            # =====================================================
            # Q3: Do a small fixed set of comp components carry all CI?
            # =====================================================
            # For each sample, count how many computational components are active
            comp_active_per_sample = active_mask[:, :N_COMPUTATIONAL].sum(dim=1).float()
            # How many UNIQUE computational components are ever active across all samples?
            ever_active_comp = (frac_active[:N_COMPUTATIONAL] > 0.01).sum().item()
            # How much total CI mass is in computational vs indexing?
            comp_ci_mass = ci_vals[:, :N_COMPUTATIONAL].sum(dim=1).mean().item()
            idx_ci_mass = ci_vals[:, N_COMPUTATIONAL:N_TRUE].sum(dim=1).mean().item()

            print(f"\n  Q3: Component diversity")
            print(f"    Computational components active per sample: "
                  f"mean={comp_active_per_sample.mean().item():.1f}  "
                  f"median={comp_active_per_sample.median().item():.0f}  "
                  f"min={comp_active_per_sample.min().item():.0f}  "
                  f"max={comp_active_per_sample.max().item():.0f}")
            print(f"    Unique comp components ever active (>1% of samples): {ever_active_comp}")
            print(f"    Mean CI mass: computational={comp_ci_mass:.3f}  indexing={idx_ci_mass:.3f}")

            # Top computational components by total CI mass
            comp_ci_total = ci_vals[:, :N_COMPUTATIONAL].mean(dim=0)  # mean CI per component
            top_comp = comp_ci_total.topk(min(10, N_COMPUTATIONAL))
            print(f"    Top computational components by mean CI:")
            for rank_idx in range(len(top_comp.indices)):
                k = top_comp.indices[rank_idx].item()
                mean_ci = top_comp.values[rank_idx].item()
                frac = frac_active[k].item()
                print(f"      {k:3d} ({component_label(k):>25s}): mean_CI={mean_ci:.4f}"
                      f"  active_frac={frac*100:5.1f}%")

            # Update hidden state
            h_target = post_target
            h_eff = post_eff


if __name__ == "__main__":
    main()
