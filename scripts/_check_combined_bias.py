"""Test the combined-bias hypothesis with proper V-sign accounting.

Decomposition reconstructs the target weight column-wise as:
    W[:, k] = Σ_c V[k, c] · U[c, :]

So the contribution of component c to bias column W[:, D+i*] is V[D+i*, c] · U[c, :],
NOT just U[c, :]. Summing raw U[c, :] vectors discards the V sign and magnitude.

Refined idea: the target-model bias from ohi[i] = 1 (column W[:, D+i]) is
implemented by the SUM of two sets of components:

  (A) NC[i] — i-specific, j-noncommital components: fire at all 8 j for source i
      and only for source i. Bias-like (fires whenever ohi[i] = 1).
  (B) H1-ohi (i*, j*) components: V → ohi[i*], routing-gated, fire only at
      (i*, j*). Per-circuit corrections.

For each H1-ohi (i*, j*, c) we test THREE reconstruction variants of W[:, D+i*]:

  - RAW:    Σ U[k, :]                          (ignores V, baseline I had before)
  - SIGN:   Σ sign(V[D+i*, k]) · U[k, :]       (sign-corrected sum)
  - VW:     Σ V[D+i*, k] · U[k, :]             (the actual decomposition contribution)

over k ∈ NC[i*] ∪ {c}. Each is compared to W[:, D+i*] both as a full vector and
restricted to the block-j* slice.

If the combined-bias story holds, the VW variant should match best — that's
literally the partial reconstruction that the decomposition is doing for these
components.

Usage:
    python scripts/_check_combined_bias.py /path/to/checkpoint
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8
LAYER = "model.0"
N_SAMPLES = 512
CI_ACTIVE = 0.3


def random_inputs(i_val: int, j_val: int, n: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(n, 80, device=device)
    x[:, i_val * d : (i_val + 1) * d] = torch.rand(n, d, device=device)
    x[:, D + i_val] = 1.0
    x[:, D + NUM_BLOCKS + j_val] = 1.0
    return x


def constant_input(i_val: int, j_val: int, val: float, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, 80, device=device)
    x[0, i_val * d : (i_val + 1) * d] = val
    x[0, D + i_val] = 1.0
    x[0, D + NUM_BLOCKS + j_val] = 1.0
    return x


def one_hot_input(i_val: int, j_val: int, r: int, device: torch.device) -> torch.Tensor:
    x = torch.zeros(1, 80, device=device)
    x[0, i_val * d + r] = 1.0
    x[0, D + i_val] = 1.0
    x[0, D + NUM_BLOCKS + j_val] = 1.0
    return x


def compute_ci_batch(model: ComponentModel, layer: str, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ci_out = model.calc_causal_importances({layer: x}, sampling="continuous")
    return ci_out.lower_leaky[layer].cpu()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    n = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / n).item()


def main() -> None:
    path = sys.argv[1]
    print(f"Loading {path} ...")
    model = ComponentModel.from_pretrained(path)
    model.eval()
    device = next(model.parameters()).device

    C = model.components[LAYER].V.shape[1]
    V = model.components[LAYER].V.detach().cpu().float()
    U = model.components[LAYER].U.detach().cpu().float()
    W = model.target_weight(LAYER).detach().cpu().float()  # (d_out=80, d_in=80)
    print(f"V {tuple(V.shape)}  U {tuple(U.shape)}  W {tuple(W.shape)}")

    # Auto-detect ohj masks
    mask_comps: set[int] = set()
    for c in range(C):
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        one_hotness = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
        if v_argmax >= D + NUM_BLOCKS and one_hotness > 0.95:
            mask_comps.add(c)

    # Compute (i, j) baseline grid
    print("Computing baseline (i, j) CI grid ...")
    ci_grid = torch.zeros(NUM_BLOCKS, NUM_BLOCKS, C)
    for i_val in range(NUM_BLOCKS):
        for j_val in range(NUM_BLOCKS):
            x = random_inputs(i_val, j_val, N_SAMPLES, device)
            ci_grid[i_val, j_val] = compute_ci_batch(model, LAYER, x).mean(dim=0)

    # ------------------------------------------------------------------
    # Find i-specific, j-noncommital components per i
    # ------------------------------------------------------------------
    print("\nFinding i-specific, j-noncommital components per i ...")
    noncommital: dict[int, list[int]] = {i: [] for i in range(NUM_BLOCKS)}
    for c in range(C):
        if c in mask_comps:
            continue
        # active in all 8 j for some unique i, none for other i
        for i in range(NUM_BLOCKS):
            n_at_i = sum(1 for j in range(NUM_BLOCKS) if ci_grid[i, j, c].item() > CI_ACTIVE)
            if n_at_i != NUM_BLOCKS:
                continue
            other_max = 0.0
            for ii in range(NUM_BLOCKS):
                if ii == i:
                    continue
                other_max = max(other_max, float(ci_grid[ii, :, c].max().item()))
            if other_max < CI_ACTIVE:
                noncommital[i].append(c)

    print(f"\n  i-specific, j-noncommital components per i:")
    for i in range(NUM_BLOCKS):
        info: list[str] = []
        for c in noncommital[i]:
            v = V[:, c]
            v_argmax = int(v.abs().argmax())
            v_1hot = (v[v_argmax] ** 2 / (v ** 2).sum()).item()
            if v_argmax < D:
                loc = f"comp[{v_argmax // d},{v_argmax % d}]"
            elif v_argmax < D + NUM_BLOCKS:
                loc = f"ohi[{v_argmax - D}]"
            else:
                loc = f"ohj[{v_argmax - D - NUM_BLOCKS}]"
            info.append(f"c={c}({loc},1hot={v_1hot:.2f})")
        print(f"    i={i}: N={len(noncommital[i])}  {info}")

    # ------------------------------------------------------------------
    # Find H1-ohi components: circuit-specific, V→ohi[i*], routing-gated
    # ------------------------------------------------------------------
    circuit_specific: list[tuple[int, int, int]] = []
    for c in range(C):
        if c in mask_comps:
            continue
        active = [
            (i, j)
            for i in range(NUM_BLOCKS)
            for j in range(NUM_BLOCKS)
            if ci_grid[i, j, c].item() > CI_ACTIVE
        ]
        if len(active) == 1:
            circuit_specific.append((active[0][0], active[0][1], c))

    print(f"\nFound {len(circuit_specific)} circuit-specific components total")
    print("Probing for H1-ohi (V→ohi[i*], routing-gated) ...")

    h1_ohi: list[tuple[int, int, int]] = []
    for i_val, j_val, c in circuit_specific:
        v = V[:, c]
        v_argmax = int(v.abs().argmax())
        if v_argmax != D + i_val:  # V must point at ohi[i*]
            continue
        ci_zero = compute_ci_batch(
            model, LAYER, constant_input(i_val, j_val, 0.0, device)
        )[0, c].item()
        ci_onehots = [
            compute_ci_batch(model, LAYER, one_hot_input(i_val, j_val, r, device))[0, c].item()
            for r in range(d)
        ]
        spread = max(ci_onehots) - min(ci_onehots)
        if ci_zero > 0.5 and spread < 0.3:
            h1_ohi.append((i_val, j_val, c))

    print(f"Found {len(h1_ohi)} H1-ohi components: {h1_ohi}")

    # ------------------------------------------------------------------
    # Combined-bias test
    # ------------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("COMBINED BIAS TEST")
    print(f"{'=' * 100}")
    print("Target = W[:, D+i*]  (bias contribution from ohi[i*] = 1, full output vector)")
    print("Hypothesis: U[H1] + Σ U[NC[i*]]  ≈  W[:, D+i*]")
    print()

    def reconstruct(comp_ids: list[int], i_val: int, mode: str) -> torch.Tensor:
        """Σ weight_c · U[c, :] over comp_ids, where weight depends on mode."""
        if not comp_ids:
            return torch.zeros(80)
        out = torch.zeros(80)
        for c in comp_ids:
            v_proj = float(V[D + i_val, c].item())
            if mode == "raw":
                w = 1.0
            elif mode == "sign":
                w = float(np.sign(v_proj)) if v_proj != 0 else 0.0
            elif mode == "vw":
                w = v_proj
            else:
                raise ValueError(mode)
            out = out + w * U[c, :]
        return out

    rows: list[dict] = []
    for i_val, j_val, c in h1_ohi:
        nc_list = noncommital[i_val]
        all_list = nc_list + [c]
        bias_full = W[:, D + i_val]
        j_slice = slice(j_val * d, (j_val + 1) * d)
        bias_j = bias_full[j_slice]

        # Per-component V projection on D+i*
        h1_v_proj = float(V[D + i_val, c].item())
        nc_v_projs = [float(V[D + i_val, k].item()) for k in nc_list]

        # Three reconstruction variants × three subsets (H1 only, NC only, combined)
        results: dict[str, dict[str, torch.Tensor]] = {}
        for mode in ["raw", "sign", "vw"]:
            results[mode] = {
                "h1": reconstruct([c], i_val, mode),
                "nc": reconstruct(nc_list, i_val, mode),
                "all": reconstruct(all_list, i_val, mode),
            }

        # Cosines for full vector and j*-block
        cos_full: dict[str, dict[str, float]] = {}
        cos_j: dict[str, dict[str, float]] = {}
        for mode in ["raw", "sign", "vw"]:
            cos_full[mode] = {
                k: cos(results[mode][k], bias_full) for k in ["h1", "nc", "all"]
            }
            cos_j[mode] = {
                k: cos(results[mode][k][j_slice], bias_j) for k in ["h1", "nc", "all"]
            }

        n_bias = bias_full.norm().item()
        n_all_vw = results["vw"]["all"].norm().item()

        rows.append(
            dict(
                c=c,
                i=i_val,
                j=j_val,
                nc=nc_list,
                h1_v_proj=h1_v_proj,
                nc_v_projs=nc_v_projs,
                cos_full=cos_full,
                cos_j=cos_j,
                n_bias=n_bias,
                n_all_vw=n_all_vw,
                results=results,
            )
        )

        print(f"\n--- c={c}  (i*={i_val}, j*={j_val})  NC[i*]={nc_list}")
        print(
            f"  V projections on D+i*:    H1={h1_v_proj:+.3f}   "
            f"NC={[f'{v:+.3f}' for v in nc_v_projs]}"
        )
        print(
            f"  ||W[:,D+i*]||={n_bias:.3f}   "
            f"||VW combined||={n_all_vw:.3f}   "
            f"(ratio = {n_all_vw / n_bias:.3f})"
        )
        print(f"  cos vs W[:, D+i*]   FULL VECTOR:")
        print(
            f"    {'mode':>6s}  {'H1 only':>10s}  {'Σ NC only':>10s}  {'H1 + Σ NC':>12s}"
        )
        for mode in ["raw", "sign", "vw"]:
            tag = "  <==" if mode == "vw" else ""
            print(
                f"    {mode:>6s}  {cos_full[mode]['h1']:>+10.4f}  "
                f"{cos_full[mode]['nc']:>+10.4f}  "
                f"{cos_full[mode]['all']:>+12.4f}{tag}"
            )
        print(f"  cos vs W[j*-blk, D+i*]   BLOCK-j*:")
        print(
            f"    {'mode':>6s}  {'H1 only':>10s}  {'Σ NC only':>10s}  {'H1 + Σ NC':>12s}"
        )
        for mode in ["raw", "sign", "vw"]:
            tag = "  <==" if mode == "vw" else ""
            print(
                f"    {mode:>6s}  {cos_j[mode]['h1']:>+10.4f}  "
                f"{cos_j[mode]['nc']:>+10.4f}  "
                f"{cos_j[mode]['all']:>+12.4f}{tag}"
            )

    # Aggregate
    if rows:
        print(f"\n{'=' * 100}")
        print(f"AGGREGATE (N={len(rows)})")
        print(f"{'=' * 100}")
        print("\n  cos(reconstruction, W[:, D+i*])  FULL VECTOR")
        print(
            f"    {'mode':>6s}  {'H1 mean':>10s}  {'NC mean':>10s}  {'H1+NC mean':>12s}  "
            f"{'H1+NC med':>12s}  {'H1+NC max':>12s}"
        )
        for mode in ["raw", "sign", "vw"]:
            h1s = [r["cos_full"][mode]["h1"] for r in rows]
            ncs = [r["cos_full"][mode]["nc"] for r in rows]
            alls = [r["cos_full"][mode]["all"] for r in rows]
            print(
                f"    {mode:>6s}  {np.mean(h1s):>+10.4f}  {np.mean(ncs):>+10.4f}  "
                f"{np.mean(alls):>+12.4f}  {np.median(alls):>+12.4f}  "
                f"{np.max(alls):>+12.4f}"
            )

        print("\n  cos(reconstruction, W[j*-blk, D+i*])  BLOCK-j*")
        print(
            f"    {'mode':>6s}  {'H1 mean':>10s}  {'NC mean':>10s}  {'H1+NC mean':>12s}  "
            f"{'H1+NC med':>12s}  {'H1+NC max':>12s}"
        )
        for mode in ["raw", "sign", "vw"]:
            h1s = [r["cos_j"][mode]["h1"] for r in rows]
            ncs = [r["cos_j"][mode]["nc"] for r in rows]
            alls = [r["cos_j"][mode]["all"] for r in rows]
            print(
                f"    {mode:>6s}  {np.mean(h1s):>+10.4f}  {np.mean(ncs):>+10.4f}  "
                f"{np.mean(alls):>+12.4f}  {np.median(alls):>+12.4f}  "
                f"{np.max(alls):>+12.4f}"
            )

        # Did the VW combined improve over the better of (H1, NC) per row?
        for mode in ["raw", "sign", "vw"]:
            better_full = sum(
                1
                for r in rows
                if r["cos_full"][mode]["all"]
                > max(r["cos_full"][mode]["h1"], r["cos_full"][mode]["nc"])
            )
            better_j = sum(
                1
                for r in rows
                if r["cos_j"][mode]["all"]
                > max(r["cos_j"][mode]["h1"], r["cos_j"][mode]["nc"])
            )
            print(
                f"\n  [{mode:>4s}]  #(combined FULL  > max(H1, NC)):  {better_full}/{len(rows)}"
                f"     #(combined BLK-j* > max(H1, NC)): {better_j}/{len(rows)}"
            )

    # ------------------------------------------------------------------
    # Plot the VW reconstruction (the principled one) per H1-ohi
    # ------------------------------------------------------------------
    if rows:
        n = len(rows)
        fig, axes = plt.subplots(n, 2, figsize=(16, 2.4 * n))
        if n == 1:
            axes = axes.reshape(1, 2)

        for k, r in enumerate(rows):
            i_val, j_val = r["i"], r["j"]
            recon_all_vw = r["results"]["vw"]["all"].numpy()
            recon_nc_vw = r["results"]["vw"]["nc"].numpy()
            recon_h1_vw = r["results"]["vw"]["h1"].numpy()
            bias_full = W[:, D + i_val].numpy()

            ax = axes[k, 0]
            x_pos = np.arange(80)
            w = 0.28
            ax.bar(x_pos - w, recon_all_vw, w, label="VW: H1 + Σ NC", color="#1f77b4")
            ax.bar(x_pos, bias_full, w, label="W[:, D+i*]", color="#d62728")
            ax.bar(x_pos + w, recon_nc_vw, w, label="VW: Σ NC", color="#888888", alpha=0.6)
            for rb in range(NUM_BLOCKS + 1):
                ax.axvline(rb * d - 0.5, color="gray", linestyle=":", alpha=0.4)
            ax.axvspan(j_val * d - 0.5, (j_val + 1) * d - 0.5, alpha=0.12, color="yellow")
            ax.axhline(0, color="black", lw=0.3)
            ax.set_xlim(-0.5, 79.5)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(
                f"c={r['c']}  (i*={i_val}, j*={j_val})  NC={r['nc']}   "
                f"cos_full vw: H1={r['cos_full']['vw']['h1']:+.2f}  "
                f"NC={r['cos_full']['vw']['nc']:+.2f}  "
                f"H1+NC={r['cos_full']['vw']['all']:+.2f}",
                fontsize=8,
            )

            ax = axes[k, 1]
            j_slice = slice(j_val * d, (j_val + 1) * d)
            x = np.arange(d)
            ax.bar(x - w, recon_all_vw[j_slice], w, label="VW: H1+NC", color="#1f77b4")
            ax.bar(x, bias_full[j_slice], w, label="W[j*,D+i*]", color="#d62728")
            ax.bar(x + w, recon_h1_vw[j_slice], w, label="VW: H1 only", color="#444444")
            ax.axhline(0, color="black", lw=0.3)
            ax.set_xticks(x)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_title(
                f"BLOCK-j* zoom   cos_j vw: H1={r['cos_j']['vw']['h1']:+.2f}  "
                f"NC={r['cos_j']['vw']['nc']:+.2f}  "
                f"H1+NC={r['cos_j']['vw']['all']:+.2f}",
                fontsize=8,
            )

        plt.tight_layout()
        out = "combined_bias.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
