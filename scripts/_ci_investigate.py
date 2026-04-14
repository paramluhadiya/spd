"""Quick investigation of c=72-79 CI patterns."""

import torch

from spd.models.component_model import ComponentModel

D, d, NUM_BLOCKS = 64, 8, 8


def main() -> None:
    model = ComponentModel.from_pretrained("wandb:paramluhadiya/spd/s-64ee330f")
    model.eval()
    device = next(model.parameters()).device
    name = "model.0"
    comps = list(range(72, 80))

    def header(comps: list[int]) -> str:
        h = f"{'':>10s}"
        for c in comps:
            h += f"  c={c:3d}"
        return h

    # 1. CI per j value
    print("=== CI of c=72-79 for each j value (i=0, random comp) ===")
    print(header(comps))
    for j_val in range(NUM_BLOCKS):
        x = torch.zeros(64, 80, device=device)
        x[:, 0:d] = torch.rand(64, d, device=device)
        x[:, D] = 1.0  # ohi = 0
        x[:, D + NUM_BLOCKS + j_val] = 1.0  # ohj = j_val
        with torch.no_grad():
            ci = model.calc_causal_importances({name: x}, sampling="continuous")
            ci_vals = ci.lower_leaky[name].mean(dim=0).cpu()
        row = f"{'j=' + str(j_val):>10s}"
        for c in comps:
            row += f"  {ci_vals[c]:.4f}"
        print(row)

    # 2. Independence from comp content (fix i=0, j=0)
    print("\n=== CI independence of comp content (i=0, j=0) ===")
    print(header(comps))
    for label, comp_vals in [
        ("zeros", torch.zeros(64, d)),
        ("rand_0.1", 0.1 * torch.rand(64, d)),
        ("rand_1.0", torch.rand(64, d)),
        ("rand_5.0", 5.0 * torch.rand(64, d)),
        ("ones", torch.ones(64, d)),
    ]:
        x = torch.zeros(64, 80, device=device)
        x[:, 0:d] = comp_vals.to(device)
        x[:, D] = 1.0
        x[:, D + NUM_BLOCKS] = 1.0  # j=0
        with torch.no_grad():
            ci = model.calc_causal_importances({name: x}, sampling="continuous")
            ci_vals = ci.lower_leaky[name].mean(dim=0).cpu()
        row = f"{label:>10s}"
        for c in comps:
            row += f"  {ci_vals[c]:.4f}"
        print(row)

    # 3. CI across different source blocks i (fix j=0)
    print("\n=== CI of c=72-79 across source blocks i (j=0) ===")
    print(header(comps))
    for i_val in range(NUM_BLOCKS):
        x = torch.zeros(64, 80, device=device)
        x[:, i_val * d : (i_val + 1) * d] = torch.rand(64, d, device=device)
        x[:, D + i_val] = 1.0  # ohi = i_val
        x[:, D + NUM_BLOCKS] = 1.0  # j=0
        with torch.no_grad():
            ci = model.calc_causal_importances({name: x}, sampling="continuous")
            ci_vals = ci.lower_leaky[name].mean(dim=0).cpu()
        row = f"{'i=' + str(i_val):>10s}"
        for c in comps:
            row += f"  {ci_vals[c]:.4f}"
        print(row)

    # 4. V vectors in indexing dims
    print("\n=== V[64:80, c] for c=72-79 ===")
    V = model.components[name].V.detach().cpu().float()
    for c in comps:
        v_idx = V[D:, c]
        v_ohi = v_idx[:NUM_BLOCKS]
        v_ohj = v_idx[NUM_BLOCKS:]
        print(f"  c={c}: ohi_abs=[{','.join(f'{x:.3f}' for x in v_ohi.abs().tolist())}]")
        print(f"        ohj_abs=[{','.join(f'{x:.3f}' for x in v_ohj.abs().tolist())}]")
        print(
            f"        ohi_argmax={int(v_ohi.abs().argmax())} "
            f"ohj_argmax={int(v_ohj.abs().argmax())}"
        )

    # 5. Also check: for j=0 high-CI comps, are they ALSO sensitive to j=1,2,...?
    # i.e., does each component fire for exactly one j?
    print("\n=== Top 15 CI components for block 0: their CI across all j values ===")
    x_b0 = torch.zeros(256, 80, device=device)
    x_b0[:, 0:d] = torch.rand(256, d, device=device)
    x_b0[:, D] = 1.0
    for s in range(256):
        x_b0[s, D + NUM_BLOCKS + (s % NUM_BLOCKS)] = 1.0
    with torch.no_grad():
        ci = model.calc_causal_importances({name: x_b0}, sampling="continuous")
        ci_all = ci.lower_leaky[name].cpu()  # (256, 600)
    mean_ci = ci_all.mean(dim=0)
    top15 = mean_ci.topk(15).indices.tolist()

    hdr = f"{'comp':>6s} {'mean_CI':>8s}"
    for jj in range(NUM_BLOCKS):
        hdr += f"  {'j=' + str(jj):>6s}"
    print(hdr)
    for c in top15:
        row = f"{c:>6d} {mean_ci[c]:>8.4f}"
        for jj in range(NUM_BLOCKS):
            # mean CI when j = jj
            mask = torch.zeros(256, dtype=torch.bool)
            for s in range(256):
                if s % NUM_BLOCKS == jj:
                    mask[s] = True
            ci_jj = ci_all[mask, c].mean().item()
            row += f"  {ci_jj:>6.4f}"
        print(row)


if __name__ == "__main__":
    main()
