"""Check if scaled-down components stayed tiny during training.

Compares V and U norms across all 600 components for components 0-79 (alive at init)
vs 80-599 (scaled down by 0.01 at init).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spd.models.component_model import ComponentModel

LAYER = "model.0"
N_TRUE_OLD = 80   # for old scale_down threshold


def main() -> None:
    path = sys.argv[1]
    model = ComponentModel.from_pretrained(path)
    V = model.components[LAYER].V.detach().cpu().float()  # (80, C)
    U = model.components[LAYER].U.detach().cpu().float()  # (C, 80)
    C = V.shape[1]
    print(f"C = {C}")

    v_norms = V.norm(dim=0)              # (C,)
    u_norms = U.norm(dim=1)              # (C,)
    vu_outer_norms = v_norms * u_norms   # rough scale of V[:,c] @ U[c,:]

    # Init values (computed analytically)
    init_v_norm = (1.0 / 80) ** 0.5 * (80 ** 0.5)   # = 1.0 expected for full Kaiming column over 80 entries
    init_u_norm = (1.0 / C) ** 0.5 * (80 ** 0.5)    # for U row over 80 entries
    print("\nExpected init norms (Kaiming):")
    print(f"  V column norm ≈ {init_v_norm:.4f}")
    print(f"  U row norm ≈ {init_u_norm:.4f}")
    print(f"  After scale_down (×0.01): V≈{init_v_norm*0.01:.6f}, U≈{init_u_norm*0.01:.6f}")

    # Group stats
    alive = slice(0, N_TRUE_OLD)
    scaled = slice(N_TRUE_OLD, C)

    print(f"\n=== Alive group (c=0..{N_TRUE_OLD-1}, {N_TRUE_OLD} components) ===")
    print(f"  V norm: mean={v_norms[alive].mean():.4f}  median={v_norms[alive].median():.4f}  "
          f"min={v_norms[alive].min():.4f}  max={v_norms[alive].max():.4f}")
    print(f"  U norm: mean={u_norms[alive].mean():.4f}  median={u_norms[alive].median():.4f}  "
          f"min={u_norms[alive].min():.4f}  max={u_norms[alive].max():.4f}")
    print(f"  V*U  : mean={vu_outer_norms[alive].mean():.4f}  median={vu_outer_norms[alive].median():.4f}")

    print(f"\n=== Scaled-down group (c={N_TRUE_OLD}..{C-1}, {C-N_TRUE_OLD} components) ===")
    print(f"  V norm: mean={v_norms[scaled].mean():.4f}  median={v_norms[scaled].median():.4f}  "
          f"min={v_norms[scaled].min():.4f}  max={v_norms[scaled].max():.4f}")
    print(f"  U norm: mean={u_norms[scaled].mean():.4f}  median={u_norms[scaled].median():.4f}  "
          f"min={u_norms[scaled].min():.4f}  max={u_norms[scaled].max():.4f}")
    print(f"  V*U  : mean={vu_outer_norms[scaled].mean():.4f}  median={vu_outer_norms[scaled].median():.4f}")

    # How many scaled-down components grew to comparable size?
    alive_median_vu = vu_outer_norms[alive].median().item()
    n_scaled_grown = int((vu_outer_norms[scaled] > 0.5 * alive_median_vu).sum())
    n_scaled_recovered = int((vu_outer_norms[scaled] > 0.1 * alive_median_vu).sum())
    print(f"\n  # scaled-down with V*U > 50% of alive median: {n_scaled_grown}/{C-N_TRUE_OLD}")
    print(f"  # scaled-down with V*U > 10% of alive median: {n_scaled_recovered}/{C-N_TRUE_OLD}")

    # Print detailed top scaled-down components
    print("\n  Top 15 scaled-down components by V*U norm:")
    top_vals, top_idxs = vu_outer_norms[scaled].topk(15)
    for val, idx_in_scaled in zip(top_vals, top_idxs):
        c = int(idx_in_scaled) + N_TRUE_OLD
        print(f"    c={c}  V_norm={v_norms[c]:.4f}  U_norm={u_norms[c]:.4f}  V*U={val:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(v_norms.numpy(), '.', markersize=2)
    axes[0].axvline(N_TRUE_OLD, color='red', linestyle='--', alpha=0.5, label=f'scale_down threshold (c={N_TRUE_OLD})')
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("V column norm")
    axes[0].set_title("V column norms")
    axes[0].set_yscale('log')
    axes[0].legend()

    axes[1].plot(u_norms.numpy(), '.', markersize=2)
    axes[1].axvline(N_TRUE_OLD, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Component index")
    axes[1].set_ylabel("U row norm")
    axes[1].set_title("U row norms")
    axes[1].set_yscale('log')

    axes[2].plot(vu_outer_norms.numpy(), '.', markersize=2)
    axes[2].axvline(N_TRUE_OLD, color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel("Component index")
    axes[2].set_ylabel("V_norm × U_norm (rough V@U scale)")
    axes[2].set_title("Outer product scale per component")
    axes[2].set_yscale('log')

    plt.tight_layout()
    out = "scaled_norms.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    main()
