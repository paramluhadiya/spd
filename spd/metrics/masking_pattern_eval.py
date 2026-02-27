"""Masking pattern evaluation for PingPong model decomposition.

Checks if the SPD decomposition learns components that capture the
masking/suppression behavior of the PingPong model.

For each layer:
- Even layers (0, 2): j index controls which block survives
- Odd layers (1): i index controls which block survives

For each input, we:
1. Find components firing for this input (CI > ci_threshold)
2. Among firing components, find ones whose V aligns with the relevant one-hot
3. Sum their U vectors to form a "virtual masking component"
4. Check if the virtual component has the correct suppression pattern
"""

from typing import Any, ClassVar, override

import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.distributed_utils import all_reduce


class MaskingPatternEval(Metric):
    """Evaluate if decomposition learns correct masking components for PingPong."""

    metric_section: ClassVar[str] = "masking"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        D: int,
        d: int,
        cos_sim_threshold: float = 0.9,
        ci_threshold: float = 0.1,
    ) -> None:
        self.model = model
        self.device = device
        self.D = D
        self.d = d
        self.num_blocks = D // d
        self.cos_sim_threshold = cos_sim_threshold
        self.ci_threshold = ci_threshold
        self.input_dim = D + 2 * self.num_blocks

        # Accumulate per-layer results across update() calls
        self.coverage_per_layer: dict[str, list[float]] = {}
        self.pattern_accuracy_per_layer: dict[str, list[float]] = {}
        self.suppression_strength_per_layer: dict[str, list[float]] = {}
        self.margin_per_layer: dict[str, list[float]] = {}

        for layer_name in model.target_module_paths:
            self.coverage_per_layer[layer_name] = []
            self.pattern_accuracy_per_layer[layer_name] = []
            self.suppression_strength_per_layer[layer_name] = []
            self.margin_per_layer[layer_name] = []

    def _get_layer_index(self, layer_name: str) -> int:
        """Extract layer index from name (e.g., 'model.0' -> 0, 'model.2' -> 1)."""
        parts = layer_name.split(".")
        layer_num = int(parts[-1])
        # In PingPong, linear layers are at indices 0, 2, 4 in the Sequential
        return layer_num // 2

    def _extract_block_indices(
        self, batch: Float[Tensor, "batch input_dim"]
    ) -> tuple[Tensor, Tensor]:
        """Extract i and j block indices from the one-hot encodings in the batch."""
        one_hot_i = batch[:, self.D : self.D + self.num_blocks]
        one_hot_j = batch[:, self.D + self.num_blocks :]
        i_indices = one_hot_i.argmax(dim=1)
        j_indices = one_hot_j.argmax(dim=1)
        return i_indices, j_indices

    def _compute_per_input_metrics(
        self,
        layer_name: str,
        V: Float[Tensor, "d_in C"],
        U: Float[Tensor, "C d_out"],
        layer_ci: Float[Tensor, "batch C"],
        i_indices: Tensor,
        j_indices: Tensor,
    ) -> dict[str, float]:
        """Compute masking metrics for a layer, evaluated per input (vectorized).

        For each input:
        1. Select components with CI > ci_threshold for THIS input
        2. Among those, find V-aligned components for the relevant one-hot
        3. Sum their U vectors and evaluate the masking pattern
        """
        layer_idx = self._get_layer_index(layer_name)
        batch_size = layer_ci.shape[0]
        nb = self.num_blocks

        # Build one-hot targets for all blocks: (num_blocks, input_dim)
        targets = torch.zeros(nb, self.input_dim, device=self.device)
        block_indices = torch.arange(nb, device=self.device)
        if layer_idx % 2 == 0:
            targets[block_indices, self.D + nb + block_indices] = 1.0
        else:
            targets[block_indices, self.D + block_indices] = 1.0

        # Cosine similarity: (num_blocks, C)
        V_norm = F.normalize(V.T, dim=1)  # (C, d_in)
        targets_norm = F.normalize(targets, dim=1)  # (num_blocks, d_in)
        cos_sims = targets_norm @ V_norm.T  # (num_blocks, C)

        # Even layers: j controls masking, odd layers: i controls masking
        active_blocks = j_indices if layer_idx % 2 == 0 else i_indices  # (batch,)

        # Per-input alignment: gather the cos_sim row for each input's active block
        # cos_sims[active_blocks] -> (batch, C)
        per_input_cos_sims = cos_sims[active_blocks]  # (batch, C)
        per_input_aligned = per_input_cos_sims.abs() > self.cos_sim_threshold  # (batch, C)

        # Per-input firing
        firing = layer_ci > self.ci_threshold  # (batch, C)

        # Combined mask: firing AND aligned, weighted by sign of cos_sim
        # Sign-flip ensures negatively-aligned components contribute -U
        selected = firing & per_input_aligned  # (batch, C)
        mask = selected.float() * per_input_cos_sims.sign()  # (batch, C)

        # Inputs that have at least one matching component
        has_components = selected.any(dim=1)  # (batch,)
        n_with_components = has_components.sum().item()

        if n_with_components == 0:
            return {
                "coverage": 0.0,
                "pattern_accuracy": 0.0,
                "suppression_strength": 0.0,
                "margin": 0.0,
            }

        # Virtual U per input: (batch, C) @ (C, d_out) -> (batch, d_out)
        # Sign-weighted so negatively-aligned V components contribute -U
        U_virtual = mask @ U  # (batch, d_out)

        # Reshape computing block to (batch, num_blocks, d)
        U_blocks = U_virtual[:, : self.D].reshape(batch_size, nb, self.d)

        # For suppressed blocks: take max over d (worst-case leak, least negative)
        # For active block: take min over d (worst-case kill, most negative neuron)
        block_maxes = U_blocks.max(dim=2).values  # (batch, num_blocks)
        block_mins = U_blocks.min(dim=2).values  # (batch, num_blocks)

        # Build suppressed mask: (batch, num_blocks) with False at active block
        suppressed = torch.ones(batch_size, nb, dtype=torch.bool, device=self.device)
        batch_idx = torch.arange(batch_size, device=self.device)
        suppressed[batch_idx, active_blocks] = False

        # Active block: min value (worst-case neuron that might get killed)
        active_min = block_mins[batch_idx, active_blocks]  # (batch,)

        # Suppressed blocks: max value per block (worst-case leak), then take max across blocks
        # Set active block to -inf so it doesn't affect the max
        suppressed_maxes = block_maxes.clone()
        suppressed_maxes[batch_idx, active_blocks] = float("-inf")
        worst_leak = suppressed_maxes.max(dim=1).values  # (batch,)

        # Pattern accuracy: active block min should exceed suppressed blocks' max
        correct = (active_min > worst_leak) & has_components
        pattern_accuracy = correct.sum().item() / n_with_components

        # Margin: active block min - worst suppressed leak (higher = cleaner separation)
        margin = (active_min - worst_leak)[has_components].mean().item()

        # Suppression strength: mean of (-max) across suppressed blocks (higher = more negative)
        suppressed_max_mean = (
            (block_maxes * suppressed).sum(dim=1) / (nb - 1)
        )  # (batch,)
        suppression_strength = (-suppressed_max_mean)[has_components].mean().item()

        coverage = n_with_components / batch_size

        return {
            "coverage": coverage,
            "pattern_accuracy": pattern_accuracy,
            "suppression_strength": suppression_strength,
            "margin": margin,
        }

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        """Evaluate masking pattern per input in the batch."""
        assert batch.ndim == 2, f"Expected 2D batch (batch, input_dim), got {batch.shape}"
        i_indices, j_indices = self._extract_block_indices(batch.float())

        for layer_name in self.model.target_module_paths:
            components = self.model.components[layer_name]
            V = components.V.detach()  # (d_in, C)
            U = components.U.detach()  # (C, d_out)

            layer_ci = ci.lower_leaky[layer_name]  # (batch, C)
            assert layer_ci.ndim == 2, (
                f"Expected 2D CI (batch, C), got {layer_ci.shape}"
            )

            metrics = self._compute_per_input_metrics(
                layer_name, V, U, layer_ci, i_indices, j_indices
            )

            self.coverage_per_layer[layer_name].append(metrics["coverage"])
            self.pattern_accuracy_per_layer[layer_name].append(metrics["pattern_accuracy"])
            self.suppression_strength_per_layer[layer_name].append(metrics["suppression_strength"])
            self.margin_per_layer[layer_name].append(metrics["margin"])

    @override
    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        """Compute final aggregated metrics."""
        out: dict[str, float | wandb.plot.CustomChart] = {}

        all_coverages = []
        all_accuracies = []
        all_strengths = []
        all_margins = []

        for layer_name in self.model.target_module_paths:
            coverages = self.coverage_per_layer[layer_name]
            accuracies = self.pattern_accuracy_per_layer[layer_name]
            strengths = self.suppression_strength_per_layer[layer_name]
            margins = self.margin_per_layer[layer_name]

            if coverages:
                coverage_sum = all_reduce(
                    torch.tensor(coverages, device=self.device).sum(), op=ReduceOp.SUM
                )
                count = all_reduce(
                    torch.tensor(len(coverages), device=self.device), op=ReduceOp.SUM
                )
                avg_coverage = (coverage_sum / count).item()
                out[f"coverage_{layer_name}"] = avg_coverage
                all_coverages.append(avg_coverage)

                avg_acc = (
                    all_reduce(
                        torch.tensor(accuracies, device=self.device).sum(), op=ReduceOp.SUM
                    )
                    / count
                ).item()
                out[f"pattern_accuracy_{layer_name}"] = avg_acc
                all_accuracies.append(avg_acc)

                avg_str = (
                    all_reduce(
                        torch.tensor(strengths, device=self.device).sum(), op=ReduceOp.SUM
                    )
                    / count
                ).item()
                out[f"suppression_strength_{layer_name}"] = avg_str
                all_strengths.append(avg_str)

                avg_margin = (
                    all_reduce(
                        torch.tensor(margins, device=self.device).sum(), op=ReduceOp.SUM
                    )
                    / count
                ).item()
                out[f"margin_{layer_name}"] = avg_margin
                all_margins.append(avg_margin)

        if all_coverages:
            out["coverage_model"] = sum(all_coverages) / len(all_coverages)
        if all_accuracies:
            out["pattern_accuracy_model"] = sum(all_accuracies) / len(all_accuracies)
        if all_strengths:
            out["suppression_strength_model"] = sum(all_strengths) / len(all_strengths)
        if all_margins:
            out["margin_model"] = sum(all_margins) / len(all_margins)

        return out
