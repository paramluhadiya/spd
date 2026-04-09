from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _faithfulness_loss_update(
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
) -> tuple[Float[Tensor, ""], int]:
    assert weight_deltas, "Empty weight deltas"
    device = get_obj_device(weight_deltas)
    sum_loss = torch.tensor(0.0, device=device)
    total_params = 0
    for delta in weight_deltas.values():
        sum_loss += (delta**2).sum()
        total_params += delta.numel()
    return sum_loss, total_params


def _faithfulness_loss_compute(
    sum_loss: Float[Tensor, ""], total_params: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / total_params


def faithfulness_loss(weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]) -> Float[Tensor, ""]:
    sum_loss, total_params = _faithfulness_loss_update(weight_deltas)
    return _faithfulness_loss_compute(sum_loss, total_params)


def threshold_faithfulness_loss(
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
    model: ComponentModel,
    threshold: float = 5.0,
) -> Float[Tensor, ""]:
    """Faithfulness loss that stops penalizing suppression terms once they're 'working'.

    For parameters where target < -threshold (suppression terms):
      - If learned < -threshold: no penalty (suppression is working)
      - If learned >= -threshold: penalty = relu(learned + threshold)²

    For other parameters: normal MSE (delta²).

    Args:
        weight_deltas: Dict mapping module name to (target - learned) tensor
        model: ComponentModel to get target weights from
        threshold: Suppression is "working" if learned < -threshold
    """
    assert weight_deltas, "Empty weight deltas"
    device = get_obj_device(weight_deltas)
    sum_loss = torch.tensor(0.0, device=device)
    total_params = 0

    for name, delta in weight_deltas.items():
        target = model.target_weight(name)
        learned = target - delta

        # Identify suppression terms: target < -threshold
        is_suppression = target < -threshold

        # Suppression penalty: relu(learned + threshold)²
        # This is 0 if learned < -threshold (working), positive otherwise
        suppression_penalty = torch.relu(learned + threshold) ** 2

        # Normal penalty: delta²
        normal_penalty = delta**2

        # Combine: use suppression_penalty for suppression terms, normal_penalty for others
        penalty = torch.where(is_suppression, suppression_penalty, normal_penalty)
        sum_loss += penalty.sum()
        total_params += delta.numel()

    return sum_loss / total_params


class FaithfulnessLoss(Metric):
    """MSE between the target weights and the sum of the components."""

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.sum_loss = torch.tensor(0.0, device=device)
        self.total_params = torch.tensor(0, device=device)

    @override
    def update(self, *, weight_deltas: dict[str, Float[Tensor, "d_out d_in"]], **_: Any) -> None:
        sum_loss, total_params = _faithfulness_loss_update(weight_deltas)
        self.sum_loss += sum_loss
        self.total_params += total_params

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        total_params = all_reduce(self.total_params, op=ReduceOp.SUM)
        return _faithfulness_loss_compute(sum_loss, total_params)
