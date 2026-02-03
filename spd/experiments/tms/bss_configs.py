"""Configuration classes for Block-Structured Superposition (BSS) model."""

from typing import Self

from pydantic import PositiveInt, model_validator

from spd.base_config import BaseConfig


class BSSModelConfig(BaseConfig):
    """Config for Block-Structured Superposition model.

    The BSS model implements T = (D/d)^2 circuits in a network of width D.
    Each circuit is specified by maps f, g, k and has its own (d x d) weight
    matrix and (d,) bias vector.

    Attributes:
        D: Network width (must be divisible by d)
        d: Block/circuit width
        B: Suppression strength for inactive neurons
        device: Device to run on
    """

    D: PositiveInt
    d: PositiveInt
    B: float = 1e6
    device: str = "gpu"

    @model_validator(mode="after")
    def validate_divisibility(self) -> Self:
        assert self.D % self.d == 0, f"D ({self.D}) must be divisible by d ({self.d})"
        return self

    @property
    def num_blocks(self) -> int:
        return self.D // self.d

    @property
    def num_circuits(self) -> int:
        return self.num_blocks**2


class BSSTrainConfig(BaseConfig):
    """Config for initializing/saving a BSS model.

    Note: BSS models are constructed (not trained via gradient descent),
    so this is mainly for initialization and verification.
    """

    wandb_project: str | None = None
    bss_model_config: BSSModelConfig
    seed: int = 0
