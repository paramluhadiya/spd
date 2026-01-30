"""Tests for Block-Structured Superposition (BSS) model."""

import torch

from spd.experiments.tms.bss_configs import BSSModelConfig, BSSTrainConfig
from spd.experiments.tms.bss_models import BSSModel


def test_bss_model_creation() -> None:
    """Test that a BSS model can be created with valid config."""
    config = BSSModelConfig(D=64, d=8, B=1e6, device="cpu")
    model = BSSModel(config)

    assert model.D == 64
    assert model.d == 8
    assert model.T == 64  # (64/8)^2 = 64 circuits
    assert model.num_blocks == 8


def test_bss_model_verify_all_circuits() -> None:
    """Test that all circuits pass verification with near-zero error."""
    config = BSSModelConfig(D=32, d=4, B=1e6, device="cpu")
    model = BSSModel(config)

    max_error = model.verify_all_circuits()
    assert max_error < 1e-5, f"Max error {max_error} too high"


def test_bss_model_forward_pass() -> None:
    """Test that forward pass produces correct output shape."""
    config = BSSModelConfig(D=64, d=8, B=1e6, device="cpu")
    model = BSSModel(config)

    batch_size = 4
    x = torch.zeros(batch_size, model.D)
    x[:, : model.d] = torch.randn(batch_size, model.d)
    circuits = torch.zeros(batch_size, dtype=torch.long)

    out = model(x, circuits)
    assert out.shape == (batch_size, model.D)


def test_bss_model_forward_single() -> None:
    """Test single-sample forward pass."""
    config = BSSModelConfig(D=32, d=4, B=1e6, device="cpu")
    model = BSSModel(config)

    x = torch.zeros(model.D)
    x[: model.d] = torch.randn(model.d)

    out = model.forward_single(x, active_circuit=0)
    assert out.shape == (model.D,)


def test_bss_model_state_dict() -> None:
    """Test that state_dict includes circuit weights/biases."""
    config = BSSModelConfig(D=32, d=4, B=1e6, device="cpu")
    model = BSSModel(config)

    state = model.state_dict()
    assert "_circuit_weights" in state
    assert "_circuit_biases" in state
    assert len(state["_circuit_weights"]) == model.T
    assert len(state["_circuit_biases"]) == model.T


def test_bss_model_load_state_dict() -> None:
    """Test that model can be saved and loaded correctly."""
    config = BSSModelConfig(D=32, d=4, B=1e6, device="cpu")
    model1 = BSSModel(config)

    # Get a test output before saving
    x = torch.zeros(model1.D)
    x[: model1.d] = torch.randn(model1.d)
    out1 = model1.forward_single(x, active_circuit=0)

    # Save and load
    state = model1.state_dict()
    model2 = BSSModel(config)
    model2.load_state_dict(state)

    # Check outputs match
    out2 = model2.forward_single(x, active_circuit=0)
    assert torch.allclose(out1, out2)


def test_bss_train_config() -> None:
    """Test BSSTrainConfig creation."""
    config = BSSTrainConfig(
        wandb_project=None,
        bss_model_config=BSSModelConfig(D=64, d=8, B=1e6, device="cpu"),
        seed=42,
    )
    assert config.seed == 42
    assert config.bss_model_config.D == 64
