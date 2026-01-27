"""Block-Structured Superposition (BSS) model.

This model demonstrates computation in superposition where exactly one circuit
is active per forward pass. It can emulate T = (D/d)^2 different circuits using
a fixed ReLU network of width D(1+2/d).

Based on:
- https://www.lesswrong.com/posts/g9uMJkcWj8jQDjybb/ping-pong-computation-in-superposition
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from spd.experiments.tms.bss_configs import BSSModelConfig, BSSTrainConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.spd_types import ModelPath


@dataclass
class BSSTargetRunInfo(RunInfo[BSSTrainConfig]):
    """Run info from initializing a BSSModel."""

    config_class = BSSTrainConfig
    config_filename = "bss_train_config.yaml"
    checkpoint_filename = "bss.pth"


class BSSModel(LoadableModule):
    """Block-Structured Superposition Model.

    Implements T = (D/d)^2 circuits in a network of width D.
    Each circuit is specified by maps f, g, k and has its own
    (d x d) weight matrix and (d,) bias vector.

    The model uses the "ping-pong" construction:
    1. Input x_block is in block coordinates (D,)
    2. Active circuit determines which block to read from (f map)
    3. Computation happens in network coordinates via W_g scatter
    4. Results are gathered back to block coordinates via W_h

    Forward pass (one layer):
        v = one_hot(f[active_circuit])           # Active block indicator
        u = indicator for active neuron set      # Which neurons to use
        y = ReLU(W_b' @ u + W_b @ v + W_g @ x)   # Network computation
        z = ReLU(W_h @ y + B_h @ v)              # Gather back to block coords
    """

    def __init__(self, config: BSSModelConfig):
        super().__init__()
        self.config = config
        D, d = config.D, config.d
        num_blocks = D // d
        T = num_blocks**2

        self.D = D
        self.d = d
        self.T = T
        self.num_blocks = num_blocks

        # Setup maps f, g, k
        self.f, self.g, self.k = self._setup_maps()

        # Store circuit weights and biases
        # Using regular tensors stored in a dict, not nn.Parameters
        # since they're conceptual - the actual parameters are W_g etc.
        self._circuit_weights: dict[int, Tensor] = {}
        self._circuit_biases: dict[int, Tensor] = {}

        # Initialize random circuits
        for t in range(T):
            self._circuit_weights[t] = torch.randn(d, d) * 0.5
            self._circuit_biases[t] = torch.randn(d)

        # Build and register weight matrices as parameters
        # These are the actual network weights
        self.W_g = nn.Linear(D, D, bias=False)
        self.W_b = nn.Linear(num_blocks, D, bias=False)
        self.W_b_prime = nn.Linear(num_blocks, D, bias=False)
        self.W_h = nn.Linear(D, D, bias=False)
        self.B_h = nn.Linear(num_blocks, D, bias=False)

        # Build the weight matrices from circuit definitions
        self._build_weight_matrices()

    def _setup_maps(
        self,
    ) -> tuple[dict[int, int], dict[tuple[int, int], int], dict[int, tuple[int, ...]]]:
        """Create f, g, k maps using contiguous neuron sets.

        Returns:
            f: dict circuit -> block (which block a circuit reads from)
            g: dict (circuit, pos) -> neuron (which neurons a circuit uses)
            k: dict block -> neuron_set (tuple of neurons belonging to each block)
        """
        d = self.d
        num_blocks = self.num_blocks
        T = self.T
        circuits_per_block = T // num_blocks

        assert num_blocks * circuits_per_block == T

        # Create neuron sets (contiguous)
        neuron_sets = []
        for i in range(num_blocks):
            neuron_sets.append(tuple(range(d * i, d * (i + 1))))

        f: dict[int, int] = {}
        g: dict[tuple[int, int], int] = {}
        k: dict[int, tuple[int, ...]] = {}

        # Assign circuits to blocks and neuron sets
        circuit_id = 0
        for block in range(num_blocks):
            k[block] = neuron_sets[block]

            for circuit_in_block in range(circuits_per_block):
                f[circuit_id] = block

                # This circuit uses the circuit_in_block'th neuron set
                neuron_set = neuron_sets[circuit_in_block]

                for pos in range(d):
                    g[(circuit_id, pos)] = neuron_set[pos]

                circuit_id += 1

        return f, g, k

    def _build_Wg(self) -> Float[Tensor, "D D"]:
        """Build W_g: scatters from block coordinates to network coordinates.

        (W_g @ x)_i = sum over (t',n) where g(t',n)=i of:
                      sum over k in [d]: W^{t'}_{n,k} * x_{d*f(t')+k}
        """
        D, d, T = self.D, self.d, self.T
        W_g = torch.zeros(D, D)

        for i in range(D):
            for t in range(T):
                for n in range(d):
                    if self.g.get((t, n)) == i:
                        W_t = self._circuit_weights[t]
                        block_start = d * self.f[t]

                        for k_idx in range(d):
                            W_g[i, block_start + k_idx] += W_t[n, k_idx]

        return W_g

    def _build_Wb(self) -> Float[Tensor, "D num_blocks"]:
        """Build W_b: provides biases based on active block.

        W_b @ e_i gives bias vector for block i
        (W_b @ e_i)_j = b^t_k when f(t)=i and g(t,k)=j
        """
        D, d, T = self.D, self.d, self.T
        num_blocks = self.num_blocks
        W_b = torch.zeros(D, num_blocks)

        for block_i in range(num_blocks):
            bias_vec = torch.zeros(D)

            for t in range(T):
                if self.f[t] == block_i:
                    b_t = self._circuit_biases[t]
                    for k_idx in range(d):
                        neuron_j = self.g[(t, k_idx)]
                        bias_vec[neuron_j] = b_t[k_idx]

            W_b[:, block_i] = bias_vec

        return W_b

    def _build_Wb_prime(self) -> Float[Tensor, "D num_blocks"]:
        """Build W_b': suppresses neurons not in active neuron set.

        (W_b' @ e_i)_j = -B if j not in k(i), else 0
        """
        D = self.D
        num_blocks = self.num_blocks
        B = self.config.B
        W_b_prime = torch.zeros(D, num_blocks)

        for block_i in range(num_blocks):
            neuron_set = self.k[block_i]

            for j in range(D):
                if j not in neuron_set:
                    W_b_prime[j, block_i] = -B

        return W_b_prime

    def _build_Wh(self) -> Float[Tensor, "D D"]:
        """Build W_h: gathers from network coordinates back to block coordinates.

        (W_h @ y)_{d*q + r} = sum over j in g(f^{-1}(q) x {r}): y_j
        """
        D, d, T = self.D, self.d, self.T
        num_blocks = self.num_blocks
        W_h = torch.zeros(D, D)

        for q in range(num_blocks):
            for r in range(d):
                output_pos = d * q + r

                for t in range(T):
                    if self.f[t] == q:
                        neuron_j = self.g[(t, r)]
                        W_h[output_pos, neuron_j] += 1.0

        return W_h

    def _build_Bh(self) -> Float[Tensor, "D num_blocks"]:
        """Build B_h: suppresses outputs outside active block.

        (B_h @ e_i)_j = -B if j not in [d*i, d*(i+1)), else 0
        """
        D, d = self.D, self.d
        num_blocks = self.num_blocks
        B = self.config.B
        B_h = torch.zeros(D, num_blocks)

        for block_i in range(num_blocks):
            block_start = d * block_i
            block_end = d * (block_i + 1)

            for j in range(D):
                if not (block_start <= j < block_end):
                    B_h[j, block_i] = -B

        return B_h

    def _build_weight_matrices(self) -> None:
        """Build all weight matrices from circuit definitions."""
        self.W_g.weight.data = self._build_Wg()
        self.W_b.weight.data = self._build_Wb()
        self.W_b_prime.weight.data = self._build_Wb_prime()
        self.W_h.weight.data = self._build_Wh()
        self.B_h.weight.data = self._build_Bh()

    def _get_active_neuron_set(self, active_circuit: int) -> tuple[int, ...]:
        """Get the neuron set used by a circuit."""
        return tuple(sorted([self.g[(active_circuit, i)] for i in range(self.d)]))

    @override
    def to(self, *args: Any, **kwargs: Any) -> Self:
        self = super().to(*args, **kwargs)
        # Move circuit weights/biases to same device
        device = self.W_g.weight.device
        for t in self._circuit_weights:
            self._circuit_weights[t] = self._circuit_weights[t].to(device)
            self._circuit_biases[t] = self._circuit_biases[t].to(device)
        return self

    @override
    def forward(
        self,
        x_block: Float[Tensor, "... n_features"],
        active_circuits: Int[Tensor, "..."] | None = None,
        **_: Any,
    ) -> Float[Tensor, "... n_features"]:
        """Forward pass for a batch of inputs with their active circuits.

        Args:
            x_block: Input in block coordinates (batch, D)
            active_circuits: Which circuit to activate for each batch element (batch,)
                If None, assumes circuit info is encoded in the input somehow.

        Returns:
            z_block: Output in block coordinates (batch, D)
        """
        assert active_circuits is not None, "active_circuits must be provided"

        batch_size = x_block.shape[0]
        device = x_block.device
        num_blocks = self.num_blocks

        # Build v (one-hot for active block) and u (indicator for active neuron set)
        v = torch.zeros(batch_size, num_blocks, device=device)
        u = torch.zeros(batch_size, num_blocks, device=device)

        for b in range(batch_size):
            circuit = int(active_circuits[b].item())
            active_block = self.f[circuit]
            v[b, active_block] = 1.0

            active_neuron_set = self._get_active_neuron_set(circuit)
            for block_idx in range(num_blocks):
                if self.k[block_idx] == active_neuron_set:
                    u[b, block_idx] = 1.0

        # y = ReLU(W_b' @ u + W_b @ v + W_g @ x)
        term1 = self.W_b_prime(u)  # (batch, D)
        term2 = self.W_b(v)  # (batch, D)
        term3 = self.W_g(x_block)  # (batch, D)

        y = F.relu(term1 + term2 + term3)

        # z = ReLU(W_h @ y + B_h @ v)
        term4 = self.W_h(y)  # (batch, D)
        term5 = self.B_h(v)  # (batch, D)

        z_block = F.relu(term4 + term5)

        return z_block

    def forward_single(
        self,
        x_block: Tensor,
        active_circuit: int,
    ) -> Tensor:
        """Forward pass for a single input (useful for testing)."""
        x_batch = x_block.unsqueeze(0)
        circuits = torch.tensor([active_circuit], device=x_block.device)
        return self.forward(x_batch, circuits).squeeze(0)

    def compute_direct(
        self,
        x_input: Tensor,
        circuit_id: int,
    ) -> Tensor:
        """Compute circuit output directly (for verification).

        This bypasses the network and computes: ReLU(W^t @ x + b^t)
        """
        W = self._circuit_weights[circuit_id].to(x_input.device)
        b = self._circuit_biases[circuit_id].to(x_input.device)
        return F.relu(W @ x_input + b)

    def verify_circuit(self, circuit_id: int, x_input: Tensor | None = None) -> float:
        """Verify that network output matches direct circuit computation.

        Returns the L2 error between network and direct computation.
        """
        device = self.W_g.weight.device

        x_input = torch.randn(self.d, device=device) if x_input is None else x_input.to(device)

        # Direct computation
        expected = self.compute_direct(x_input, circuit_id)

        # Network computation
        x_block = torch.zeros(self.D, device=device)
        block_start = self.d * self.f[circuit_id]
        x_block[block_start : block_start + self.d] = x_input

        output_block = self.forward_single(x_block, circuit_id)
        actual = output_block[block_start : block_start + self.d]

        error = torch.norm(expected - actual).item()
        return error

    def verify_all_circuits(self) -> float:
        """Verify all circuits and return max error."""
        max_error = 0.0
        for circuit_id in range(self.T):
            error = self.verify_circuit(circuit_id)
            max_error = max(max_error, error)
            assert error < 1e-6, f"Circuit {circuit_id} failed with error {error}"
        return max_error

    @classmethod
    @override
    def from_run_info(cls, run_info: RunInfo[BSSTrainConfig]) -> "BSSModel":
        """Load a model from a run info object."""
        bss_model = cls(config=run_info.config.bss_model_config)
        state_dict = torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        bss_model.load_state_dict(state_dict)
        return bss_model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "BSSModel":
        """Fetch a pretrained model from wandb or a local path."""
        run_info = BSSTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)

    @override
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Override to include circuit weights/biases."""
        state = super().state_dict(*args, **kwargs)
        # Store circuit weights and biases
        state["_circuit_weights"] = {k: v.cpu() for k, v in self._circuit_weights.items()}
        state["_circuit_biases"] = {k: v.cpu() for k, v in self._circuit_biases.items()}
        return state

    @override
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Override to load circuit weights/biases."""
        # Convert to mutable dict and extract circuit data
        state_dict_mutable = dict(state_dict)
        circuit_weights = state_dict_mutable.pop("_circuit_weights", None)
        circuit_biases = state_dict_mutable.pop("_circuit_biases", None)

        result = super().load_state_dict(state_dict_mutable, strict=strict, assign=assign)

        if circuit_weights is not None:
            self._circuit_weights = circuit_weights
        if circuit_biases is not None:
            self._circuit_biases = circuit_biases

        return result
