import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp


class QuantumReservoir(nn.Module):
    """
    Fixed (non-trainable) quantum circuit used as a nonlinear feature expander.

    Maps n_qubits classical inputs to 3*n_qubits expectation values by measuring
    PauliX, PauliY, PauliZ on each qubit via three separate circuits.
    The reservoir weights are randomly initialized once and frozen.
    Uses TorchLayer for efficient batch processing (same speed as ADQRL).
    """
    def __init__(self, n_qubits, n_layers, seed=42):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_outputs = n_qubits * 3

        rng = np.random.RandomState(seed)

        def make_circuit(pauli_obs):
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface='torch', diff_method='backprop')
            def circuit(inputs, weights):
                for l in range(n_layers):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
                    qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
                return [qml.expval(pauli_obs(i)) for i in range(n_qubits)]

            return circuit

        self.circuit_x = qml.qnn.TorchLayer(
            make_circuit(qml.PauliX),
            {"weights": (n_layers, n_qubits, 3)}
        )
        self.circuit_y = qml.qnn.TorchLayer(
            make_circuit(qml.PauliY),
            {"weights": (n_layers, n_qubits, 3)}
        )
        self.circuit_z = qml.qnn.TorchLayer(
            make_circuit(qml.PauliZ),
            {"weights": (n_layers, n_qubits, 3)}
        )

        # freeze all quantum parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: (B, n_qubits)
        out_x = self.circuit_x(x)
        out_y = self.circuit_y(x)
        out_z = self.circuit_z(x)
        return torch.cat([out_x, out_y, out_z], dim=-1).float()


class QRes(nn.Module):
    """
    Quantum Reservoir DLinear (QRes-DLinear).

    Inspired by quantum reservoir computing (QRC): uses a fixed random
    quantum circuit as a nonlinear feature expander rather than training
    the quantum parameters. The classical readout layers learn to map
    the reservoir's rich output to useful forecast modulations.

    Why this is different from ADQRL:
    - ADQRL trains VQC weights via backpropagation through the quantum circuit.
      This requires computing quantum gradients, which is the main CPU bottleneck.
    - QRes freezes the quantum circuit. Backpropagation only flows through
      classical layers. The quantum circuit acts as a fixed nonlinear kernel.

    Why reservoir features are richer:
    - ADQRL measures only PauliZ on each qubit -> 4 outputs
    - QRes measures PauliX, PauliY, AND PauliZ on each qubit -> 12 outputs
      These capture complementary projections of the quantum state, providing
      a richer feature set for the classical readout.

    Architecture:
        Input -> Learnable Decomposition -> Trend + Seasonal
        Trend    -> Linear -> trend_out
        Seasonal -> Linear -> seasonal_base

        Lag extraction (demand at PACF positions) -> compress to n_qubits
        -> Fixed Quantum Reservoir (no training) -> 3*n_qubits features
        -> Classical readout -> modulation(pred_len)

        seasonal_out = seasonal_base * (1 + modulation)
        Output = sigmoid(alpha) * seasonal_out + (1 - sigmoid(alpha)) * trend_out
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_reservoir_layers=3, lag_positions=None, seed=42):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_qubits = n_qubits

        if lag_positions is None:
            lag_positions = [1, 2, 3, 4, 5, 6, 23, 24, 25, 167, 168]
        self.lag_indices = [seq_len - lag for lag in lag_positions if lag <= seq_len]
        n_lags = len(self.lag_indices)

        self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)

        # compress lag values to qubit count
        self.lag_compressor = nn.Sequential(
            nn.Linear(n_lags, n_qubits),
            nn.Tanh(),
        )

        # fixed quantum reservoir
        self.reservoir = QuantumReservoir(n_qubits, n_reservoir_layers, seed=seed)
        reservoir_out_dim = n_qubits * 3

        # classical readout on reservoir features
        self.readout = nn.Sequential(
            nn.Linear(reservoir_out_dim, reservoir_out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reservoir_out_dim, pred_len),
            nn.Tanh(),
        )

        self.fusion_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_base = self.seasonal_linear(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        demand = x[:, :, -1]
        lag_vals = demand[:, self.lag_indices]
        q_in = self.lag_compressor(lag_vals) * np.pi

        reservoir_features = self.reservoir(q_in)  # (B, n_qubits*3)
        modulation = self.readout(reservoir_features)

        seasonal_out = seasonal_base * (1.0 + modulation.unsqueeze(-1))

        alpha = torch.sigmoid(self.fusion_alpha)
        return alpha * seasonal_out + (1 - alpha) * trend_out