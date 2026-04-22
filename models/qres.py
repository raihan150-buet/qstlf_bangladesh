import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp


class QuantumReservoir(nn.Module):
    """
    Fixed (non-trainable) quantum circuit used as a nonlinear feature expander.

    Maps n_qubits classical inputs to 3*n_qubits expectation values by measuring
    PauliX, PauliY, PauliZ on each qubit. The circuit parameters are randomly
    initialized once and frozen — only the classical layers around it are trained.

    This is the quantum reservoir computing paradigm: the quantum system provides
    a rich, high-dimensional nonlinear mapping that a simple linear readout can
    learn to exploit. No quantum gradients needed, so it's much faster than VQC
    training.
    """
    def __init__(self, n_qubits, n_layers, seed=42):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_outputs = n_qubits * 3

        dev = qml.device("default.qubit", wires=n_qubits)

        # fixed random reservoir weights
        rng = np.random.RandomState(seed)
        fixed_weights = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        self.fixed_weights = torch.tensor(fixed_weights, dtype=torch.float32)

        @qml.qnode(dev, interface='torch', diff_method=None)
        def reservoir_circuit(inputs, res_weights):
            # data encoding with re-uploading through reservoir layers
            for l in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
                # fixed entangling structure
                for i in range(n_qubits):
                    qml.RX(res_weights[l, i, 0], wires=i)
                    qml.RY(res_weights[l, i, 1], wires=i)
                    qml.RZ(res_weights[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

            # measure all three Pauli observables on each qubit
            observables = []
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliX(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliY(i)))
            for i in range(n_qubits):
                observables.append(qml.expval(qml.PauliZ(i)))
            return observables

        self.circuit = reservoir_circuit

    def forward(self, x):
        # x: (B, n_qubits)
        batch_results = []
        for i in range(x.shape[0]):
            result = self.circuit(x[i], self.fixed_weights)
            batch_results.append(torch.stack(result))
        return torch.stack(batch_results)  # (B, n_qubits * 3)


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