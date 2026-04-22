import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp


class QuantumGate(nn.Module):
    """Single VQC that serves as one GRU gate with data re-uploading."""
    def __init__(self, n_qubits, n_qlayers):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            for l in range(n_qlayers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.vqc = qml.qnn.TorchLayer(circuit, {"weights": (n_qlayers, n_qubits, 3)})

    def forward(self, x):
        return self.vqc(x)


class SingleStepQGRU(nn.Module):
    """
    A single-step Quantum GRU cell inspired by QLSTM (Chen et al.).

    Classical GRU has 3 gates: reset (r), update (z), candidate (n).
    We replace each gate's linear transformation with a VQC.

    Input: compressed lag features (n_qubits dims)
    Hidden state: initialized from lag features, updated in one step.

    GRU equations with quantum gates:
        r_t = sigmoid(expand(VQC_reset(compress(concat(h, x)))))
        z_t = sigmoid(expand(VQC_update(compress(concat(h, x)))))
        n_t = tanh(expand(VQC_candidate(compress(concat(r*h, x)))))
        h_new = (1 - z_t) * n_t + z_t * h

    Only 3 VQC calls per sample. Same order as ADQRL's 1 call.
    """
    def __init__(self, input_dim, hidden_dim, n_qubits, n_qlayers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.compress_rz = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.compress_n = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.vqc_reset = QuantumGate(n_qubits, n_qlayers)
        self.vqc_update = QuantumGate(n_qubits, n_qlayers)
        self.vqc_candidate = QuantumGate(n_qubits, n_qlayers)

        self.expand = nn.Linear(n_qubits, hidden_dim)

    def forward(self, x, h):
        # x: (B, input_dim), h: (B, hidden_dim)
        xh = torch.cat([x, h], dim=1)

        r_in = torch.tanh(self.compress_rz(xh)) * np.pi
        r = torch.sigmoid(self.expand(self.vqc_reset(r_in)))

        z_in = torch.tanh(self.compress_rz(xh)) * np.pi
        z = torch.sigmoid(self.expand(self.vqc_update(z_in)))

        rh_x = torch.cat([r * h, x], dim=1)
        n_in = torch.tanh(self.compress_n(rh_x)) * np.pi
        n = torch.tanh(self.expand(self.vqc_candidate(n_in)))

        h_new = (1 - z) * n + z * h
        return h_new


class QDGRU(nn.Module):
    """
    Quantum Decomposition with Gated Recurrent Unit (QDGRU-DLinear).

    Combines DLinear's decomposition framework with a quantum-enhanced
    GRU cell inspired by the QLSTM architecture.

    Key differences from ADQRL:
    - ADQRL: compress seasonal -> 1 VQC -> project to forecast
    - QDGRU: extract lags -> init hidden state -> 1-step quantum GRU
             with 3 quantum gates -> project to forecast

    The GRU cell provides gated memory: the reset gate learns what to
    forget from the hidden state, the update gate controls how much new
    information to incorporate, and the candidate gate (via VQC) generates
    the new representation. This structured information flow is richer
    than ADQRL's single-pass VQC.

    3 VQC calls per sample. CPU training time similar to ADQRL.

    Architecture:
        Input -> Learnable Decomposition -> Trend + Seasonal
        Trend    -> Linear -> trend_out
        Seasonal -> Linear -> seasonal_base

        Lag extraction (demand at critical positions) -> hidden state init
        Compressed seasonal stats -> GRU input
        1-step Quantum GRU (3 quantum gates) -> hidden_out
        hidden_out -> modulation(pred_len)

        seasonal_out = seasonal_base * (1 + modulation)
        Output = sigmoid(alpha) * seasonal_out + (1 - sigmoid(alpha)) * trend_out
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2, lag_positions=None, context_size=32):
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

        hidden_dim = n_qubits * 2

        # lag values -> initial hidden state
        self.h0_proj = nn.Linear(n_lags, hidden_dim)

        # seasonal summary -> GRU input
        self.seasonal_compress = nn.Sequential(
            nn.Linear(n_features * context_size, hidden_dim),
            nn.Tanh(),
        )
        self.time_adapter = nn.Linear(seq_len, context_size)

        self.qgru = SingleStepQGRU(hidden_dim, hidden_dim, n_qubits, n_qlayers)

        self.modulation_head = nn.Sequential(
            nn.Linear(hidden_dim, pred_len),
            nn.Tanh(),
        )

        self.fusion_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_base = self.seasonal_linear(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        # init hidden from lag values
        demand = x[:, :, -1]
        lag_vals = demand[:, self.lag_indices]
        h0 = torch.tanh(self.h0_proj(lag_vals))

        # seasonal summary as GRU input
        ctx = self.time_adapter(seasonal.permute(0, 2, 1))
        gru_input = self.seasonal_compress(ctx.reshape(B, -1))

        # single-step quantum GRU
        h_out = self.qgru(gru_input, h0)

        modulation = self.modulation_head(h_out)
        seasonal_out = seasonal_base * (1.0 + modulation.unsqueeze(-1))

        alpha = torch.sigmoid(self.fusion_alpha)
        return alpha * seasonal_out + (1 - alpha) * trend_out