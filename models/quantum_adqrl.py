import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


class AdaptiveSeriesDecomp(nn.Module):
    """Learnable 1D convolution as low-pass filter for trend extraction."""
    def __init__(self, kernel_size, n_features):
        super().__init__()
        self.conv = nn.Conv1d(
            n_features, n_features,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=n_features,
            bias=False
        )
        nn.init.constant_(self.conv.weight, 1.0 / kernel_size)

    def forward(self, x):
        # x: (B, L, C) -> permute to (B, C, L) for conv
        trend = self.conv(x.permute(0, 2, 1))
        if trend.shape[2] != x.shape[1]:
            trend = nn.functional.interpolate(trend, size=x.shape[1])
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_qlayers):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.q_layer = qml.qnn.TorchLayer(circuit, {"weights": (n_qlayers, n_qubits, 3)})
        self.output_proj = nn.Linear(n_qubits, n_qubits)

    def forward(self, x):
        return self.output_proj(self.q_layer(x)) + x


class ADQRL(nn.Module):
    """
    Adaptive Decomposition with Quantum Residual Learning.
    - Learnable Conv1d decomposition
    - Trend branch: linear layer
    - Seasonal branch: compress -> VQC -> project back
    - Learnable fusion weight alpha
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2, context_size=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_qubits = n_qubits
        self.n_features = n_features

        self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)
        self.trend_linear = nn.Linear(seq_len, pred_len)

        self.time_adapter = nn.Sequential(
            nn.Linear(seq_len, context_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.feature_compressor = nn.Sequential(
            nn.Linear(n_features * context_size, n_qubits),
            nn.BatchNorm1d(n_qubits),
            nn.Tanh(),
        )
        self.vqc = QuantumLayer(n_qubits, n_qlayers)
        self.seasonal_head = nn.Linear(n_qubits, pred_len * n_features)
        self.fusion_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)

        ctx = self.time_adapter(seasonal.permute(0, 2, 1))
        q_in = self.feature_compressor(ctx.reshape(B, -1)) * np.pi
        q_out = self.vqc(q_in)
        s_out = self.seasonal_head(q_out).reshape(B, self.pred_len, self.n_features)

        alpha = torch.sigmoid(self.fusion_alpha)
        return alpha * s_out + (1 - alpha) * trend_out
