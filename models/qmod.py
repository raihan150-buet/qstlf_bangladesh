import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp, QuantumLayer


class QMod(nn.Module):
    """
    Quantum-Modulated DLinear (QMod-DLinear).

    Problem with ADQRL: the seasonal branch compresses everything into
    4 qubits, runs VQC, then a single Linear(4, pred_len * n_features)
    must reconstruct the full forecast from just 4 numbers. The linear
    projection carries too much burden.

    QMod keeps a classical seasonal linear (like DLinear) to produce
    a base seasonal prediction, then uses the VQC output to generate
    per-horizon modulation factors that scale the classical prediction.
    The VQC's 4-dimensional output is expanded to pred_len modulation
    weights via a small network, and applied element-wise.

    This way:
    - Classical seasonal linear handles the temporal mapping (168 -> 24)
    - VQC learns nonlinear modulation patterns (which hours to amplify/dampen)
    - The VQC has a much easier job: learn a 24-dim scaling vector, not
      reconstruct the entire forecast from scratch

    Architecture:
        Input -> Learnable Decomposition -> Trend + Seasonal
        Trend    -> Linear -> trend_out
        Seasonal -> Linear -> seasonal_base  (classical, like DLinear)
        Seasonal -> compress -> VQC -> expand to pred_len modulation weights
        seasonal_out = seasonal_base * (1 + modulation)
        Output = sigmoid(alpha) * seasonal_out + (1 - sigmoid(alpha)) * trend_out

    Single VQC call. Same training speed as ADQRL.
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2, context_size=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_qubits = n_qubits

        self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)

        # quantum modulation branch
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

        # expand VQC output to per-hour modulation factors
        self.modulation_head = nn.Sequential(
            nn.Linear(n_qubits, pred_len),
            nn.Tanh(),
        )

        self.fusion_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # classical seasonal base prediction
        seasonal_base = self.seasonal_linear(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        # quantum modulation
        ctx = self.time_adapter(seasonal.permute(0, 2, 1))
        q_in = self.feature_compressor(ctx.reshape(B, -1)) * np.pi
        q_out = self.vqc(q_in)
        modulation = self.modulation_head(q_out)  # (B, pred_len), values in [-1, 1]

        # apply modulation per-hour, broadcast across features
        # (B, T, C) * (1 + (B, T, 1))
        seasonal_out = seasonal_base * (1.0 + modulation.unsqueeze(-1))

        alpha = torch.sigmoid(self.fusion_alpha)
        return alpha * seasonal_out + (1 - alpha) * trend_out