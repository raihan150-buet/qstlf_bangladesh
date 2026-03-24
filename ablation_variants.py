import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp, QuantumLayer
from models.classical_dlinear import MovingAvg


class AblationADQRL(nn.Module):
    """
    Ablation variants of ADQRL:
      'full'         — complete ADQRL (VQC seasonal + learnable decomp + learnable fusion)
      'no_quantum'   — classical MLP replaces VQC, same dimensions
      'trend_only'   — no seasonal branch at all
      'fixed_decomp' — fixed moving average instead of learnable Conv1d
      'no_fusion'    — hard sum instead of learnable alpha
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2, context_size=32, mode='full'):
        super().__init__()
        self.mode = mode
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_qubits = n_qubits
        self.n_features = n_features

        if mode == 'fixed_decomp':
            self.decomp = None
            self._moving_avg = MovingAvg(kernel_size)
        else:
            self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)

        self.trend_linear = nn.Linear(seq_len, pred_len)

        if mode != 'trend_only':
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

            if mode in ('full', 'fixed_decomp', 'no_fusion'):
                self.seasonal_block = QuantumLayer(n_qubits, n_qlayers)
            elif mode == 'no_quantum':
                self.seasonal_block = nn.Sequential(
                    nn.Linear(n_qubits, n_qubits * 2),
                    nn.Tanh(),
                    nn.Linear(n_qubits * 2, n_qubits),
                    nn.Tanh(),
                )

            self.seasonal_head = nn.Linear(n_qubits, pred_len * n_features)

            if mode != 'no_fusion':
                self.fusion_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B = x.shape[0]

        if self.decomp is not None:
            seasonal, trend = self.decomp(x)
        else:
            trend = self._moving_avg(x)
            seasonal = x - trend

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)

        if self.mode == 'trend_only':
            return trend_out

        ctx = self.time_adapter(seasonal.permute(0, 2, 1))
        q_in = self.feature_compressor(ctx.reshape(B, -1)) * np.pi
        q_out = self.seasonal_block(q_in)
        s_out = self.seasonal_head(q_out).reshape(B, self.pred_len, self.n_features)

        if self.mode == 'no_fusion':
            return s_out + trend_out

        alpha = torch.sigmoid(self.fusion_alpha)
        return alpha * s_out + (1 - alpha) * trend_out
