import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp, QuantumLayer


class QRC(nn.Module):
    """
    Quantum Residual Correction DLinear (QRC-DLinear).

    The idea: classical linear layers are good at capturing trend and
    periodic patterns, but they miss nonlinear interactions. Instead of
    routing the seasonal component through a quantum bottleneck (like ADQRL),
    let the classical model make its best prediction first, then use the
    VQC to learn a correction on the prediction residual pattern.

    The VQC sees a compressed summary of the classical prediction
    alongside the original input statistics, and outputs a small
    additive correction. This is closer to boosting/residual learning
    than to feature extraction.

    Architecture:
        Input -> Learnable Decomposition -> Trend + Seasonal
        Trend    -> Linear -> trend_pred
        Seasonal -> Linear -> seasonal_pred
        base_pred = trend_pred + seasonal_pred   (this is standard DLinear)

        Correction branch:
        [base_pred stats, input stats] -> compress to n_qubits -> VQC -> correction
        final = base_pred + correction

    Single VQC call, same training speed as ADQRL.
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_qubits = n_qubits

        self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)
        self.trend_linear = nn.Linear(seq_len, pred_len)

        # correction branch input: summary stats from base prediction + input
        # per-feature: mean and std of base_pred (2) + mean and std of input (2) = 4 per feature
        # we compress n_features * 4 stats into n_qubits
        summary_dim = n_features * 4
        self.correction_compressor = nn.Sequential(
            nn.Linear(summary_dim, n_qubits),
            nn.BatchNorm1d(n_qubits),
            nn.Tanh(),
        )

        self.vqc = QuantumLayer(n_qubits, n_qlayers)

        self.correction_head = nn.Sequential(
            nn.Linear(n_qubits, pred_len * n_features),
        )

        self.correction_scale = nn.Parameter(torch.tensor([0.1]))

    def _compute_summary(self, x_input, base_pred):
        # x_input: (B, L, C), base_pred: (B, T, C)
        pred_mean = base_pred.mean(dim=1)    # (B, C)
        pred_std = base_pred.std(dim=1) + 1e-6
        inp_mean = x_input.mean(dim=1)
        inp_std = x_input.std(dim=1) + 1e-6
        return torch.cat([pred_mean, pred_std, inp_mean, inp_std], dim=1)  # (B, C*4)

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        seasonal_pred = self.seasonal_linear(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend_pred = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        base_pred = seasonal_pred + trend_pred

        summary = self._compute_summary(x, base_pred)
        q_in = self.correction_compressor(summary) * np.pi
        q_out = self.vqc(q_in)
        correction = self.correction_head(q_out).reshape(B, self.pred_len, self.n_features)

        scale = torch.sigmoid(self.correction_scale)
        return base_pred + scale * correction