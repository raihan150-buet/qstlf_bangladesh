import numpy as np
import torch
import torch.nn as nn

from models.quantum_adqrl import AdaptiveSeriesDecomp, QuantumLayer


class LAQ(nn.Module):
    """
    Lag-Aware Quantum DLinear (LAQ-DLinear).

    Observation: PACF analysis on Bangladesh load data shows short-term lags
    (1-6), daily lags (23-25), and weekly lags (167-169) carry the most
    predictive signal. ADQRL ignores this structure — it compresses
    n_features * context_size values into 4 qubits, losing the lag hierarchy.

    LAQ directly extracts demand values at the critical lag positions from
    the input sequence's last timestep perspective, and feeds exactly those
    values into the VQC. The quantum circuit then models nonlinear
    interactions between the most informative lag positions — something
    a linear layer fundamentally cannot do.

    The classical DLinear branch handles the full temporal mapping as usual.
    The VQC branch specifically captures cross-lag interactions at the
    positions that matter most.

    Architecture:
        Input -> Learnable Decomposition -> Trend + Seasonal
        Trend    -> Linear -> trend_out
        Seasonal -> Linear -> seasonal_base

        Lag extraction from input (demand column):
          [x(t-1), x(t-2), ..., x(t-6),      # short-term (6 values)
           x(t-23), x(t-24), x(t-25),          # daily cycle (3 values)
           x(t-167), x(t-168)]                  # weekly cycle (2 values)
          = 11 values -> Linear(11, n_qubits) -> VQC -> modulation(pred_len)

        seasonal_out = seasonal_base * (1 + modulation)
        Output = sigmoid(alpha) * seasonal_out + (1 - sigmoid(alpha)) * trend_out

    Single VQC call. Fewer parameters than ADQRL (no time_adapter, smaller compressor).
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2, lag_positions=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_qubits = n_qubits

        if lag_positions is None:
            lag_positions = [1, 2, 3, 4, 5, 6, 23, 24, 25, 167, 168]
        # convert lag offsets (e.g. lag=1 means t-1) to sequence indices
        self.lag_offsets = [seq_len - lag for lag in lag_positions if lag <= seq_len]

        self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)

        n_lags = len(self.lag_offsets)
        self.lag_compressor = nn.Sequential(
            nn.Linear(n_lags, n_qubits),
            nn.Tanh(),
        )

        self.vqc = QuantumLayer(n_qubits, n_qlayers)

        self.modulation_head = nn.Sequential(
            nn.Linear(n_qubits, pred_len),
            nn.Tanh(),
        )

        self.fusion_alpha = nn.Parameter(torch.tensor([0.5]))

    def _extract_lags(self, x):
        # x: (B, L, C) — extract demand column (last feature) at lag positions
        demand = x[:, :, -1]  # (B, L)
        lag_vals = demand[:, self.lag_offsets]  # (B, n_lags)
        return lag_vals

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_base = self.seasonal_linear(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        lag_vals = self._extract_lags(x)
        q_in = self.lag_compressor(lag_vals) * np.pi
        q_out = self.vqc(q_in)
        modulation = self.modulation_head(q_out)

        seasonal_out = seasonal_base * (1.0 + modulation.unsqueeze(-1))

        alpha = torch.sigmoid(self.fusion_alpha)
        return alpha * seasonal_out + (1 - alpha) * trend_out