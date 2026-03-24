import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from models.quantum_adqrl import AdaptiveSeriesDecomp, QuantumLayer


class SpectralGate(nn.Module):
    """
    Learnable frequency-domain splitter.
    Applies real-valued FFT to the seasonal component, then uses a learnable
    soft mask to separate high-freq and low-freq bands.
    The gate is parameterized so it can learn the optimal cutoff during training
    rather than using a fixed frequency threshold.
    """
    def __init__(self, seq_len):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.gate_logits = nn.Parameter(torch.zeros(n_freq))

    def forward(self, x):
        # x: (B, L, C) — seasonal component
        x_perm = x.permute(0, 2, 1)  # (B, C, L)
        X_freq = torch.fft.rfft(x_perm, dim=-1)

        gate = torch.sigmoid(self.gate_logits)  # values in (0,1), one per freq bin
        # gate near 1 = low-freq pass, gate near 0 = high-freq pass

        X_low = X_freq * gate.unsqueeze(0).unsqueeze(0)
        X_high = X_freq * (1 - gate).unsqueeze(0).unsqueeze(0)

        low_freq = torch.fft.irfft(X_low, n=x.shape[1], dim=-1).permute(0, 2, 1)
        high_freq = torch.fft.irfft(X_high, n=x.shape[1], dim=-1).permute(0, 2, 1)

        return low_freq, high_freq


class QuantumBranch(nn.Module):
    """One compress-VQC-project branch."""
    def __init__(self, n_features, seq_len, pred_len, n_qubits, n_qlayers, context_size):
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        self.n_qubits = n_qubits

        self.time_adapter = nn.Sequential(
            nn.Linear(seq_len, context_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.compressor = nn.Sequential(
            nn.Linear(n_features * context_size, n_qubits),
            nn.BatchNorm1d(n_qubits),
            nn.Tanh(),
        )
        self.vqc = QuantumLayer(n_qubits, n_qlayers)
        self.head = nn.Linear(n_qubits, pred_len * n_features)

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        ctx = self.time_adapter(x.permute(0, 2, 1))  # (B, C, ctx_size)
        q_in = self.compressor(ctx.reshape(B, -1)) * np.pi
        q_out = self.vqc(q_in)
        out = self.head(q_out).reshape(B, self.pred_len, self.n_features)
        return out


class MSQD(nn.Module):
    """
    Multi-Scale Quantum Decomposition (MSQD).

    Key idea: electricity demand contains overlapping periodicities (daily, weekly).
    Instead of collapsing the full seasonal component into a single VQC bottleneck,
    we split the seasonal signal into frequency bands using a learnable spectral gate,
    then process each band with its own dedicated VQC. This allows each quantum
    circuit to specialize in a particular frequency range.

    Architecture:
        Input -> Learnable Decomposition -> Trend + Seasonal
        Trend  -> Linear projection -> trend_out
        Seasonal -> SpectralGate -> low_freq_seasonal, high_freq_seasonal
        low_freq_seasonal  -> VQC_low  -> low_out
        high_freq_seasonal -> VQC_high -> high_out
        Output = alpha * trend_out + beta * low_out + gamma * high_out
        (alpha, beta, gamma are learnable, softmax-normalized)
    """
    def __init__(self, n_features, pred_len, seq_len=168, kernel_size=25,
                 n_qubits=4, n_qlayers=2, context_size=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features

        self.decomp = AdaptiveSeriesDecomp(kernel_size, n_features)
        self.trend_linear = nn.Linear(seq_len, pred_len)

        self.spectral_gate = SpectralGate(seq_len)

        self.low_branch = QuantumBranch(
            n_features, seq_len, pred_len, n_qubits, n_qlayers, context_size
        )
        self.high_branch = QuantumBranch(
            n_features, seq_len, pred_len, n_qubits, n_qlayers, context_size
        )

        # 3-way learnable fusion via softmax over logits
        self.fusion_logits = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, x):
        B = x.shape[0]
        seasonal, trend = self.decomp(x)

        trend_out = self.trend_linear(trend.permute(0, 2, 1)).permute(0, 2, 1)

        low_seas, high_seas = self.spectral_gate(seasonal)
        low_out = self.low_branch(low_seas)
        high_out = self.high_branch(high_seas)

        weights = torch.softmax(self.fusion_logits, dim=0)
        out = weights[0] * trend_out + weights[1] * low_out + weights[2] * high_out
        return out
