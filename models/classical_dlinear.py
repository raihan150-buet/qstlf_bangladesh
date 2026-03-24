import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        # pad front and end to keep length after avg pooling
        front_pad = (self.kernel_size - 1) // 2
        end_pad = self.kernel_size // 2
        front = x[:, 0:1, :].repeat(1, front_pad, 1)
        end = x[:, -1:, :].repeat(1, end_pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        # AvgPool1d expects (B, C, L)
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        return trend


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class ClassicalDLinear(nn.Module):
    """
    DLinear from Zeng et al. "Are Transformers Effective for Time Series Forecasting?"
    Decomposes input into trend + seasonal via moving average,
    applies separate linear layers along the temporal axis for each component.
    Weights are shared across variates (channel-independent).
    """
    def __init__(self, n_features, pred_len, kernel_size=25, seq_len=168):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.decomposition = SeriesDecomp(kernel_size)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)
        self.trend_linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        seasonal, trend = self.decomposition(x)
        # transpose to (B, C, L) for linear along temporal axis
        seasonal_out = self.seasonal_linear(seasonal.permute(0, 2, 1))
        trend_out = self.trend_linear(trend.permute(0, 2, 1))
        # sum and transpose back to (B, T, C)
        out = (seasonal_out + trend_out).permute(0, 2, 1)
        return out
