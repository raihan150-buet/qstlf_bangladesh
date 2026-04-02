"""
SOTA Classical Model Benchmarks for STLF
==========================================

Benchmarks proposed quantum-classical models against established SOTA
time series forecasting methods using the same data pipeline, splits,
normalization, and metrics for fair comparison.

Models:
  - NLinear (Zeng et al., 2022)
  - PatchTST (Nie et al., ICLR 2023)
  - iTransformer (Liu et al., ICLR 2024)
  - TimesNet (Wu et al., ICLR 2023)
  - Vanilla LSTM
  - Persistence baseline (repeat previous day)

Fairness guarantees:
  - Same chronological train/val/test split (80/10/10)
  - Same MinMaxScaler fitted on training data only
  - Same sliding window (168h input, 24h output)
  - Same seed (42)
  - Same evaluation metrics computed on inverse-transformed predictions
  - All models trained with Adam, MSE loss, early stopping

Usage:
  1. Git clone your repo into Colab
  2. Upload selected_features.xlsx
  3. Run: python run_sota_benchmarks.py
  4. Or run individual: python run_sota_benchmarks.py --model patchtst
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import time
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import wandb

from configs.config import *
from utils.reproducibility import set_seed
from utils.data import load_data, make_loaders
from utils.metrics import compute_metrics, compute_hourly_metrics
from utils.plotting import (
    plot_forecast_window, plot_residual_distribution,
    plot_training_curves, plot_scatter, plot_comparison_bar,
)

SOTA_DIR = os.path.join(BASE_DIR, "sota_benchmarks")
os.makedirs(os.path.join(SOTA_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(SOTA_DIR, "checkpoints"), exist_ok=True)

ALL_MODELS = ['persistence', 'lstm', 'nlinear', 'patchtst', 'itransformer', 'timesnet']

# ============================================================================
# Model Definitions
# ============================================================================

class PersistenceModel:
    """Repeats the last 24h of demand as the forecast. No training needed."""
    def predict(self, X):
        # X: (B, 168, C), demand is last column
        # last 24 hours of demand
        return X[:, -FORECAST_HORIZON:, -1]


class NLinear(nn.Module):
    """NLinear from Zeng et al. 2022 — subtracts last value, linear, add back."""
    def __init__(self, n_features, pred_len, seq_len):
        super().__init__()
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        last_val = x[:, -1:, :]  # (B, 1, C)
        x_norm = x - last_val
        out = self.linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        out = out + last_val
        return out


class VanillaLSTM(nn.Module):
    def __init__(self, n_features, pred_len, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, pred_len)
        self.n_features = n_features

    def forward(self, x):
        # x: (B, L, C)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last)  # (B, pred_len) — demand only
        return pred


class PatchTST(nn.Module):
    """
    Simplified PatchTST (Nie et al., ICLR 2023).
    Patches the time series, applies transformer encoder, projects to forecast.
    Channel-independent.
    """
    def __init__(self, n_features, pred_len, seq_len, patch_len=16, stride=8,
                 d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C) — process each channel independently, average
        B, L, C = x.shape
        # unfold into patches: (B, C, n_patches, patch_len)
        x_p = x.permute(0, 2, 1)  # (B, C, L)
        patches = x_p.unfold(2, self.patch_len, self.stride)  # (B, C, n_patches, patch_len)

        # reshape to process all channels together
        patches = patches.reshape(B * C, self.n_patches, self.patch_len)
        tokens = self.patch_proj(patches) + self.pos_embed
        tokens = self.dropout(tokens)

        encoded = self.encoder(tokens)  # (B*C, n_patches, d_model)
        flat = encoded.reshape(B * C, -1)
        out = self.head(flat)  # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len)

        return out.permute(0, 2, 1)  # (B, pred_len, C)


class iTransformer(nn.Module):
    """
    Simplified iTransformer (Liu et al., ICLR 2024).
    Treats each variate as a token (inverted view).
    Self-attention across variates, not across time.
    """
    def __init__(self, n_features, pred_len, seq_len, d_model=64,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.pred_len = pred_len

        self.embed = nn.Linear(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (B, L, C) -> treat C as token dimension
        x_t = x.permute(0, 2, 1)  # (B, C, L) — each variate is a token
        tokens = self.embed(x_t)    # (B, C, d_model)

        encoded = self.encoder(tokens)  # (B, C, d_model)
        out = self.head(encoded)        # (B, C, pred_len)

        return out.permute(0, 2, 1)  # (B, pred_len, C)


class TimesBlock(nn.Module):
    """Single TimesNet block — FFT to find periods, 2D conv on reshaped data."""
    def __init__(self, seq_len, d_model, d_ff, n_kernels=3, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=(n_kernels, 1), padding=(n_kernels // 2, 0)),
            nn.GELU(),
            nn.Conv2d(d_ff, d_model, kernel_size=(n_kernels, 1), padding=(n_kernels // 2, 0)),
        )

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.shape
        # find top-k periods via FFT
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(dim=-1)[:, 1:]  # skip DC
        _, top_idx = amp.topk(self.top_k, dim=1)
        periods = (L / (top_idx.float() + 1)).int().clamp(min=2)

        res = torch.zeros_like(x)
        for k in range(self.top_k):
            p = periods[:, k].max().item()
            n_seg = (L + p - 1) // p
            pad_len = n_seg * p - L
            x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            x_2d = x_pad.reshape(B, n_seg, p, D).permute(0, 3, 1, 2)  # (B, D, n_seg, p)
            out_2d = self.conv(x_2d)
            out = out_2d.permute(0, 2, 3, 1).reshape(B, -1, D)[:, :L, :]
            res = res + out

        return res / self.top_k + x


class TimesNet(nn.Module):
    """Simplified TimesNet (Wu et al., ICLR 2023)."""
    def __init__(self, n_features, pred_len, seq_len, d_model=32,
                 d_ff=64, n_layers=2, top_k=3, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.embed = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, top_k=top_k) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(seq_len * d_model, pred_len * n_features)
        self.n_features = n_features
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        h = self.embed(x)  # (B, L, d_model)
        for block in self.blocks:
            h = self.dropout(block(h))
        h = self.norm(h)
        out = self.head(h.reshape(B, -1))
        return out.reshape(B, self.pred_len, self.n_features)


# ============================================================================
# Training & Evaluation
# ============================================================================

def extract_target(out):
    if out.dim() == 3:
        return out[:, :, -1]
    return out


def train_model(model, train_loader, val_loader, model_name, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val = float('inf')
    best_state = None
    patience_ctr = 0
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        tl = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(extract_target(model(bx)), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        avg_t = tl / len(train_loader)

        model.eval()
        vl = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                vl += criterion(extract_target(model(bx)), by).item()
        avg_v = vl / len(val_loader)

        train_losses.append(avg_t)
        val_losses.append(avg_v)
        scheduler.step(avg_v)
        elapsed = time.time() - t0

        imp = ""
        if avg_v < best_val:
            best_val = avg_v
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            imp = " *"
        else:
            patience_ctr += 1

        print(f"    Epoch {epoch+1:03d}/{EPOCHS}  train={avg_t:.6f}  val={avg_v:.6f}  ({elapsed:.1f}s){imp}")

        if patience_ctr >= 15:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y, device, is_persistence=False):
    if is_persistence:
        preds_list, actuals_list = [], []
        for bx, by in test_loader:
            p = model.predict(bx.numpy())
            preds_list.append(p)
            actuals_list.append(by.numpy())
        preds = np.vstack(preds_list)
        actuals = np.vstack(actuals_list)
    else:
        model.eval()
        preds_list, actuals_list = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                out = extract_target(model(bx))
                preds_list.append(out.cpu().numpy())
                actuals_list.append(by.numpy())
        preds = np.vstack(preds_list)
        actuals = np.vstack(actuals_list)

    preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    actuals_inv = scaler_y.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)

    metrics = compute_metrics(actuals_inv.flatten(), preds_inv.flatten())
    return metrics, preds_inv, actuals_inv


def build_model(name, n_features):
    if name == 'persistence':
        return PersistenceModel()
    elif name == 'nlinear':
        return NLinear(n_features, FORECAST_HORIZON, SEQ_LENGTH)
    elif name == 'lstm':
        return VanillaLSTM(n_features, FORECAST_HORIZON, hidden_dim=64, n_layers=2, dropout=0.1)
    elif name == 'patchtst':
        return PatchTST(n_features, FORECAST_HORIZON, SEQ_LENGTH,
                        patch_len=16, stride=8, d_model=64, n_heads=4, n_layers=2)
    elif name == 'itransformer':
        return iTransformer(n_features, FORECAST_HORIZON, SEQ_LENGTH,
                            d_model=64, n_heads=4, n_layers=2)
    elif name == 'timesnet':
        return TimesNet(n_features, FORECAST_HORIZON, SEQ_LENGTH,
                        d_model=32, d_ff=64, n_layers=2, top_k=3)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="*", default=None, choices=ALL_MODELS,
                        help="Which models to benchmark. Omit to run all.")
    parser.add_argument("--report", action="store_true",
                        help="Generate comparison report from saved results.")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 60)
    print("SOTA Benchmarks")
    print("=" * 60)

    (X_train, y_train, dates_train,
     X_val, y_val, dates_val,
     X_test, y_test, dates_test,
     data_info) = load_data(DATA_PATH, SEQ_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO)

    # for persistence model, keep test data on CPU
    test_loader_cpu = DataLoader(TensorDataset(X_test.cpu(), y_test.cpu()),
                                 batch_size=BATCH_SIZE, shuffle=False)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE
    )

    n_feat = data_info["n_features"]
    print(f"Data: train={data_info['n_train']}, val={data_info['n_val']}, "
          f"test={data_info['n_test']}, features={n_feat}")

    if args.report:
        # just load saved results and generate comparison
        all_results = {}
        results_path = os.path.join(SOTA_DIR, "all_sota_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results = json.load(f)
        # also load our models
        for name, path in [("Classical DLinear", os.path.join(CLASSICAL_DIR, "test_metrics.json")),
                           ("ADQRL", os.path.join(QUANTUM_DIR, "test_metrics.json")),
                           ("MSQD", os.path.join(BASE_DIR, "msqd", "test_metrics.json"))]:
            if os.path.exists(path):
                with open(path) as f:
                    all_results[name] = json.load(f)
        generate_report(all_results)
        return

    models_to_run = args.model if args.model else ALL_MODELS
    all_results = {}

    # load existing results
    results_path = os.path.join(SOTA_DIR, "all_sota_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    for model_name in models_to_run:
        set_seed(SEED)
        print(f"\n{'='*50}")
        print(f"  {model_name.upper()}")
        print(f"{'='*50}")

        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=f"sota_{model_name}", group="sota_benchmarks",
            config={"model": model_name, "seq_length": SEQ_LENGTH,
                    "forecast_horizon": FORECAST_HORIZON, "seed": SEED},
            tags=["sota", "benchmark", model_name], reinit=True,
        )

        model = build_model(model_name, n_feat)

        if model_name == 'persistence':
            metrics, preds_inv, actuals_inv = evaluate_model(
                model, test_loader_cpu, data_info["scaler_y"], device, is_persistence=True
            )
            train_losses, val_losses = [], []
        else:
            total_p = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_p}")
            run.config.update({"total_params": total_p})

            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, model_name, device
            )
            metrics, preds_inv, actuals_inv = evaluate_model(
                model, test_loader, data_info["scaler_y"], device
            )

            ckpt_path = os.path.join(SOTA_DIR, "checkpoints", f"{model_name}_best.pth")
            torch.save({"model_state": model.state_dict(), "config": CONFIG}, ckpt_path)

        print(f"\n  Results:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        run.log({f"test/{k}": v for k, v in metrics.items()})

        hourly = compute_hourly_metrics(actuals_inv, preds_inv)
        run.log({"test/hourly_metrics": wandb.Table(dataframe=pd.DataFrame(hourly))})

        fig_dir = os.path.join(SOTA_DIR, "figures")

        p = plot_forecast_window(dates_test, actuals_inv, preds_inv,
                                 os.path.join(fig_dir, f"{model_name}_forecast.png"),
                                 tag=model_name.upper())
        run.log({"figures/forecast": wandb.Image(p)})

        if train_losses:
            p = plot_training_curves(train_losses, val_losses,
                                     os.path.join(fig_dir, f"{model_name}_curves.png"),
                                     tag=model_name.upper())
            run.log({"figures/curves": wandb.Image(p)})

        run.finish()
        all_results[model_name.upper()] = metrics

    # save combined results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    pd.DataFrame([{"model": k, **v} for k, v in all_results.items()]).to_csv(
        os.path.join(SOTA_DIR, "all_sota_results.csv"), index=False
    )

    # load our models' results and generate full comparison
    for name, path in [("Classical DLinear", os.path.join(CLASSICAL_DIR, "test_metrics.json")),
                       ("ADQRL", os.path.join(QUANTUM_DIR, "test_metrics.json")),
                       ("MSQD", os.path.join(BASE_DIR, "msqd", "test_metrics.json"))]:
        if os.path.exists(path):
            with open(path) as f:
                all_results[name] = json.load(f)

    generate_report(all_results)
    print("\nAll SOTA benchmarks complete.")


def generate_report(all_results):
    fig_dir = os.path.join(SOTA_DIR, "figures")

    p = plot_comparison_bar(all_results,
                            os.path.join(fig_dir, "sota_comparison_rmse_mae.png"),
                            metrics_to_plot=['rmse', 'mae'])
    print(f"  Saved: {p}")

    p = plot_comparison_bar(all_results,
                            os.path.join(fig_dir, "sota_comparison_mape.png"),
                            metrics_to_plot=['mape'])
    print(f"  Saved: {p}")

    # latex table
    models = list(all_results.keys())
    metric_keys = ['rmse', 'mae', 'mape', 'r2', 'mean_residual']
    headers = ['RMSE (MW)', 'MAE (MW)', 'MAPE (\\%)', 'R²', 'Bias (MW)']

    best = {}
    for m in metric_keys:
        vals = [all_results[k].get(m, float('inf')) for k in models]
        if m == 'r2':
            best[m] = max(vals)
        elif m == 'mean_residual':
            best[m] = min(abs(v) for v in vals)
        else:
            best[m] = min(vals)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    h_label = "Next-Hour" if FORECAST_HORIZON == 1 else f"{FORECAST_HORIZON}h Day-Ahead"
    lines.append(f"\\caption{{Comparison with SOTA Models — {h_label} Forecast}}")
    lines.append("\\label{tab:sota_comparison}")
    lines.append("\\begin{tabular}{l" + "r" * len(metric_keys) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for model in models:
        r = all_results[model]
        cells = []
        for m in metric_keys:
            v = r.get(m, float('nan'))
            if m == 'r2':
                fmt = f"{v:.4f}"
                is_best = (v == best[m])
            elif m == 'mean_residual':
                fmt = f"{v:.2f}"
                is_best = (abs(v) == best[m])
            else:
                fmt = f"{v:.2f}"
                is_best = (v == best[m])
            if is_best:
                fmt = f"\\textbf{{{fmt}}}"
            cells.append(fmt)
        lines.append(f"{model} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_path = os.path.join(SOTA_DIR, "sota_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table: {tex_path}")

    # summary wandb run
    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name="sota_summary", tags=["sota", "summary"], reinit=True,
    )
    df = pd.DataFrame([{"model": k, **v} for k, v in all_results.items()])
    run.log({"sota_comparison": wandb.Table(dataframe=df)})

    for fname in os.listdir(fig_dir):
        if fname.startswith("sota_") and fname.endswith(".png"):
            run.log({f"figures/{fname}": wandb.Image(os.path.join(fig_dir, fname))})

    art = wandb.Artifact("sota_benchmark_results", type="results")
    art.add_dir(SOTA_DIR)
    run.log_artifact(art)
    run.finish()

    print("\n  SOTA comparison report generated.")
    print(f"  Results: {os.path.join(SOTA_DIR, 'all_sota_results.csv')}")
    print(f"  LaTeX:   {tex_path}")


if __name__ == "__main__":
    main()