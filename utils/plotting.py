import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


def _horizon_label(pred_len):
    if pred_len == 1:
        return "Next-Hour"
    return f"{pred_len}h Day-Ahead"


def plot_forecast_window(dates, actuals, preds, save_path, tag="Model",
                         n_points=336, color='steelblue'):
    n = min(n_points, len(dates))
    horizon = actuals.shape[1] if actuals.ndim > 1 else 1
    a = actuals[:n, -1] if actuals.ndim > 1 else actuals[:n]
    p = preds[:n, -1] if preds.ndim > 1 else preds[:n]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates[:n], a, color='black', label='Actual', linewidth=1.0)
    ax.plot(dates[:n], p, color=color, label=tag, linewidth=1.0, linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand (MW)')
    ax.set_title(f'{tag} — {_horizon_label(horizon)} Forecast ({n}-Hour Window)')
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_residual_distribution(actuals, preds, save_path, tag="Model", color='steelblue'):
    residuals = actuals.flatten() - preds.flatten()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=80, color=color, edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.0)
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    ax.axvline(mean_r, color='orange', linestyle=':', linewidth=1.0)
    ax.set_title(f'{tag} — Residual Distribution (mean={mean_r:.1f}, std={std_r:.1f} MW)')
    ax.set_xlabel('Residual (MW)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_monthly_grid(dates, actuals, preds, save_path, tag="Model"):
    a = actuals[:, -1] if actuals.ndim > 1 else actuals
    p = preds[:, -1] if preds.ndim > 1 else preds

    df = pd.DataFrame({
        'date': dates,
        'actual': a,
        'predicted': p,
    })
    df['ym'] = df['date'].dt.to_period('M')
    months = df['ym'].unique()[-12:]

    n_cols = 3
    n_rows = (len(months) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = axes.flatten()

    for i, m in enumerate(months):
        d = df[df['ym'] == m]
        axes[i].plot(d['date'], d['actual'], 'b-', linewidth=1.0, alpha=0.7, label='Actual')
        axes[i].plot(d['date'], d['predicted'], 'r--', linewidth=1.0, alpha=0.8, label='Predicted')
        axes[i].set_title(str(m), fontsize=10)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(fontsize='small')

    for j in range(len(months), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'{tag} — Monthly Forecast Comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_training_curves(train_losses, val_losses, save_path, tag="Model"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=1.2)
    ax.plot(epochs, val_losses, label='Val Loss', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{tag} — Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_scatter(actuals, preds, save_path, tag="Model"):
    a = actuals.flatten()
    p = preds.flatten()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, p, s=2, alpha=0.3, color='steelblue')
    lims = [min(a.min(), p.min()), max(a.max(), p.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.0, label='Perfect')
    ax.set_xlabel('Actual (MW)')
    ax.set_ylabel('Predicted (MW)')
    ax.set_title(f'{tag} — Actual vs Predicted')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_hourly_error(hourly_metrics, save_path, tag="Model"):
    if len(hourly_metrics) <= 1:
        return None

    hours = [m['hour'] for m in hourly_metrics]
    rmses = [m['rmse'] for m in hourly_metrics]
    maes = [m['mae'] for m in hourly_metrics]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hours, rmses, 'o-', label='RMSE', markersize=4)
    ax.plot(hours, maes, 's--', label='MAE', markersize=4)
    ax.set_xlabel('Forecast Horizon (hours ahead)')
    ax.set_ylabel('Error (MW)')
    ax.set_title(f'{tag} — Error by Forecast Horizon')
    ax.set_xticks(hours)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_comparison_bar(results_dict, save_path, metrics_to_plot=None):
    if metrics_to_plot is None:
        metrics_to_plot = ['rmse', 'mae', 'mape']

    models = list(results_dict.keys())
    n_metrics = len(metrics_to_plot)
    x = np.arange(len(models))
    w = 0.8 / n_metrics

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics_to_plot):
        vals = [results_dict[m][metric] for m in models]
        offset = (i - n_metrics / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=metric.upper(), color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Error')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path