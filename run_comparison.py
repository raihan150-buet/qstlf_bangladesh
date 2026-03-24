import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from configs.config import *
from utils.plotting import plot_comparison_bar


def load_results(model_name, base_dir):
    json_path = os.path.join(base_dir, "test_metrics.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    csv_path = os.path.join(base_dir, "test_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.iloc[0].to_dict()
    return None


def generate_latex_table(results_dict, caption="Model Comparison"):
    models = list(results_dict.keys())
    metrics = ['rmse', 'mae', 'mape', 'r2', 'mean_residual', 'std_residual']
    headers = ['RMSE (MW)', 'MAE (MW)', 'MAPE (%)', 'R²', 'Bias (MW)', 'Std Res. (MW)']

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:model_comparison}")
    col_fmt = "l" + "r" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    # find best for each metric
    best_vals = {}
    for m in metrics:
        vals = [results_dict[model].get(m, float('inf')) for model in models]
        if m == 'r2':
            best_vals[m] = max(vals)
        elif m == 'mean_residual':
            best_vals[m] = min(abs(v) for v in vals)
        else:
            best_vals[m] = min(vals)

    for model in models:
        r = results_dict[model]
        cells = []
        for m in metrics:
            v = r.get(m, float('nan'))
            if m == 'r2':
                fmt = f"{v:.4f}"
                is_best = (v == best_vals[m])
            elif m == 'mean_residual':
                fmt = f"{v:.2f}"
                is_best = (abs(v) == best_vals[m])
            else:
                fmt = f"{v:.2f}"
                is_best = (v == best_vals[m])

            if is_best:
                fmt = f"\\textbf{{{fmt}}}"
            cells.append(fmt)

        lines.append(f"{model} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_improvement_table(results_dict, baseline_name="Classical DLinear"):
    if baseline_name not in results_dict:
        return ""

    baseline = results_dict[baseline_name]
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Percentage Improvement Over Classical DLinear}")
    lines.append("\\label{tab:improvement}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Model & RMSE (\\%) & MAE (\\%) & MAPE (\\%) & R² (\\%) \\\\")
    lines.append("\\midrule")

    for model, r in results_dict.items():
        if model == baseline_name:
            continue
        rmse_imp = (baseline['rmse'] - r['rmse']) / baseline['rmse'] * 100
        mae_imp = (baseline['mae'] - r['mae']) / baseline['mae'] * 100
        mape_imp = (baseline['mape'] - r['mape']) / baseline['mape'] * 100
        r2_imp = (r['r2'] - baseline['r2']) / abs(baseline['r2']) * 100 if baseline['r2'] != 0 else 0

        lines.append(f"{model} & {rmse_imp:+.2f} & {mae_imp:+.2f} & {mape_imp:+.2f} & {r2_imp:+.2f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def plot_overlay_forecast(model_dirs, model_names, dates_test_path=None):
    """Overlay predictions from multiple models on the same plot."""
    import pickle

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['steelblue', 'purple', 'darkgreen', '#C44E52']
    n_points = 336

    for i, (name, mdir) in enumerate(zip(model_names, model_dirs)):
        hourly_path = os.path.join(mdir, "hourly_metrics.csv")
        if not os.path.exists(hourly_path):
            continue

    ax.set_xlabel('Forecast Horizon (hours ahead)')
    ax.set_ylabel('RMSE (MW)')
    ax.set_title('Per-Horizon RMSE Comparison')
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    save_path = os.path.join(COMPARISON_DIR, "figures", "horizon_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_per_horizon_comparison(model_dirs, model_names):
    colors = ['steelblue', 'purple', 'darkgreen', '#C44E52']
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, mdir) in enumerate(zip(model_names, model_dirs)):
        csv_path = os.path.join(mdir, "hourly_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        ax.plot(df['hour'], df['rmse'], 'o-', color=colors[i % len(colors)],
                label=name, markersize=4)

    ax.set_xlabel('Forecast Horizon (hours ahead)')
    ax.set_ylabel('RMSE (MW)')
    ax.set_title('Per-Horizon RMSE Comparison Across Models')
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_xticks(range(1, 25))
    plt.tight_layout()

    save_path = os.path.join(COMPARISON_DIR, "figures", "horizon_rmse_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def main():
    print("=" * 60)
    print("Model Comparison & Paper Figures")
    print("=" * 60)

    model_configs = {
        "Classical DLinear": CLASSICAL_DIR,
        "ADQRL": QUANTUM_DIR,
        "MSQD": os.path.join(BASE_DIR, "msqd"),
        "QMod": os.path.join(BASE_DIR, "qmod"),
    }

    results = {}
    for name, mdir in model_configs.items():
        r = load_results(name, mdir)
        if r is not None:
            results[name] = r
            print(f"  Loaded {name}: RMSE={r['rmse']:.2f}, MAE={r['mae']:.2f}, "
                  f"MAPE={r['mape']:.2f}%, R²={r['r2']:.4f}")
        else:
            print(f"  WARNING: No results found for {name} in {mdir}")

    if len(results) < 2:
        print("Need at least 2 models with results to compare. Run training first.")
        return

    fig_dir = os.path.join(COMPARISON_DIR, "figures")

    # bar comparison
    p = plot_comparison_bar(results,
                            os.path.join(fig_dir, "model_comparison_bar.png"),
                            metrics_to_plot=['rmse', 'mae'])
    print(f"  Saved: {p}")

    p = plot_comparison_bar(results,
                            os.path.join(fig_dir, "model_comparison_mape_r2.png"),
                            metrics_to_plot=['mape'])
    print(f"  Saved: {p}")

    # per-horizon comparison
    model_dirs = [v for k, v in model_configs.items() if k in results]
    model_names = [k for k in model_configs if k in results]
    p = plot_per_horizon_comparison(model_dirs, model_names)
    print(f"  Saved: {p}")

    # generate latex tables
    latex_main = generate_latex_table(results, caption="Performance Comparison of Proposed Models")
    latex_improve = generate_improvement_table(results, baseline_name="Classical DLinear")

    with open(os.path.join(COMPARISON_DIR, "table_comparison.tex"), "w") as f:
        f.write(latex_main)
    print(f"  Saved LaTeX table: {os.path.join(COMPARISON_DIR, 'table_comparison.tex')}")

    with open(os.path.join(COMPARISON_DIR, "table_improvement.tex"), "w") as f:
        f.write(latex_improve)
    print(f"  Saved LaTeX table: {os.path.join(COMPARISON_DIR, 'table_improvement.tex')}")

    # combined summary CSV
    rows = []
    for name, r in results.items():
        rows.append({"model": name, **r})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(COMPARISON_DIR, "all_models_summary.csv"), index=False)

    # push to wandb
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="final_comparison",
        tags=["comparison", "summary"],
        reinit=True,
    )

    run.log({"comparison_table": wandb.Table(dataframe=summary_df)})

    for fname in os.listdir(fig_dir):
        if fname.endswith(".png"):
            run.log({f"comparison/{fname}": wandb.Image(os.path.join(fig_dir, fname))})

    art = wandb.Artifact("comparison_results", type="results")
    art.add_dir(COMPARISON_DIR)
    run.log_artifact(art)

    run.finish()
    print("\nComparison complete.")


if __name__ == "__main__":
    main()