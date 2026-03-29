"""
Ablation study for ADQRL.

Usage:
    python run_ablation.py                          # run all 5 modes
    python run_ablation.py --mode full              # run only 'full'
    python run_ablation.py --mode no_quantum        # run only 'no_quantum'
    python run_ablation.py --mode full no_quantum   # run two specific modes
    python run_ablation.py --report                 # generate report from saved checkpoints (no training)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import wandb
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config import *
from utils.reproducibility import set_seed
from utils.data import load_data, make_loaders
from utils.trainer import Trainer
from utils.metrics import compute_metrics

ALL_MODES = ['full', 'no_quantum', 'trend_only', 'fixed_decomp', 'no_fusion']
DISPLAY_NAMES = {
    'full': 'Full ADQRL',
    'no_quantum': 'No Quantum',
    'trend_only': 'Trend Only',
    'fixed_decomp': 'Fixed Decomp',
    'no_fusion': 'No Fusion',
}


def run_one_ablation(mode, train_loader, val_loader, test_loader,
                     data_info, run_config):
    from models.ablation_variants import AblationADQRL

    set_seed(SEED)

    model = AblationADQRL(
        n_features=data_info["n_features"],
        pred_len=FORECAST_HORIZON,
        seq_len=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_qubits=N_QUBITS,
        n_qlayers=N_QLAYERS,
        context_size=CONTEXT_SIZE,
        mode=mode,
    )

    cfg = {**run_config, "ablation_mode": mode}
    run_name = f"ablation_{mode}"

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        group="component_ablation",
        config=cfg,
        tags=["ablation", "component_ablation"],
        reinit=True,
    )

    trainer = Trainer(model, cfg, ABLATION_DIR, run_name=run_name)
    total_p, train_p = trainer.count_parameters()
    print(f"  [{run_name}] params: {total_p} total, {train_p} trainable")
    run.config.update({"total_params": total_p, "trainable_params": train_p})

    trainer.fit(train_loader, val_loader, wandb_run=run)
    metrics, preds_inv, actuals_inv = trainer.evaluate(test_loader, data_info["scaler_y"])

    run.log({f"test/{k}": v for k, v in metrics.items()})
    trainer.save_checkpoint(extra_info={"n_features": data_info["n_features"], "metrics": metrics})

    run.finish()
    return metrics


def evaluate_from_checkpoint(mode, test_loader, data_info):
    from models.ablation_variants import AblationADQRL

    ckpt_path = os.path.join(ABLATION_DIR, "checkpoints", f"ablation_{mode}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: checkpoint not found for '{mode}' at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = AblationADQRL(
        n_features=data_info["n_features"],
        pred_len=FORECAST_HORIZON,
        seq_len=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_qubits=N_QUBITS,
        n_qlayers=N_QLAYERS,
        context_size=CONTEXT_SIZE,
        mode=mode,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds_list, actuals_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx)
            if out.dim() == 3:
                out = out[:, :, -1]
            preds_list.append(out.cpu().numpy())
            actuals_list.append(by.numpy())

    preds = np.vstack(preds_list)
    actuals = np.vstack(actuals_list)

    scaler_y = data_info["scaler_y"]
    preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    actuals_inv = scaler_y.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)

    metrics = compute_metrics(actuals_inv.flatten(), preds_inv.flatten())
    return metrics


def generate_report(results):
    fig_dir = os.path.join(ABLATION_DIR, "figures")

    modes = [m for m in ALL_MODES if m in results]
    display = [DISPLAY_NAMES[m] for m in modes]
    rmses = [results[m]['rmse'] for m in modes]
    maes = [results[m]['mae'] for m in modes]
    r2s = [results[m]['r2'] for m in modes]

    x = np.arange(len(modes))
    w = 0.3

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - w/2, rmses, w, label='RMSE (MW)', color='#4C72B0')
    ax1.bar(x + w/2, maes, w, label='MAE (MW)', color='#DD8452')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display, rotation=15, ha='right')
    ax1.set_ylabel('Error (MW)')
    ax1.set_title('Ablation Study — Component Contribution')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.25, axis='y')

    ax2 = ax1.twinx()
    ax2.plot(x, r2s, 'go-', label='R²', markersize=6)
    ax2.set_ylabel('R²')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ablation_components.png"), dpi=300)
    plt.close()

    rows = [{"experiment": m, "display_name": DISPLAY_NAMES[m], **results[m]} for m in modes]
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(ABLATION_DIR, "ablation_results.csv"), index=False)

    with open(os.path.join(ABLATION_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # latex table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study — Component Contribution}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Variant & RMSE (MW) & MAE (MW) & MAPE (\\%) & R² & Bias (MW) \\\\")
    lines.append("\\midrule")
    for m in modes:
        r = results[m]
        lines.append(f"{DISPLAY_NAMES[m]} & {r['rmse']:.2f} & {r['mae']:.2f} & "
                     f"{r['mape']:.2f} & {r['r2']:.4f} & {r['mean_residual']:.2f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(ABLATION_DIR, "ablation_table.tex"), "w") as f:
        f.write("\n".join(lines))

    summary_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="ablation_summary",
        tags=["ablation", "summary"],
        reinit=True,
    )
    summary_run.log({"ablation_results": wandb.Table(dataframe=results_df)})

    art = wandb.Artifact("ablation_results", type="results")
    art.add_file(os.path.join(ABLATION_DIR, "ablation_results.csv"))
    art.add_file(os.path.join(fig_dir, "ablation_components.png"))
    art.add_file(os.path.join(ABLATION_DIR, "ablation_table.tex"))
    summary_run.log_artifact(art)
    summary_run.finish()

    print("\nAblation report generated:")
    print(f"  {os.path.join(ABLATION_DIR, 'ablation_results.csv')}")
    print(f"  {os.path.join(ABLATION_DIR, 'ablation_table.tex')}")
    print(f"  {os.path.join(fig_dir, 'ablation_components.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", nargs="*", default=None,
                        choices=ALL_MODES,
                        help="Which ablation modes to train. Omit to run all.")
    parser.add_argument("--report", action="store_true",
                        help="Skip training. Generate report from saved checkpoints.")
    args = parser.parse_args()

    set_seed(SEED)

    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

    (X_train, y_train, dates_train,
     X_val, y_val, dates_val,
     X_test, y_test, dates_test,
     data_info) = load_data(DATA_PATH, SEQ_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO)

    if args.report:
        print("\nGenerating report from saved checkpoints...")
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=BATCH_SIZE, shuffle=False
        )
        results = {}
        for mode in ALL_MODES:
            m = evaluate_from_checkpoint(mode, test_loader, data_info)
            if m is not None:
                results[mode] = m
                print(f"  {mode}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
                      f"MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}")
        if results:
            generate_report(results)
        else:
            print("No checkpoints found. Train first.")
        return

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE
    )

    base_config = {
        "seq_length": SEQ_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "kernel_size": KERNEL_SIZE,
        "n_qubits": N_QUBITS,
        "n_qlayers": N_QLAYERS,
        "context_size": CONTEXT_SIZE,
        "seed": SEED,
        "n_features": data_info["n_features"],
        "early_stop_patience": 15,
    }

    modes_to_run = args.mode if args.mode else ALL_MODES

    for mode in modes_to_run:
        print(f"\n{'='*40} {mode} {'='*40}")
        m = run_one_ablation(mode, train_loader, val_loader, test_loader,
                             data_info, base_config)
        print(f"  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
              f"MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}")

    # if all modes have been trained, generate the report
    all_done = all(
        os.path.exists(os.path.join(ABLATION_DIR, "checkpoints", f"ablation_{m}_best.pth"))
        for m in ALL_MODES
    )
    if all_done:
        print("\nAll modes complete. Generating report...")
        test_loader_eval = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=BATCH_SIZE, shuffle=False
        )
        results = {}
        for mode in ALL_MODES:
            results[mode] = evaluate_from_checkpoint(mode, test_loader_eval, data_info)
        generate_report(results)
    else:
        missing = [m for m in ALL_MODES if not os.path.exists(
            os.path.join(ABLATION_DIR, "checkpoints", f"ablation_{m}_best.pth"))]
        print(f"\nRemaining modes to train: {missing}")
        print("Run them, then use --report to generate the final report.")


if __name__ == "__main__":
    main()