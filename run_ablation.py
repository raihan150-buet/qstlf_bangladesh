import os
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


def run_one_ablation(mode, train_loader, val_loader, test_loader,
                     data_info, run_config, group_name):
    from models.ablation_variants import AblationADQRL

    set_seed(SEED)

    model = AblationADQRL(
        n_features=data_info["n_features"],
        pred_len=FORECAST_HORIZON,
        seq_len=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_qubits=run_config.get("n_qubits", N_QUBITS),
        n_qlayers=N_QLAYERS,
        context_size=CONTEXT_SIZE,
        mode=mode,
    )

    cfg = {**run_config, "ablation_mode": mode}
    run_name = f"ablation_{mode}" if "qubits" not in mode else mode

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        group=group_name,
        config=cfg,
        tags=["ablation", group_name],
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


def main():
    set_seed(SEED)

    print("=" * 60)
    print("Ablation Study")
    print("=" * 60)

    (X_train, y_train, dates_train,
     X_val, y_val, dates_val,
     X_test, y_test, dates_test,
     data_info) = load_data(DATA_PATH, SEQ_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO)

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

    # --- Part 1: Component ablation ---
    print("\n--- Component Ablation ---")
    ablation_modes = ['full', 'no_quantum', 'trend_only', 'fixed_decomp', 'no_fusion']
    ablation_results = {}

    for mode in ablation_modes:
        print(f"\n{'='*40} {mode} {'='*40}")
        m = run_one_ablation(mode, train_loader, val_loader, test_loader,
                             data_info, base_config, "component_ablation")
        ablation_results[mode] = m
        print(f"  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}")

    # --- Part 2: Qubit sensitivity ---
    print("\n--- Qubit Sensitivity ---")
    qubit_counts = [2, 4, 6, 8]
    qubit_results = {}

    for q in qubit_counts:
        print(f"\n{'='*40} qubits={q} {'='*40}")
        cfg = {**base_config, "n_qubits": q}
        m = run_one_ablation("full", train_loader, val_loader, test_loader,
                             data_info, cfg, "qubit_sensitivity")
        qubit_results[q] = m
        print(f"  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE={m['mape']:.2f}%")

    # --- Save combined results ---
    fig_dir = os.path.join(ABLATION_DIR, "figures")

    # component ablation bar chart
    labels = list(ablation_results.keys())
    display_labels = ['Full ADQRL', 'No Quantum', 'Trend Only', 'Fixed Decomp', 'No Fusion']
    rmses = [ablation_results[k]['rmse'] for k in labels]
    maes = [ablation_results[k]['mae'] for k in labels]
    r2s = [ablation_results[k]['r2'] for k in labels]

    x = np.arange(len(labels))
    w = 0.3

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - w/2, rmses, w, label='RMSE (MW)', color='#4C72B0')
    ax1.bar(x + w/2, maes, w, label='MAE (MW)', color='#DD8452')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, rotation=15, ha='right')
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

    # qubit sensitivity plot
    qubits = list(qubit_results.keys())
    q_rmse = [qubit_results[q]['rmse'] for q in qubits]
    q_mape = [qubit_results[q]['mape'] for q in qubits]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(qubits, q_rmse, 'o-', color='#C44E52', label='RMSE (MW)', markersize=6)
    l2 = ax2.plot(qubits, q_mape, 's--', color='#4C72B0', label='MAPE (%)', markersize=6)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('RMSE (MW)', color='#C44E52')
    ax2.set_ylabel('MAPE (%)', color='#4C72B0')
    ax1.set_xticks(qubits)
    ax1.tick_params(axis='y', labelcolor='#C44E52')
    ax2.tick_params(axis='y', labelcolor='#4C72B0')

    lines = l1 + l2
    labels_l = [l.get_label() for l in lines]
    ax1.legend(lines, labels_l, loc='best')

    plt.title('Qubit Count Sensitivity Analysis')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "qubit_sensitivity.png"), dpi=300)
    plt.close()

    # save tables
    rows = []
    for k, v in ablation_results.items():
        rows.append({"experiment": k, "group": "component", **v})
    for q, v in qubit_results.items():
        rows.append({"experiment": f"qubits_{q}", "group": "qubit_sensitivity", **v})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(ABLATION_DIR, "ablation_all_results.csv"), index=False)

    with open(os.path.join(ABLATION_DIR, "ablation_all_results.json"), "w") as f:
        json.dump({"component": ablation_results,
                    "qubit_sensitivity": {str(k): v for k, v in qubit_results.items()}},
                   f, indent=2)

    # log summary to wandb
    summary_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="ablation_summary",
        tags=["ablation", "summary"],
        reinit=True,
    )
    summary_run.log({"ablation_results": wandb.Table(dataframe=results_df)})

    art = wandb.Artifact("ablation_results", type="results")
    art.add_file(os.path.join(ABLATION_DIR, "ablation_all_results.csv"))
    art.add_file(os.path.join(fig_dir, "ablation_components.png"))
    art.add_file(os.path.join(fig_dir, "qubit_sensitivity.png"))
    summary_run.log_artifact(art)

    summary_run.finish()
    print("\nAblation study complete. All results saved.")


if __name__ == "__main__":
    main()
