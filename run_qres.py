import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import wandb
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs.config import *
from utils.reproducibility import set_seed
from utils.data import load_data, make_loaders
from utils.trainer import Trainer
from utils.metrics import compute_hourly_metrics
from utils.plotting import (
    plot_forecast_window, plot_residual_distribution, plot_monthly_grid,
    plot_training_curves, plot_scatter, plot_hourly_error,
)
from models.qres import QRes


def main():
    set_seed(SEED)

    print("=" * 60)
    print("Quantum Reservoir DLinear (QRes)")
    print("=" * 60)

    save_dir = os.path.join(BASE_DIR, "qres")
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    (X_train, y_train, dates_train,
     X_val, y_val, dates_val,
     X_test, y_test, dates_test,
     data_info) = load_data(DATA_PATH, SEQ_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE
    )

    print(f"Data: train={data_info['n_train']}, val={data_info['n_val']}, "
          f"test={data_info['n_test']}, features={data_info['n_features']}")

    N_RESERVOIR_LAYERS = 3

    model = QRes(
        n_features=data_info["n_features"],
        pred_len=FORECAST_HORIZON,
        seq_len=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_qubits=N_QUBITS,
        n_reservoir_layers=N_RESERVOIR_LAYERS,
        lag_positions=LAQ_LAGS,
        seed=SEED,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_params = sum(p.numel() for p in model.reservoir.parameters())
    print(f"Parameters: {total_params} total, {trainable_params} trainable, {quantum_params} quantum (frozen)")
    print(f"Lag positions: {LAQ_LAGS}")
    print(f"Reservoir outputs: {model.reservoir.n_outputs} features from {N_QUBITS} qubits")

    run_config = {
        "model": "QRes",
        "seq_length": SEQ_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "kernel_size": KERNEL_SIZE,
        "n_qubits": N_QUBITS,
        "n_reservoir_layers": N_RESERVOIR_LAYERS,
        "seed": SEED,
        "train_samples": data_info["n_train"],
        "val_samples": data_info["n_val"],
        "test_samples": data_info["n_test"],
        "n_features": data_info["n_features"],
        "early_stop_patience": 15,
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="QRes",
        config=run_config,
        tags=["quantum", "qres", "reservoir", "novel"],
        reinit=True,
    )
    run.config.update({"total_params": total_params, "trainable_params": trainable_params})

    trainer = Trainer(model, run_config, save_dir, run_name="qres")
    trainer.fit(train_loader, val_loader, wandb_run=run)

    metrics, preds_inv, actuals_inv = trainer.evaluate(test_loader, data_info["scaler_y"])

    print("\n--- Test Set Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    run.log({f"test/{k}": v for k, v in metrics.items()})

    hourly = compute_hourly_metrics(actuals_inv, preds_inv)
    hourly_df = pd.DataFrame(hourly)
    hourly_df.to_csv(os.path.join(save_dir, "hourly_metrics.csv"), index=False)
    run.log({"test/hourly_metrics": wandb.Table(dataframe=hourly_df)})

    ckpt_path = trainer.save_checkpoint(extra_info={
        "n_features": data_info["n_features"],
        "metrics": metrics,
    })

    art = wandb.Artifact("qres_best", type="model")
    art.add_file(ckpt_path)
    run.log_artifact(art)

    fig_dir = os.path.join(save_dir, "figures")

    p = plot_forecast_window(dates_test, actuals_inv, preds_inv,
                             os.path.join(fig_dir, "forecast_window.png"),
                             tag="QRes", color='#17becf')
    run.log({"figures/forecast_window": wandb.Image(p)})

    p = plot_residual_distribution(actuals_inv, preds_inv,
                                   os.path.join(fig_dir, "residual_dist.png"),
                                   tag="QRes", color='#17becf')
    run.log({"figures/residual_dist": wandb.Image(p)})

    p = plot_monthly_grid(dates_test, actuals_inv, preds_inv,
                          os.path.join(fig_dir, "monthly_grid.png"),
                          tag="QRes")
    run.log({"figures/monthly_grid": wandb.Image(p)})

    p = plot_training_curves(trainer.train_losses, trainer.val_losses,
                             os.path.join(fig_dir, "training_curves.png"),
                             tag="QRes")
    run.log({"figures/training_curves": wandb.Image(p)})

    p = plot_scatter(actuals_inv, preds_inv,
                     os.path.join(fig_dir, "scatter.png"),
                     tag="QRes")
    run.log({"figures/scatter": wandb.Image(p)})

    p = plot_hourly_error(hourly,
                          os.path.join(fig_dir, "hourly_error.png"),
                          tag="QRes")
    if p:
        run.log({"figures/hourly_error": wandb.Image(p)})

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False)

    # visualize reservoir feature distribution
    model.eval()
    all_feats = []
    with torch.no_grad():
        for bx, _ in test_loader:
            demand = bx[:, :, -1]
            lag_vals = demand[:, model.lag_indices]
            q_in = model.lag_compressor(lag_vals) * np.pi
            feats = model.reservoir(q_in)
            all_feats.append(feats.cpu().numpy())

    all_feats = np.vstack(all_feats)
    feat_labels = ([f"X{i}" for i in range(N_QUBITS)] +
                   [f"Y{i}" for i in range(N_QUBITS)] +
                   [f"Z{i}" for i in range(N_QUBITS)])

    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot([all_feats[:, i] for i in range(all_feats.shape[1])],
                    labels=feat_labels, patch_artist=True)
    colors = ['#ff9999'] * N_QUBITS + ['#99ccff'] * N_QUBITS + ['#99ff99'] * N_QUBITS
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_xlabel('Observable')
    ax.set_ylabel('Expectation Value')
    ax.set_title('QRes — Reservoir Feature Distribution (Test Set)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    feat_path = os.path.join(fig_dir, "reservoir_features.png")
    plt.savefig(feat_path, dpi=300)
    plt.close()
    run.log({"figures/reservoir_features": wandb.Image(feat_path)})

    alpha = torch.sigmoid(model.fusion_alpha).item()
    print(f"\nFusion alpha (seasonal weight): {alpha:.4f}")
    run.log({"fusion_alpha": alpha})

    run.finish()
    print("\nDone. All artifacts saved.")


if __name__ == "__main__":
    main()