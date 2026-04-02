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
from models.laq import LAQ


def main():
    set_seed(SEED)

    print("=" * 60)
    print("Lag-Aware Quantum DLinear (LAQ)")
    print("=" * 60)

    save_dir = os.path.join(BASE_DIR, "laq")
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

    model = LAQ(
        n_features=data_info["n_features"],
        pred_len=FORECAST_HORIZON,
        seq_len=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_qubits=N_QUBITS,
        n_qlayers=N_QLAYERS,
        lag_positions=LAQ_LAGS,
    )

    print(f"Lag positions used: {model.lag_offsets}")
    print(f"Number of lag features: {len(model.lag_offsets)}")

    run_config = {
        "model": "LAQ",
        "seq_length": SEQ_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "epochs": EPOCHS,
        "kernel_size": KERNEL_SIZE,
        "n_qubits": N_QUBITS,
        "n_qlayers": N_QLAYERS,
        "n_lag_features": len(model.lag_offsets),
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
        name="LAQ",
        config=run_config,
        tags=["quantum", "laq", "novel"],
        reinit=True,
    )

    trainer = Trainer(model, run_config, save_dir, run_name="laq")
    total_params, trainable_params = trainer.count_parameters()
    print(f"Parameters: {total_params} total, {trainable_params} trainable")
    run.config.update({"total_params": total_params, "trainable_params": trainable_params})

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

    art = wandb.Artifact("laq_best", type="model")
    art.add_file(ckpt_path)
    run.log_artifact(art)

    fig_dir = os.path.join(save_dir, "figures")

    p = plot_forecast_window(dates_test, actuals_inv, preds_inv,
                             os.path.join(fig_dir, "forecast_window.png"),
                             tag="LAQ", color='#2ca02c')
    run.log({"figures/forecast_window": wandb.Image(p)})

    p = plot_residual_distribution(actuals_inv, preds_inv,
                                   os.path.join(fig_dir, "residual_dist.png"),
                                   tag="LAQ", color='#2ca02c')
    run.log({"figures/residual_dist": wandb.Image(p)})

    p = plot_monthly_grid(dates_test, actuals_inv, preds_inv,
                          os.path.join(fig_dir, "monthly_grid.png"),
                          tag="LAQ")
    run.log({"figures/monthly_grid": wandb.Image(p)})

    p = plot_training_curves(trainer.train_losses, trainer.val_losses,
                             os.path.join(fig_dir, "training_curves.png"),
                             tag="LAQ")
    run.log({"figures/training_curves": wandb.Image(p)})

    p = plot_scatter(actuals_inv, preds_inv,
                     os.path.join(fig_dir, "scatter.png"),
                     tag="LAQ")
    run.log({"figures/scatter": wandb.Image(p)})

    p = plot_hourly_error(hourly,
                          os.path.join(fig_dir, "hourly_error.png"),
                          tag="LAQ")
    if p:
        run.log({"figures/hourly_error": wandb.Image(p)})

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False)

    # log modulation pattern
    model.eval()
    all_mods = []
    with torch.no_grad():
        for bx, _ in test_loader:
            lag_vals = model._extract_lags(bx)
            q_in = model.lag_compressor(lag_vals) * np.pi
            q_out = model.vqc(q_in)
            mod = model.modulation_head(q_out)
            all_mods.append(mod.cpu().numpy())

    all_mods = np.vstack(all_mods)
    mean_mod = all_mods.mean(axis=0)
    std_mod = all_mods.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    hours = range(1, FORECAST_HORIZON + 1)
    ax.plot(hours, mean_mod, 'o-', color='#2ca02c', markersize=5)
    ax.fill_between(hours, mean_mod - std_mod, mean_mod + std_mod,
                    color='#2ca02c', alpha=0.2)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Modulation Factor')
    ax.set_title('LAQ — Learned Per-Hour Quantum Modulation')
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    mod_path = os.path.join(fig_dir, "modulation_pattern.png")
    plt.savefig(mod_path, dpi=300)
    plt.close()
    run.log({"figures/modulation_pattern": wandb.Image(mod_path)})

    alpha = torch.sigmoid(model.fusion_alpha).item()
    print(f"\nFusion alpha (seasonal weight): {alpha:.4f}")
    run.log({"fusion_alpha": alpha})

    run.finish()
    print("\nDone. All artifacts saved.")


if __name__ == "__main__":
    main()