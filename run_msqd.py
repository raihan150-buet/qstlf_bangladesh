import os
import sys
import json
import wandb
import torch
import pandas as pd

from configs.config import *
from utils.reproducibility import set_seed
from utils.data import load_data, make_loaders
from utils.trainer import Trainer
from utils.metrics import compute_hourly_metrics
from utils.plotting import (
    plot_forecast_window, plot_residual_distribution, plot_monthly_grid,
    plot_training_curves, plot_scatter, plot_hourly_error,
)
from models.msqd import MSQD


def main():
    set_seed(SEED)

    print("=" * 60)
    print("Multi-Scale Quantum Decomposition (MSQD)")
    print("=" * 60)

    save_dir = os.path.join(BASE_DIR, "msqd")
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

    model = MSQD(
        n_features=data_info["n_features"],
        pred_len=FORECAST_HORIZON,
        seq_len=SEQ_LENGTH,
        kernel_size=KERNEL_SIZE,
        n_qubits=N_QUBITS,
        n_qlayers=N_QLAYERS,
        context_size=CONTEXT_SIZE,
    )

    run_config = {
        "model": "MSQD",
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
        "train_samples": data_info["n_train"],
        "val_samples": data_info["n_val"],
        "test_samples": data_info["n_test"],
        "n_features": data_info["n_features"],
        "early_stop_patience": 15,
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="MSQD",
        config=run_config,
        tags=["quantum", "msqd", "novel"],
        reinit=True,
    )

    trainer = Trainer(model, run_config, save_dir, run_name="msqd")
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

    art = wandb.Artifact("msqd_best", type="model")
    art.add_file(ckpt_path)
    run.log_artifact(art)

    fig_dir = os.path.join(save_dir, "figures")

    p = plot_forecast_window(dates_test, actuals_inv, preds_inv,
                             os.path.join(fig_dir, "forecast_window.png"),
                             tag="MSQD", color='darkgreen')
    run.log({"figures/forecast_window": wandb.Image(p)})

    p = plot_residual_distribution(actuals_inv, preds_inv,
                                   os.path.join(fig_dir, "residual_dist.png"),
                                   tag="MSQD", color='darkgreen')
    run.log({"figures/residual_dist": wandb.Image(p)})

    p = plot_monthly_grid(dates_test, actuals_inv, preds_inv,
                          os.path.join(fig_dir, "monthly_grid.png"),
                          tag="MSQD")
    run.log({"figures/monthly_grid": wandb.Image(p)})

    p = plot_training_curves(trainer.train_losses, trainer.val_losses,
                             os.path.join(fig_dir, "training_curves.png"),
                             tag="MSQD")
    run.log({"figures/training_curves": wandb.Image(p)})

    p = plot_scatter(actuals_inv, preds_inv,
                     os.path.join(fig_dir, "scatter.png"),
                     tag="MSQD")
    run.log({"figures/scatter": wandb.Image(p)})

    p = plot_hourly_error(hourly,
                          os.path.join(fig_dir, "hourly_error.png"),
                          tag="MSQD")
    run.log({"figures/hourly_error": wandb.Image(p)})

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False)

    # log the learned spectral gate for analysis
    gate_vals = torch.sigmoid(model.spectral_gate.gate_logits).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(gate_vals, linewidth=1.2)
    ax.set_xlabel('Frequency Bin')
    ax.set_ylabel('Gate Value (1=low-freq pass)')
    ax.set_title('Learned Spectral Gate')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    gate_path = os.path.join(fig_dir, "spectral_gate.png")
    plt.savefig(gate_path, dpi=300)
    plt.close()
    run.log({"figures/spectral_gate": wandb.Image(gate_path)})

    # log fusion weights
    fweights = torch.softmax(model.fusion_logits, dim=0).detach().cpu().numpy()
    print(f"\nLearned fusion weights: trend={fweights[0]:.3f}, "
          f"low_freq={fweights[1]:.3f}, high_freq={fweights[2]:.3f}")
    run.log({
        "fusion/trend_weight": fweights[0],
        "fusion/low_freq_weight": fweights[1],
        "fusion/high_freq_weight": fweights[2],
    })

    run.finish()
    print("\nDone. All artifacts saved.")


if __name__ == "__main__":
    main()
