"""
Evaluate a saved model checkpoint without retraining.

Usage:
    python evaluate_checkpoint.py --model adqrl --checkpoint outputs/quantum_adqrl/checkpoints/quantum_adqrl_best.pth
    python evaluate_checkpoint.py --model classical --checkpoint outputs/classical_dlinear/checkpoints/classical_dlinear_best.pth
    python evaluate_checkpoint.py --model msqd --checkpoint outputs/msqd/checkpoints/msqd_best.pth
    python evaluate_checkpoint.py --model ablation --mode no_quantum --checkpoint outputs/ablation/checkpoints/ablation_no_quantum_best.pth
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import torch
import numpy as np

from configs.config import *
from utils.reproducibility import set_seed
from utils.data import load_data, make_loaders
from utils.metrics import compute_metrics, compute_hourly_metrics


def load_model(model_type, n_features, mode=None):
    if model_type == "classical":
        from models.classical_dlinear import ClassicalDLinear
        return ClassicalDLinear(n_features, FORECAST_HORIZON, KERNEL_SIZE, SEQ_LENGTH)

    elif model_type == "adqrl":
        from models.quantum_adqrl import ADQRL
        return ADQRL(n_features, FORECAST_HORIZON, SEQ_LENGTH, KERNEL_SIZE,
                     N_QUBITS, N_QLAYERS, CONTEXT_SIZE)

    elif model_type == "msqd":
        from models.msqd import MSQD
        return MSQD(n_features, FORECAST_HORIZON, SEQ_LENGTH, KERNEL_SIZE,
                    N_QUBITS, N_QLAYERS, CONTEXT_SIZE)

    elif model_type == "qmod":
        from models.qmod import QMod
        return QMod(n_features, FORECAST_HORIZON, SEQ_LENGTH, KERNEL_SIZE,
                    N_QUBITS, N_QLAYERS, CONTEXT_SIZE)

    elif model_type == "laq":
        from models.laq import LAQ
        return LAQ(n_features, FORECAST_HORIZON, SEQ_LENGTH, KERNEL_SIZE,
                   N_QUBITS, N_QLAYERS)

    elif model_type == "ablation":
        from models.ablation_variants import AblationADQRL
        if mode is None:
            raise ValueError("--mode required for ablation model")
        return AblationADQRL(n_features, FORECAST_HORIZON, SEQ_LENGTH, KERNEL_SIZE,
                             N_QUBITS, N_QLAYERS, CONTEXT_SIZE, mode=mode)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["classical", "adqrl", "msqd", "qmod", "laq", "ablation"])
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--mode", default=None, help="Ablation mode (only for --model ablation)")
    args = parser.parse_args()

    set_seed(SEED)

    (_, _, _, _, _, _, X_test, y_test, dates_test, data_info) = load_data(
        DATA_PATH, SEQ_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = load_model(args.model, data_info["n_features"], mode=args.mode)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Trained for {ckpt.get('best_epoch', '?')} epochs, val_loss={ckpt.get('best_val_loss', '?')}")

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

    print("\n--- Test Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    hourly = compute_hourly_metrics(actuals_inv, preds_inv)
    print("\n--- Per-Horizon RMSE ---")
    for h in hourly:
        print(f"  Hour {h['hour']:2d}: RMSE={h['rmse']:.2f}  MAE={h['mae']:.2f}")


if __name__ == "__main__":
    main()