import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(actuals, preds):
    mse = mean_squared_error(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actuals - preds) / (np.abs(actuals) + 1e-8))) * 100
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mean_residual = np.mean(actuals - preds)
    std_residual = np.std(actuals - preds)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
        "mean_residual": float(mean_residual),
        "std_residual": float(std_residual),
    }


def compute_hourly_metrics(actuals_2d, preds_2d):
    """Per-horizon-step metrics. actuals_2d shape: (n_samples, horizon)"""
    horizon = actuals_2d.shape[1]
    results = []
    for h in range(horizon):
        m = compute_metrics(actuals_2d[:, h], preds_2d[:, h])
        m["hour"] = h + 1
        results.append(m)
    return results
