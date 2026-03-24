import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath, seq_length, forecast_horizon, train_ratio=0.8, val_ratio=0.1):
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    dates = df['datetime'].values

    cols_except_demand = [c for c in df.columns if c not in ['datetime', 'demand']]
    feature_cols = cols_except_demand + ['demand']

    data_x = df[feature_cols].values.astype(float)
    data_y = df[['demand']].values.astype(float)

    # fit scalers on training portion only
    train_end = int(len(data_x) * train_ratio)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    data_x_scaled = data_x.copy()
    data_y_scaled = data_y.copy()

    data_x_scaled[:train_end] = scaler_x.fit_transform(data_x[:train_end])
    data_x_scaled[train_end:] = scaler_x.transform(data_x[train_end:])

    data_y_scaled[:train_end] = scaler_y.fit_transform(data_y[:train_end])
    data_y_scaled[train_end:] = scaler_y.transform(data_y[train_end:])

    # build sliding window sequences
    xs, ys, date_list = [], [], []
    for i in range(len(data_x_scaled) - seq_length - forecast_horizon + 1):
        xs.append(data_x_scaled[i:i + seq_length])
        ys.append(data_y_scaled[i + seq_length:i + seq_length + forecast_horizon].flatten())
        date_list.append(dates[i + seq_length + forecast_horizon - 1])

    X = np.array(xs)
    y = np.array(ys)
    seq_dates = pd.to_datetime(np.array(date_list))

    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.float32)
    dates_train = seq_dates[:n_train]

    X_val = torch.tensor(X[n_train:n_train + n_val], dtype=torch.float32)
    y_val = torch.tensor(y[n_train:n_train + n_val], dtype=torch.float32)
    dates_val = seq_dates[n_train:n_train + n_val]

    X_test = torch.tensor(X[n_train + n_val:], dtype=torch.float32)
    y_test = torch.tensor(y[n_train + n_val:], dtype=torch.float32)
    dates_test = seq_dates[n_train + n_val:]

    n_features = len(feature_cols)

    data_info = {
        "n_features": n_features,
        "feature_cols": feature_cols,
        "n_train": n_train,
        "n_val": len(X_val),
        "n_test": len(X_test),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "total_samples": n,
    }

    return (X_train, y_train, dates_train,
            X_val, y_val, dates_val,
            X_test, y_test, dates_test,
            data_info)


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
