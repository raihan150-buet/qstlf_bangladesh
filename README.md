# Hybrid Quantum-Classical Short-Term Load Forecasting

Comparative study of DLinear, ADQRL, and MSQD architectures for day-ahead electricity demand forecasting on Bangladesh's power grid data.

## Project Structure

```
project/
├── .env                       # W&B API key (not committed to git)
├── .gitignore
├── requirements.txt
├── configs/
│   └── config.py              # All hyperparameters and paths (single source of truth)
├── models/
│   ├── classical_dlinear.py   # DLinear (Zeng et al. 2022) — benchmark
│   ├── quantum_adqrl.py       # Adaptive Decomposition with Quantum Residual Learning
│   ├── msqd.py                # Multi-Scale Quantum Decomposition (novel)
│   └── ablation_variants.py   # Ablation study model variants
├── utils/
│   ├── reproducibility.py     # Seed management
│   ├── data.py                # Data loading with train/val/test split
│   ├── metrics.py             # Evaluation metrics (MSE, RMSE, MAE, MAPE, R², per-horizon)
│   ├── trainer.py             # Training engine with early stopping, checkpointing, wandb logging
│   └── plotting.py            # Publication-quality figures
├── run_classical.py           # Train & evaluate Classical DLinear
├── run_quantum.py             # Train & evaluate Quantum ADQRL
├── run_msqd.py                # Train & evaluate MSQD
├── run_ablation.py            # Component ablation + qubit sensitivity study
├── run_comparison.py          # Generate comparison figures and LaTeX tables
├── run_all.py                 # Master script — runs full pipeline
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Weights & Biases — create a `.env` file in the project root:
```
WANDB_API_KEY=your_actual_key_here
```
Then set your username or team in `configs/config.py`:
```python
WANDB_ENTITY = "your_wandb_username"
```

3. Place `selected_features.xlsx` in the project root directory.

## Running Experiments

Run everything:
```bash
python run_all.py
```

Or run individually:
```bash
python run_classical.py       # Step 1: DLinear benchmark
python run_quantum.py         # Step 2: ADQRL
python run_msqd.py            # Step 3: MSQD
python run_ablation.py        # Step 4: Ablation study (takes longest)
python run_comparison.py      # Step 5: Cross-model comparison tables/figures
```

Skip ablation (fastest for iteration):
```bash
python run_all.py --skip-ablation
```

Regenerate comparison figures only (no retraining):
```bash
python run_all.py --comparison-only
```

## Models

### Classical DLinear (Benchmark)
Faithful implementation of DLinear from Zeng et al. "Are Transformers Effective for Time Series Forecasting?" (2022). Decomposes input into trend and seasonal components via moving average, applies separate linear layers along the temporal axis.

### ADQRL (Quantum)
Adaptive Decomposition with Quantum Residual Learning. Replaces the seasonal linear layer with a VQC branch: learnable Conv1d decomposition → compress features to 4 qubits → Angle Embedding + StronglyEntanglingLayers → project back. Uses learnable sigmoid fusion weight.

### MSQD (Novel — Multi-Scale Quantum Decomposition)
Extends ADQRL by recognizing that electricity demand contains overlapping periodicities (daily, weekly). Instead of collapsing the full seasonal signal into one VQC, MSQD uses a learnable spectral gate (FFT domain) to split seasonal patterns into frequency bands, then processes each band with a dedicated VQC. Uses softmax-normalized 3-way fusion (trend + low-freq + high-freq).

## Outputs

After a full run, the `outputs/` directory contains:

- `*/checkpoints/` — Best model weights (.pth files)
- `*/figures/` — All publication-quality figures (300 DPI)
- `*/test_metrics.json` — Final test metrics
- `*/hourly_metrics.csv` — Per-forecast-horizon error breakdown
- `comparison/table_comparison.tex` — LaTeX table for the paper
- `comparison/table_improvement.tex` — Percentage improvement table
- `comparison/all_models_summary.csv` — Combined results CSV

Everything is also uploaded to Weights & Biases as artifacts.

## Reproducibility

- All experiments use seed=42 with deterministic PyTorch settings
- The data split is chronological (no shuffling across time)
- Scalers are fit on training data only
- Config is saved inside every checkpoint
- Full train/val loss curves are logged per-epoch to W&B

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Input sequence | 168 hours (1 week) |
| Forecast horizon | 24 hours (1 day ahead) |
| Train/Val/Test split | 80%/10%/10% (chronological) |
| Batch size | 32 |
| Learning rate | 0.005 |
| Max epochs | 50 |
| Early stopping patience | 15 |
| Optimizer | Adam |
| Loss function | MSE |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Qubits | 4 |
| VQC layers | 2 |
| Kernel size (decomposition) | 25 |