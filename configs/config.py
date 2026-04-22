import os
import wandb
from dotenv import load_dotenv

load_dotenv()

SEED = 42

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
else:
    wandb.login()

DATA_PATH = "selected_features.xlsx"

SEQ_LENGTH = 168
FORECAST_HORIZON = 24
BATCH_SIZE = 32
LEARNING_RATE = 0.005
EPOCHS = 50
KERNEL_SIZE = 25
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # from the training portion

# quantum params
N_QUBITS = 4
N_QLAYERS = 2
CONTEXT_SIZE = 32

# LAQ lag positions (identified via PACF analysis on training data)
# short-term: 1-6, daily cycle: 23-25, weekly cycle: 167-168
LAQ_LAGS = [1, 2, 3, 4, 5, 6, 23, 24, 25, 167, 168]

# wandb
WANDB_PROJECT = "STLF_SOTA_Vs_QDLinear"
WANDB_ENTITY = None  # set your wandb username or team name

# output dirs
BASE_DIR = "outputs"
CLASSICAL_DIR = os.path.join(BASE_DIR, "classical_dlinear")
QUANTUM_DIR = os.path.join(BASE_DIR, "quantum_adqrl")
ABLATION_DIR = os.path.join(BASE_DIR, "ablation")
COMPARISON_DIR = os.path.join(BASE_DIR, "comparison")

for d in [CLASSICAL_DIR, QUANTUM_DIR, ABLATION_DIR, COMPARISON_DIR]:
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)