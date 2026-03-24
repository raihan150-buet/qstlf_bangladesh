import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils.metrics import compute_metrics


class Trainer:
    def __init__(self, model, config, save_dir, run_name="model"):
        self.model = model
        self.config = config
        self.save_dir = save_dir
        self.run_name = run_name
        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.patience_counter = 0
        self.early_stop_patience = config.get("early_stop_patience", 15)

        self.train_losses = []
        self.val_losses = []

    def _extract_target(self, model_output):
        if model_output.dim() == 3:
            return model_output[:, :, -1]
        return model_output

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(self.device), by.to(self.device)
            self.optimizer.zero_grad()
            out = self._extract_target(self.model(bx))
            loss = self.criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = self._extract_target(self.model(bx))
                total_loss += self.criterion(out, by).item()
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, wandb_run=None):
        print(f"\nTraining {self.run_name} for up to {self.config['epochs']} epochs...")
        print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(self.config["epochs"]):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            elapsed = time.time() - t0

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "epoch_time_s": elapsed,
                })

            improved = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                improved = " *"
            else:
                self.patience_counter += 1

            print(f"  Epoch {epoch+1:03d}/{self.config['epochs']}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={lr:.6f}  ({elapsed:.1f}s){improved}")

            if self.patience_counter >= self.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {self.early_stop_patience} epochs)")
                break

        self.model.load_state_dict(self.best_state)
        print(f"  Best epoch: {self.best_epoch}, best val_loss: {self.best_val_loss:.6f}")
        return self.best_val_loss

    def evaluate(self, test_loader, scaler_y):
        self.model.eval()
        preds_list, actuals_list = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(self.device)
                out = self._extract_target(self.model(bx))
                preds_list.append(out.cpu().numpy())
                actuals_list.append(by.numpy())

        preds = np.vstack(preds_list)
        actuals = np.vstack(actuals_list)

        preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
        actuals_inv = scaler_y.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)

        metrics = compute_metrics(actuals_inv.flatten(), preds_inv.flatten())
        return metrics, preds_inv, actuals_inv

    def save_checkpoint(self, extra_info=None):
        ckpt = {
            "model_state": self.best_state,
            "optimizer_state": self.optimizer.state_dict(),
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        if extra_info:
            ckpt.update(extra_info)

        path = os.path.join(self.save_dir, "checkpoints", f"{self.run_name}_best.pth")
        torch.save(ckpt, path)
        print(f"  Checkpoint saved: {path}")
        return path

    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
