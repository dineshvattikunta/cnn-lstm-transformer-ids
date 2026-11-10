import torch
import torch.nn as nn
import os
import json
import numpy as np
from tqdm import tqdm

class IDSTrainer:
    """
    Manages model training with Adam optimizer, CrossEntropy loss,
    and Early Stopping (patience=3) as specified in paper Section 4.3.1.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: str = None
    ):
        self.config = config
        self.train_cfg = config.get("training", {})
        
        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Hyperparameters
        lr = self.train_cfg.get("learning_rate", 0.001)
        b1 = self.train_cfg.get("adam_beta1", 0.9)
        b2 = self.train_cfg.get("adam_beta2", 0.999)
        weight_decay = self.train_cfg.get("weight_decay", 0.0)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.patience = self.train_cfg.get("early_stopping_patience", 3)
        self.epochs = self.train_cfg.get("epochs", 50)
        self.save_dir = self.train_cfg.get("save_dir", "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_epoch": 0
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
            
        epoch_loss = total_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)
                
        val_loss = total_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def train(self, model_filename: str = "model.pt", resume: bool = True):
        print(f"Starting training on {self.device} (Max Epochs: {self.epochs}, Patience: {self.patience})")
        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 1
        save_path = os.path.join(self.save_dir, model_filename)

        if resume and os.path.exists(save_path):
            try:
                ckpt = torch.load(save_path, map_location=self.device, weights_only=False)
                if "model_state_dict" in ckpt and "epoch" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    best_val_loss = ckpt.get("val_loss", float("inf"))
                    start_epoch = ckpt["epoch"] + 1
                    print(f"Resumed from checkpoint: Epoch {ckpt['epoch']}, Best Val Loss: {best_val_loss:.4f}, Val Acc: {ckpt.get('val_acc', 0.0)*100:.2f}%")
            except Exception as e:
                print(f"Could not resume from checkpoint ({e}), starting fresh.")

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
            )

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                self.history["best_epoch"] = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": self.config
                }, save_path)
                print(f"  --> Checkpoint saved to {save_path} (Best Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at Epoch {epoch} (No improvement for {self.patience} epochs).")
                    break

        # Save training history
        history_path = os.path.join(self.save_dir, model_filename.replace(".pt", "_history.json"))
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
        return save_path, self.history

