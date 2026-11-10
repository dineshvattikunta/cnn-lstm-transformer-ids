import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

class IDSEvaluator:
    """
    Computes precision, recall, f1, accuracy, and confusion matrix
    matching Equations (10)-(13) and Tables 2-3 in the paper.
    """
    def __init__(self, model, test_loader, target_classes, device=None, results_dir="results"):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.target_classes = target_classes
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def evaluate(self, model_name: str = "Hybrid_IDS"):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                logits, _ = self.model(X_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Metrics
        acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
        
        # Per-class report
        class_names = [self.target_classes[i] if i < len(self.target_classes) else f"Class_{i}" for i in range(len(np.unique(np.concatenate([all_targets, all_preds]))))]
        report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        
        cm = confusion_matrix(all_targets, all_preds)

        summary = {
            "model_name": model_name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "per_class_metrics": report_dict
        }

        # Save summary json
        json_path = os.path.join(self.results_dir, f"{model_name}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Plot & save confusion matrix
        self._plot_confusion_matrix(cm, class_names, model_name)

        print(f"\n=================== EVALUATION RESULTS ({model_name}) ===================")
        print(f"Overall Accuracy:  {acc*100:.2f}%")
        print(f"Weighted Precision:{prec*100:.2f}%")
        print(f"Weighted Recall:   {rec*100:.2f}%")
        print(f"Weighted F1-Score: {f1*100:.2f}%")
        print(f"Results saved to:  {self.results_dir}")
        print("=========================================================================\n")

        return summary

    def _plot_confusion_matrix(self, cm, class_names, model_name):
        plt.figure(figsize=(10, 8), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=15)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(plot_path)
        plt.close()

    @staticmethod
    def plot_training_curves(history: dict, model_name: str = "Hybrid_IDS", results_dir: str = "results"):
        """Plot loss and accuracy curves over epochs."""
        os.makedirs(results_dir, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(12, 5), dpi=300)

        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss", color="#3b82f6", linewidth=2)
        plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#ef4444", linewidth=2, linestyle="--")
        plt.title("Loss vs. Epochs", fontsize=13)
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#10b981", linewidth=2)
        plt.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#8b5cf6", linewidth=2, linestyle="--")
        plt.title("Accuracy vs. Epochs (%)", fontsize=13)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"{model_name}_training_curves.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Training curves saved to: {plot_path}")

