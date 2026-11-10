"""
Quick script to display model evaluation metrics and training accuracy in the terminal.
Usage:
    python scripts/show_metrics.py
"""
import os
import json

def display_metrics():
    baseline_path = os.path.join("results", "Baseline_CICIDS2017_metrics.json")
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found.")
        return

    with open(baseline_path, "r") as f:
        data = json.load(f)

    acc = data.get("accuracy", 0) * 100
    prec = data.get("precision", 0) * 100
    rec = data.get("recall", 0) * 100
    f1 = data.get("f1_score", 0) * 100
    support = int(data.get("per_class_metrics", {}).get("weighted avg", {}).get("support", 151530))

    print("\n" + "=" * 65)
    print("   CNN-LSTM-TRANSFORMER INTRUSION DETECTION SYSTEM")
    print("   OFFICIAL EXPERIMENTAL EVALUATION RESULTS (CICIDS2017)")
    print("=" * 65)
    print(f"  • Total Test Flows Evaluated : {support:,} flows (20% unseen test split)")
    print(f"  • Overall Test Accuracy      : {acc:.2f}%")
    print(f"  • Macro / Weighted Precision : {prec:.2f}%")
    print(f"  • Macro / Weighted Recall    : {rec:.2f}%")
    print(f"  • Macro / Weighted F1-Score  : {f1:.2f}%")
    print("-" * 65)
    print("   PER-ATTACK CLASS HIGHLIGHTS:")
    print("  -------------------------------------------------------------")
    per_class = data.get("per_class_metrics", {})
    highlight_classes = ["BENIGN", "DDoS", "PortScan", "DoS Hulk", "FTP-Patator", "SSH-Patator", "Heartbleed"]
    print(f"  {'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("  " + "-" * 57)
    for c in highlight_classes:
        if c in per_class:
            m = per_class[c]
            p = m.get("precision", 0) * 100
            r = m.get("recall", 0) * 100
            f = m.get("f1-score", 0) * 100
            print(f"  {c:<20} | {p:>8.2f}% | {r:>8.2f}% | {f:>8.2f}%")
    print("=" * 65)
    print("\n[Visual Artifacts Ready for Professor]:")
    print("  1. Training Curves Plot : results/Baseline_CICIDS2017_training_curves.png")
    print("  2. Confusion Matrix     : results/Baseline_CICIDS2017_confusion_matrix.png\n")

if __name__ == "__main__":
    display_metrics()

