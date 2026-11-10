import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_loader import IDSDataLoader
from src.data.preprocess import ATTACK_CLASSES_2017
from src.models.baseline_ids_model import BaselineHybridIDS
from src.training.trainer import IDSTrainer
from src.training.evaluate import IDSEvaluator

def main():
    config_path = "configs/config_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  PHASE 1: TRAINING BASELINE CNN-LSTM-TRANSFORMER MODEL")
    print("  Dataset: CICIDS2017 | Architecture: Original Paper Baseline")
    print("=" * 60)

    # 1. Load Data
    data_loader = IDSDataLoader(config=config, target_classes=ATTACK_CLASSES_2017, dataset_type="2017")
    train_loader, val_loader, test_loader, preprocessor = data_loader.get_dataloaders()
    
    # Save fitted preprocessor
    preprocessor_path = "saved_models/baseline_preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Fitted preprocessor saved to {preprocessor_path}")

    # 2. Build Baseline Model
    model = BaselineHybridIDS(
        num_features=len(preprocessor.feature_names),
        num_classes=len(preprocessor.target_classes),
        cnn_channels=config["model"]["cnn"]["filters"],
        lstm_hidden=config["model"]["lstm"]["hidden_size"],
        bilstm_dim=32,
        trans_embed=config["model"]["transformer"]["embed_dim"],
        trans_heads=config["model"]["transformer"]["num_heads"],
        trans_ff=config["model"]["transformer"]["ff_dim"],
        trans_layers=config["model"]["transformer"]["num_layers"],
        dropout=config["model"]["transformer"]["dropout"]
    )

    # 3. Train
    trainer = IDSTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    save_path, history = trainer.train(model_filename="baseline_cnn_lstm_transformer.pt")

    # 4. Evaluate
    evaluator = IDSEvaluator(
        model=model,
        test_loader=test_loader,
        target_classes=preprocessor.target_classes,
        results_dir=config["training"]["results_dir"]
    )
    evaluator.evaluate(model_name="Baseline_CICIDS2017")
    evaluator.plot_training_curves(history, model_name="Baseline_CICIDS2017", results_dir=config["training"]["results_dir"])

if __name__ == "__main__":
    main()

