import os
import sys
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_loader import IDSDataLoader
from src.data.preprocess import ATTACK_CLASSES_2025
from src.models.enhanced_ids_model import EnhancedBiLSTMMultiHeadIDS
from src.training.trainer import IDSTrainer
from src.training.evaluate import IDSEvaluator

def main():
    config_path = "configs/config_enhanced.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  PHASE 2: TRAINING ENHANCED BI-LSTM MULTI-HEAD ATTENTION IDS")
    print("  Dataset: Modern 2025 Benchmark | Architecture: User Innovations")
    print("=" * 60)

    # 1. Load Data
    data_loader = IDSDataLoader(config=config, target_classes=ATTACK_CLASSES_2025, dataset_type="2025")
    train_loader, val_loader, test_loader, preprocessor = data_loader.get_dataloaders()
    
    # Save fitted preprocessor
    preprocessor_path = "saved_models/enhanced_preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Fitted enhanced preprocessor saved to {preprocessor_path}")

    # 2. Build Enhanced Model
    model = EnhancedBiLSTMMultiHeadIDS(
        num_features=len(preprocessor.feature_names),
        num_classes=len(preprocessor.target_classes),
        cnn_channels=config["model"]["cnn"]["filters"],
        bilstm_hidden=config["model"]["bilstm"]["hidden_size"],
        bilstm_layers=config["model"]["bilstm"]["num_layers"],
        attn_embed=config["model"]["multi_head_attention"]["embed_dim"],
        attn_heads=config["model"]["multi_head_attention"]["num_heads"],
        attn_ff=config["model"]["multi_head_attention"]["ff_dim"],
        attn_layers=config["model"]["multi_head_attention"]["num_layers"],
        dropout=config["model"]["multi_head_attention"]["dropout"]
    )

    # 3. Train
    trainer = IDSTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    save_path, history = trainer.train(model_filename="enhanced_bilstm_mha_2025.pt")

    # 4. Evaluate
    evaluator = IDSEvaluator(
        model=model,
        test_loader=test_loader,
        target_classes=preprocessor.target_classes,
        results_dir=config["training"]["results_dir"]
    )
    evaluator.evaluate(model_name="Enhanced_2025_Benchmark")
    evaluator.plot_training_curves(history, model_name="Enhanced_2025_Benchmark", results_dir=config["training"]["results_dir"])

if __name__ == "__main__":
    main()

