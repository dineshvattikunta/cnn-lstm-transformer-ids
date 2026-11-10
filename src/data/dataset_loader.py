import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from .preprocess import DataPreprocessor, ATTACK_CLASSES_2017, ATTACK_CLASSES_2025
from .sample_generator import generate_sample_dataset

class IDSDataset(Dataset):
    """PyTorch Dataset wrapper for Network Intrusion Detection traffic."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class IDSDataLoader:
    """Manages data loading, preprocessing, and batch creation."""
    def __init__(
        self,
        config: dict,
        target_classes=None,
        dataset_type: str = "2017"
    ):
        self.config = config
        self.dataset_type = dataset_type
        self.target_classes = target_classes or (
            ATTACK_CLASSES_2017 if dataset_type == "2017" else ATTACK_CLASSES_2025
        )
        self.preprocessor = DataPreprocessor(target_classes=self.target_classes)

    def load_data(self) -> pd.DataFrame:
        """Loads data from raw_dir if CSV files exist, otherwise falls back to sample CSV."""
        ds_cfg = self.config["dataset"]
        raw_dir = ds_cfg.get("raw_dir", "data/raw")
        sample_file = ds_cfg.get("sample_file", "data/sample/cicids2017_sample.csv")

        # Check for user-provided CSVs in data/raw
        csv_files = []
        if os.path.exists(raw_dir):
            csv_files = [
                os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                if f.endswith(".csv") and not f.startswith(".")
            ]

        if csv_files:
            print(f"Loading {len(csv_files)} raw CSV file(s) from {raw_dir}...")
            dfs = []
            for f in csv_files:
                print(f"  Reading: {os.path.basename(f)}")
                sub_df = pd.read_csv(f, low_memory=False, encoding="cp1252")
                
                # Identify label column
                lbl = None
                for candidate in ["Label", "label", " Label", "newLabel", "new_label"]:
                    if candidate in sub_df.columns:
                        lbl = candidate
                        break
                        
                if lbl and len(sub_df) > 30000:
                    is_benign = sub_df[lbl].astype(str).str.strip().str.upper().isin(["BENIGN", "NORMAL"])
                    attacks_df = sub_df[~is_benign]
                    benign_df = sub_df[is_benign]
                    # Keep all attacks, sample benign to prevent OOM
                    max_benign = 25000
                    if len(benign_df) > max_benign:
                        benign_df = benign_df.sample(n=max_benign, random_state=42)
                    sub_df = pd.concat([attacks_df, benign_df], ignore_index=True)
                    
                dfs.append(sub_df)
            df = pd.concat(dfs, ignore_index=True)
            print(f"Combined total dataset flows: {len(df)}")
        else:
            if not os.path.exists(sample_file):
                print(f"No raw files found in {raw_dir}. Generating sample dataset at {sample_file}...")
                generate_sample_dataset(
                    sample_file,
                    n_samples=6000,
                    classes=self.target_classes,
                    dataset_type=self.dataset_type
                )
            print(f"Loading sample dataset: {sample_file}")
            df = pd.read_csv(sample_file, low_memory=False)

        return df

    def get_dataloaders(self):
        """Preprocesses dataset and returns PyTorch train, val, and test dataloaders."""
        df = self.load_data()
        X, y = self.preprocessor.fit_transform(df)

        test_size = self.config["dataset"].get("test_size", 0.2)
        val_size = self.config["dataset"].get("val_size", 0.1)
        batch_size = self.config["training"].get("batch_size", 32)
        random_state = self.config["dataset"].get("random_state", 42)

        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Split train_val into train and val
        val_ratio_adjusted = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=random_state, stratify=y_train_val
        )

        # Create PyTorch datasets
        train_dataset = IDSDataset(X_train, y_train)
        val_dataset = IDSDataset(X_val, y_val)
        test_dataset = IDSDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, self.preprocessor

