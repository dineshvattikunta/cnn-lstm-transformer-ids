import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

ATTACK_CLASSES_2017 = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack Brute Force",
    "Web Attack XSS",
    "Infiltration",
    "Web Attack Sql Injection",
    "Heartbleed"
]

ATTACK_CLASSES_2025 = [
    "BENIGN",
    "Modern DDoS Amplification",
    "Zero-Day C2 Beaconing",
    "IoT Mirai-Variant Botnet",
    "Cloud API Credential Abuse",
    "Ransomware Lateral Movement",
    "Advanced Web SQL Injection",
    "GraphQL Query Depth Attack",
    "Multi-Stage Infiltration",
    "SSH Modern Brute Force",
    "Encrypted Tunnel Exfiltration",
    "DNS Tunneling Exploit",
    "Slowloris v2 Protocol Abuse",
    "Supply-Chain Dependency Exploit",
    "AI Evasion Adversarial Flow"
]

class DataPreprocessor:
    """
    Cleans, normalizes, and encodes network traffic feature vectors
    matching CICIDS2017 and modern 2025 intrusion detection specifications.
    """
    def __init__(self, target_classes=None):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.target_classes = target_classes or ATTACK_CLASSES_2017
        self.feature_names = []
        self.is_fitted = False

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from columns, remove inf/nan values."""
        df = df.copy()
        df.columns = [str(col).strip() for col in df.columns]

        # Identify label column
        label_col = None
        for candidate in ["newLabel", "new_label", "Label", "label", "Attack_Type", "attack_type", "class"]:
            if candidate in df.columns:
                label_col = candidate
                break

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing numeric values with column median
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # If any columns are completely NaN, fill with 0
        df[num_cols] = df[num_cols].fillna(0.0)

        # Drop non-numeric identifier columns if present (e.g. Timestamp, Flow ID, Source IP, Dest IP)
        drop_candidates = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "src_ip", "dst_ip", "time"]
        to_drop = [c for c in drop_candidates if c in df.columns]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)

        return df

    def fit_transform(self, df: pd.DataFrame, label_col: str = None):
        """Fit scaler & encoder on training data, return processed arrays."""
        df_clean = self.clean_dataframe(df)
        
        if label_col is None or label_col not in df_clean.columns:
            for candidate in ["newLabel", "new_label", "Label", "label", "Attack_Type", "attack_type", "class"]:
                if candidate in df_clean.columns:
                    label_col = candidate
                    break
            if label_col is None:
                label_col = df_clean.columns[-1]

        y_raw = df_clean[label_col].astype(str).str.strip()
        X_df = df_clean.drop(columns=[label_col])
        
        # Keep numeric features
        X_df = X_df.select_dtypes(include=[np.number])
        self.feature_names = list(X_df.columns)

        # Scale features
        X = self.scaler.fit_transform(X_df.values.astype(np.float32))

        # Detect dataset classes
        unique_classes = sorted(list(y_raw.unique()))
        if len(unique_classes) > 1 and len(unique_classes) <= 30:
            self.target_classes = unique_classes

        # Fit label encoder on target classes
        self.label_encoder.fit(self.target_classes)
        
        mapped_labels = []
        for val in y_raw:
            if val in self.target_classes:
                mapped_labels.append(val)
            else:
                found = False
                for tc in self.target_classes:
                    if tc.lower() in val.lower() or val.lower() in tc.lower():
                        mapped_labels.append(tc)
                        found = True
                        break
                if not found:
                    mapped_labels.append(self.target_classes[0])

        y = self.label_encoder.transform(mapped_labels)
        self.is_fitted = True

        return X, y

    def transform(self, df: pd.DataFrame, label_col: str = None, return_labels: bool = False):
        """Transform test or live incoming data with fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform()")
            
        df_clean = self.clean_dataframe(df)

        if label_col is None:
            for candidate in ["newLabel", "new_label", "Label", "label", "Attack_Type", "attack_type", "class"]:
                if candidate in df_clean.columns:
                    label_col = candidate
                    break

        has_label = label_col is not None and label_col in df_clean.columns

        y = None
        if has_label:
            y_raw = df_clean[label_col].astype(str).str.strip()
            mapped = [val if val in self.target_classes else self.target_classes[0] for val in y_raw]
            y = self.label_encoder.transform(mapped)
            X_df = df_clean.drop(columns=[label_col])
        else:
            X_df = df_clean

        # Align features
        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[self.feature_names]

        X = self.scaler.transform(X_df.values.astype(np.float32))
        return (X, y) if (has_label and return_labels) else X

    def save(self, filepath: str):
        """Save preprocessor state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "target_classes": self.target_classes,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath: str):
        """Load preprocessor state."""
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.target_classes = data["target_classes"]
        self.feature_names = data["feature_names"]
        self.is_fitted = data["is_fitted"]
        return self

