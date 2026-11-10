# Deep Learning-Based Intrusion Detection System (Aegis IDS)
> **A CNN-LSTM-Transformer Approach for Enhanced Network Security**  
> *ACM ICCSIE Reference Implementation with Modern 2025 Upgrades & Apple iOS Glassmorphic UI*

---

## 🌟 Overview

Modern cyberattacks use polymorphic evasion and multi-stage campaigns that render traditional signature-based firewalls and shallow machine learning models ineffective.

This repository implements the **Hybrid Deep Learning Intrusion Detection Architecture** proposed in research, combining:
1. **1D-CNN (Spatial Extraction):** Identifies localized traffic patterns across port numbers, packet lengths, and TCP/UDP header flags.
2. **LSTM / Bi-LSTM (Temporal Dynamics):** Captures short-term sequential burst velocities, packet inter-arrival times (IAT), and DoS/DDoS flow periodicities.
3. **Transformer Multi-Head Self-Attention (Global Context):** Connects disparate, low-volume reconnaissance probes to delayed exploitation actions across wide time windows.

In addition to the baseline paper implementation, this repository includes **User Innovations**:
* **Enhanced Bidirectional LSTM (Bi-LSTM)** with residual projections and LayerNorm.
* **Optimized Multi-Head Attention** with learnable temperature scaling and GELU activations.
* **Modern 2025 Threat Benchmark Adapter** (Zero-day C2, API abuse, Mirai-variant IoT botnets, DNS tunneling).
* **Apple iOS-Style Dashboard** inspired by modern SaaS layouts (SF Pro typography, frosted glassmorphism, live traffic stream inspector).

---

## 📂 Project Directory Structure

```text
e:\NDL\
│
├── configs/                      # Configuration files (YAML)
│   ├── config_baseline.yaml       # Paper baseline: CNN + LSTM + Transformer (CICIDS2017)
│   └── config_enhanced.yaml       # Enhanced: CNN + Bi-LSTM + Multi-Head Attention (2025 Benchmark)
│
├── data/                         # Dataset repository
│   ├── raw/                      # Place real CICIDS2017 / 2025 CSV files here
│   ├── processed/                # Scaled and split numpy/dataframe caches
│   └── sample/                   # Realistic ready-to-run sample CSV datasets
│       ├── cicids2017_sample.csv
│       └── modern_ids_2025_sample.csv
│
├── frontend/                     # Apple iOS Glassmorphic Web Dashboard
│   ├── index.html                # Modern UI with Doss-style card layout
│   ├── style.css                 # SF Pro / Inter typography, OLED dark mode, glows
│   └── app.js                    # Live packet simulation, model switcher, drag-and-drop CSV
│
├── saved_models/                 # Model checkpoints (.pt) and fitted scalers (.joblib)
├── results/                      # Output confusion matrices, loss curves, metrics JSON
│
├── src/                          # Modular source code
│   ├── api/                      # Backend web server & REST endpoints
│   │   ├── __init__.py
│   │   └── app.py                # FastAPI server (predictions, stream simulation, CSV upload)
│   │
│   ├── data/                     # Data cleaning, scaling, and generation
│   │   ├── __init__.py
│   │   ├── preprocess.py         # Null/Inf handler, StandardScaler, LabelEncoder
│   │   ├── dataset_loader.py     # PyTorch DataLoader pipeline
│   │   └── sample_generator.py   # Statistical packet flow generator
│   │
│   ├── models/                   # Neural network architectures
│   │   ├── __init__.py
│   │   ├── cnn_module.py         # 1D-Convolution spatial feature extractor
│   │   ├── lstm_module.py        # Baseline LSTM and Enhanced Bi-LSTM modules
│   │   ├── attention_module.py   # Multi-Head Attention & Transformer encoder
│   │   ├── baseline_ids_model.py # Full baseline hybrid architecture (Algorithm 1)
│   │   └── enhanced_ids_model.py # User upgraded Bi-LSTM + MHA architecture
│   │
│   └── training/                 # Model training and validation
│       ├── __init__.py
│       ├── trainer.py            # Training loop with Adam & Early Stopping (patience=3)
│       └── evaluate.py           # Precision, Recall, F1, Accuracy, Confusion Matrix
│
├── scripts/                      # Runnable entry points
│   ├── train_baseline.py         # Train & evaluate original paper baseline
│   ├── train_enhanced.py         # Train & evaluate enhanced 2025 model
│   └── start_server.py           # Launch web dashboard and REST API
│
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Baseline Model (Original Paper)
Trains the 1D-CNN + LSTM + Transformer model on the CICIDS2017 dataset:
```bash
python scripts/train_baseline.py
```
* Generates `saved_models/baseline_cnn_lstm_transformer.pt`
* Saves confusion matrix plot to `results/Baseline_CICIDS2017_confusion_matrix.png`
* Saves accuracy/loss curves to `results/Baseline_CICIDS2017_training_curves.png`

### 3. Train the Enhanced Model (User Innovations)
Trains the Bi-LSTM + Enhanced Multi-Head Attention model on the Modern 2025 benchmark:
```bash
python scripts/train_enhanced.py
```
* Generates `saved_models/enhanced_bilstm_mha_2025.pt`
* Evaluates performance and saves evaluation metrics to `results/`

### 4. Launch the Apple iOS Cyber Dashboard
Start the FastAPI server:
```bash
python scripts/start_server.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 🖥️ Dashboard Capabilities

* **Model Switcher:** Toggle smoothly between **Baseline (CICIDS2017)** and **Enhanced 2025 (Bi-LSTM + MHA)** with iOS-style pill buttons.
* **Live Traffic Surveillance:** Live streaming packet feed with automatic threat verdict (Allowed vs Blocked) and confidence scores.
* **Instant Packet Diagnostics:** Inject custom port numbers, packet counts, and flow durations to observe intermediate neural activations (CNN, Bi-LSTM, Transformer).
* **Batch PCAP/CSV Scanner:** Drag-and-drop CSV traffic files for instant threat classification breakdown.

---

## 📊 Experimental Performance (Paper Benchmarks)

| Model Configuration | Test Accuracy | Test Loss | Overfitting Tendency |
| :--- | :---: | :---: | :---: |
| **Proposed Hybrid (CNN+LSTM+Transformer)** | **99.20%** | **0.0525** | **Low** |
| CNN + LSTM | 98.29% | 0.0401 | Moderate |
| Standalone LSTM | 93.15% | 0.0458 | Moderate |
| Standalone CNN | 88.62% | 0.0353 | High |

---

## 🛡️ License
Released under the MIT License for educational, academic, and research purposes.

