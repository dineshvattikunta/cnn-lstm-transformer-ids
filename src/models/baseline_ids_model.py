import torch
import torch.nn as nn
from .cnn_module import SpatialCNNModule
from .lstm_module import BaselineLSTMModule
from .attention_module import BaselineTransformerModule

class BaselineHybridIDS(nn.Module):
    """
    Original Research Paper Architecture:
    CNN-LSTM-Transformer for Network Intrusion Detection (Algorithm 1)
    
    1. Spatial Feature Extraction (CNN)
    2. Temporal Dependencies Modeling (LSTM)
    3. Global Context Multi-Head Attention (Transformer)
    4. Feature Fusion (Concatenate LSTM + Transformer) & Softmax Classification
    """
    def __init__(
        self,
        num_features: int = 78,
        num_classes: int = 15,
        cnn_channels: int = 128,
        lstm_hidden: int = 64,
        bilstm_dim: int = 32,
        trans_embed: int = 512,
        trans_heads: int = 8,
        trans_ff: int = 2048,
        trans_layers: int = 6,
        dropout: float = 0.1
    ):
        super(BaselineHybridIDS, self).__init__()
        
        # 1. CNN Module (Spatial)
        self.cnn = SpatialCNNModule(
            in_features=num_features,
            out_channels=cnn_channels,
            kernel_size=3,
            dropout=0.0
        )
        
        # 2. LSTM Module (Temporal)
        self.lstm = BaselineLSTMModule(
            in_features=cnn_channels,
            hidden_dim=lstm_hidden,
            bi_dim=bilstm_dim
        )
        lstm_out_dim = bilstm_dim * 2 # 64
        
        # 3. Transformer Module (Global Context)
        self.transformer = BaselineTransformerModule(
            in_dim=lstm_out_dim,
            embed_dim=trans_embed,
            num_heads=trans_heads,
            ff_dim=trans_ff,
            num_layers=trans_layers,
            dropout=dropout
        )
        
        # 4. Feature Fusion & Classification
        # Fused_feat = Concatenate(LSTM_feat, Trans_feat) -> 64 + 512 = 576
        fused_dim = lstm_out_dim + trans_embed
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # 1. CNN Feature extraction
        cnn_feat = self.cnn(x) # (B, 128)
        
        # 2. LSTM Temporal analysis
        lstm_feat = self.lstm(cnn_feat) # (B, 64)
        
        # 3. Transformer global context analysis
        trans_feat = self.transformer(lstm_feat) # (B, 512)
        
        # 4. Concatenation & Classification
        fused_feat = torch.cat([lstm_feat, trans_feat], dim=-1) # (B, 576)
        logits = self.classifier(fused_feat) # (B, num_classes)
        
        return logits, {
            "cnn_features": cnn_feat,
            "lstm_features": lstm_feat,
            "transformer_features": trans_feat,
            "fused_features": fused_feat
        }

