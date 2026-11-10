import torch
import torch.nn as nn
from .cnn_module import SpatialCNNModule
from .lstm_module import EnhancedBiLSTMModule
from .attention_module import EnhancedMultiHeadAttentionModule

class EnhancedBiLSTMMultiHeadIDS(nn.Module):
    """
    User Innovation: Advanced CNN-BiLSTM-MultiHeadAttention IDS
    Optimized for modern network traffic & 2025 threat benchmarks.
    
    Upgrades:
    - Dedicated Bidirectional LSTM capturing forward & reverse probe sequences
    - Custom Multi-Head Attention with GELU and adaptive layer normalization
    - Resilient dense classification head with dropout regularization
    """
    def __init__(
        self,
        num_features: int = 78,
        num_classes: int = 15,
        cnn_channels: int = 128,
        bilstm_hidden: int = 64,
        bilstm_layers: int = 2,
        attn_embed: int = 512,
        attn_heads: int = 8,
        attn_ff: int = 2048,
        attn_layers: int = 6,
        dropout: float = 0.15
    ):
        super(EnhancedBiLSTMMultiHeadIDS, self).__init__()
        
        # 1. Spatial CNN
        self.cnn = SpatialCNNModule(
            in_features=num_features,
            out_channels=cnn_channels,
            kernel_size=3,
            dropout=0.2
        )
        
        # 2. Enhanced Bidirectional LSTM (hidden_dim * 2 = 128)
        self.bilstm = EnhancedBiLSTMModule(
            in_features=cnn_channels,
            hidden_dim=bilstm_hidden,
            num_layers=bilstm_layers,
            dropout=0.2
        )
        bilstm_out_dim = bilstm_hidden * 2 # 128
        
        # 3. Enhanced Multi-Head Attention
        self.attention = EnhancedMultiHeadAttentionModule(
            in_dim=bilstm_out_dim,
            embed_dim=attn_embed,
            num_heads=attn_heads,
            ff_dim=attn_ff,
            num_layers=attn_layers,
            dropout=dropout,
            use_learnable_temperature=True
        )
        
        # 4. Feature Fusion & Enhanced Classifier Head
        fused_dim = bilstm_out_dim + attn_embed # 128 + 512 = 640
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor):
        cnn_feat = self.cnn(x)              # (B, 128)
        bilstm_feat = self.bilstm(cnn_feat) # (B, 128)
        attn_feat = self.attention(bilstm_feat) # (B, 512)
        
        fused_feat = torch.cat([bilstm_feat, attn_feat], dim=-1) # (B, 640)
        logits = self.classifier(fused_feat)                     # (B, num_classes)
        
        return logits, {
            "cnn_features": cnn_feat,
            "bilstm_features": bilstm_feat,
            "attention_features": attn_feat,
            "fused_features": fused_feat
        }

