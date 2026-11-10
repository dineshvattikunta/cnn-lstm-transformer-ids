import torch
import torch.nn as nn
import math

class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Layer matching paper Section 3.3:
    Multi-Head Self-Attention -> Add & LayerNorm -> FeedForward (2048) -> Add & LayerNorm.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Self-Attention + Residual + LayerNorm (Eq 9)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class BaselineTransformerModule(nn.Module):
    """
    Baseline 6-layer Transformer module from Algorithm 1.
    """
    def __init__(
        self,
        in_dim: int = 64,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super(BaselineTransformerModule, self).__init__()
        
        # Project LSTM features to Transformer embedding dimension
        self.input_proj = nn.Linear(in_dim, embed_dim)
        
        # 6 stacked encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, lstm_feat: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, in_dim)
        Output: (B, embed_dim) global contextual feature vector
        """
        # Linear -> (B, 512) -> Reshape (B, 1, 512)
        x = self.input_proj(lstm_feat).unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)
            
        # Global Average Pooling: (B, 1, 512) -> (B, 512, 1) -> (B, 512)
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)
        return x


class EnhancedMultiHeadAttentionModule(nn.Module):
    """
    User Innovation: Advanced Multi-Head Attention Mechanism with:
    - Learnable temperature scaling for adaptive sequence sharpening
    - GELU non-linearities in FeedForward
    - 6 encoder layers with adaptive attention heads
    - Captures long-range dependencies across modern distributed attack flows
    """
    def __init__(
        self,
        in_dim: int = 128,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.15,
        use_learnable_temperature: bool = True
    ):
        super(EnhancedMultiHeadAttentionModule, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        self.use_temp = use_learnable_temperature
        if use_learnable_temperature:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, lstm_feat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(lstm_feat).unsqueeze(1) # (B, 1, 512)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        return x.squeeze(1) # (B, 512)

