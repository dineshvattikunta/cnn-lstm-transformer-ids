import torch
import torch.nn as nn

class BaselineLSTMModule(nn.Module):
    """
    Temporal Sequence Modeling (LSTM) following Algorithm 1 from paper:
    LSTM1 (64) -> LSTM2 (64) -> Bidirectional LSTM (32 per direction = 64).
    """
    def __init__(self, in_features: int = 128, hidden_dim: int = 64, bi_dim: int = 32):
        super(BaselineLSTMModule, self).__init__()
        
        # Unidirectional LSTM stage 1 & 2
        self.lstm1 = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Bidirectional stage: 32 in each direction = 64 total
        self.bilstm = nn.LSTM(input_size=hidden_dim, hidden_size=bi_dim, batch_first=True, bidirectional=True)

    def forward(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 128) from CNN
        Output: (B, 64) temporal representation
        """
        # Reshape to sequence (B, 1, 128)
        x = cnn_feat.unsqueeze(1)
        
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out3, _ = self.bilstm(out2)
        
        # Flatten last sequence step: (B, 1, 64) -> (B, 64)
        lstm_feat = out3.squeeze(1)
        return lstm_feat


class EnhancedBiLSTMModule(nn.Module):
    """
    User Innovation: Advanced Multi-layer Bidirectional LSTM (Bi-LSTM)
    with Layer Normalization, Residual Projection, and Dropout.
    Captures both forward and backward temporal attack dynamics (e.g. multi-stage probes).
    """
    def __init__(self, in_features: int = 128, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(EnhancedBiLSTMModule, self).__init__()
        
        self.bilstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Bidirectional output dimension is hidden_dim * 2 (128)
        bi_out_dim = hidden_dim * 2
        self.layer_norm = nn.LayerNorm(bi_out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection from in_features (128) to bi_out_dim (128)
        self.residual_proj = nn.Linear(in_features, bi_out_dim) if in_features != bi_out_dim else nn.Identity()

    def forward(self, cnn_feat: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 128)
        Output: (B, 128) enhanced bidirectional temporal features
        """
        x = cnn_feat.unsqueeze(1) # (B, 1, 128)
        out, _ = self.bilstm(x)   # (B, 1, 128)
        
        # Residual connection + LayerNorm
        res = self.residual_proj(x)
        out = self.layer_norm(out + res)
        out = self.dropout(out)
        
        lstm_feat = out.squeeze(1) # (B, 128)
        return lstm_feat

