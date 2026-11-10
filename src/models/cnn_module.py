import torch
import torch.nn as nn

class SpatialCNNModule(nn.Module):
    """
    Spatial Feature Extraction Module (CNN) from Paper Section 3.1 & Algorithm 1.
    Extracts local correlations across network packet attributes (ports, lengths, flags).
    """
    def __init__(self, in_features: int = 78, out_channels: int = 128, kernel_size: int = 3, dropout: float = 0.0):
        super(SpatialCNNModule, self).__init__()
        
        # Conv1: same padding
        padding_same = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding_same
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv2: valid padding
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, D) feature vector
        Output: (B, 128) spatial feature representation
        """
        # Reshape to (Batch, Channels=1, Length=D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Global Average Pooling across length dimension
        x = self.global_avg_pool(x) # (B, 128, 1)
        cnn_feat = x.squeeze(-1)    # (B, 128)
        return cnn_feat

