"""
Sentiment Encoder - Dedicated MLP for processing sentiment features.
"""

import torch
import torch.nn as nn

class SentimentEncoder(nn.Module):
    """
    Dedicated encoder for sentiment features.
    
    Input: [Batch, Sequence=1, Sent_Dim=5]
      (Using point-in-time sequence length of 1 for context sentiment features)
      
    Features:
      - sentiment_mean
      - sentiment_count
      - sentiment_std
      - sentiment_rolling_5d
      - sentiment_momentum
      - sentiment_spike
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sentiment features (B, input_dim)
        Returns:
            Encoded sentiment (B, hidden_dim)
        """
        return self.net(x)
