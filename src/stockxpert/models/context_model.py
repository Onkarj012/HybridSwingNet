
import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    """
    Encodes market-wide context features (e.g. NIFTY50 returns, volatility, VIX)
    into the shared hidden dimension.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()  # End with activation to match other encoders
        )
        
    def forward(self, x):
        """
        x: (B, input_dim)
        """
        return self.net(x)
