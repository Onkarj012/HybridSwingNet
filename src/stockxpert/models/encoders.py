"""
Multi-scale encoders: ResNet1D, BiGRU, BiLSTM, Context MLP.
"""

import torch
import torch.nn as nn


class ResNet1DEncoder(nn.Module):
    """
    ResNet1D encoder for short window.
    Input: (B, window, features) -> Output: (B, hidden_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.block1 = self._make_residual_block(hidden_dim, dropout)
        self.block2 = self._make_residual_block(hidden_dim, dropout)
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
    def _make_residual_block(self, dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, window, input_dim)
        Returns:
            (B, hidden_dim)
        """
        # Project to hidden dim
        x = self.input_proj(x)  # (B, window, hidden_dim)
        
        # Residual block 1
        residual = x
        out = self.block1(x)
        x = torch.relu(out + residual)
        
        # Residual block 2
        residual = x
        out = self.block2(x)
        x = torch.relu(out + residual)
        
        # Global pooling: (B, window, hidden_dim) -> (B, hidden_dim, window) -> (B, hidden_dim, 1)
        x = x.transpose(1, 2)  # (B, hidden_dim, window)
        x = self.pool(x)  # (B, hidden_dim, 1)
        x = x.squeeze(-1)  # (B, hidden_dim)
        

        return x


class ResNLSEncoder(nn.Module):
    """
    ResNLS (ResNet + LSTM) hybrid encoder for short window.
    Combines point-wise residual processing with LSTM temporal modeling.
    
    Structure:
    Input -> Projection -> ResBlock -> ResBlock -> BiLSTM -> Final State
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks (Point-wise processing)
        self.block1 = self._make_residual_block(hidden_dim, dropout)
        self.block2 = self._make_residual_block(hidden_dim, dropout)
        
        # LSTM for temporal mixing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2, # Bidirectional = hidden_dim total
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
    def _make_residual_block(self, dim: int, dropout: float):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, window, input_dim)
        Returns:
            (B, hidden_dim)
        """
        # 1. Project to hidden dim
        x = self.input_proj(x)  # (B, window, hidden_dim)
        
        # 2. Residual blocks (preserving time dimension)
        residual = x
        out = self.block1(x)
        x = torch.relu(out + residual)
        
        residual = x
        out = self.block2(x)
        x = torch.relu(out + residual)
        
        # 3. LSTM Temporal Processing
        # x: (B, window, hidden_dim)
        _, (hidden, _) = self.lstm(x)
        
        # hidden: (2, B, hidden_dim/2) -> (B, hidden_dim)
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        out = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        return out


class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU encoder for mid window.
    Input: (B, window, features) -> Output: (B, hidden_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles the output
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, window, input_dim)
        Returns:
            (B, hidden_dim)
        """
        # GRU forward
        output, hidden = self.gru(x)
        # output: (B, window, hidden_dim)
        # hidden: (num_layers * 2, B, hidden_dim // 2)
        
        # Take final hidden states from both directions
        # Concatenate forward and backward final hidden states from last layer
        hidden_fwd = hidden[-2]  # Last layer, forward direction
        hidden_bwd = hidden[-1]  # Last layer, backward direction
        
        # Concatenate: (B, hidden_dim // 2) + (B, hidden_dim // 2) = (B, hidden_dim)
        out = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Norm
        out = self.layer_norm(out)
        
        return out


from .time2vec import Time2VecLayer

from .time2vec import Time2VecLayer

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for long window.
    Enhanced with Time2Vec positional encoding.
    Input: (B, window, features) -> Output: (B, hidden_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, num_layers: int = 3, dropout: float = 0.15):
        super().__init__()
        
        # Time2Vec Embedding
        self.t2v_dim = 16
        self.time2vec = Time2VecLayer(sequence_length=1, embed_dim=self.t2v_dim)
        
        # Input to LSTM is original features + time embedding
        lstm_input_dim = input_dim + self.t2v_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles the output
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, window, input_dim)
        Returns:
            (B, hidden_dim)
        """
        # Apply Time2Vec encoding
        # x -> (B, window, input_dim + 16)
        x = self.time2vec(x)
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(x)
        
        # Take final hidden states from both directions
        hidden_fwd = hidden[-2]  # Last layer, forward direction
        hidden_bwd = hidden[-1]  # Last layer, backward direction
        
        # Concatenate
        out = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Norm
        out = self.layer_norm(out)
        
        return out


class ContextEncoder(nn.Module):
    """
    MLP encoder for context features.
    Input: (B, context_dim) -> Output: (B, hidden_dim)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.15):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, hidden_dim)
        """
        return self.mlp(x)
