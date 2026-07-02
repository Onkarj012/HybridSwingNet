"""
Time2Vec: Learnable Positional Encoding.
Paper: https://arxiv.org/abs/1907.05321
"""

import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """
    Time2Vec - Learnable Vector Representation of Time.
    
    Captures both periodic and non-periodic patterns in the time dimension.
    
    Formula:
      v[0] = w0 * tau + phi0  (Linear / Non-periodic)
      v[i] = sin(wi * tau + phii)  (Periodic for i > 0)
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
        """
        Args:
            input_dim: Dimension of input (usually sequence length or feature dim if applying per feature)
                       Here we assume it's applying to the *time* dimension, but in practice
                       for LSTMs we often concatenate this to the features.
                       Actually, Time2Vec usually takes the scalar 'time' (index) and produces a vector.
            embed_dim: Dimension of the output time embedding (k+1 in paper)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear component parameters (w0, phi0)
        self.w0 = nn.Parameter(torch.randn(input_dim, 1)) 
        self.phi0 = nn.Parameter(torch.randn(input_dim, 1))
        
        # Periodic component parameters (wi, phii)
        self.w = nn.Parameter(torch.randn(input_dim, embed_dim - 1))
        self.phi = nn.Parameter(torch.randn(input_dim, embed_dim - 1))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor. 
               If x is indices (B, L), we embed 'L' dimension.
               If x is raw features (B, L, D), we might want to preserve structure.
               
               Standard usage for Transformers/LSTMs:
               Input is often just the time steps indices [0, 1, ..., L-1].
        
        Let's assume efficient usage:
        We create a standard Time2Vec that accepts (B, L, 1) or (B, L) where values are time indices, 
        OR we apply it as a layer that *adds* time information to features.
        
        For this implementation:
        Input: (B, L, input_dim) - ACTUALLY, usually we concatenate T2V output to features.
        But T2V itself computes on 'tau' (time).
        """
        # Linear term: (x * w0 + phi0)
        # We assume x is (B, L, input_dim) to check if we act on features or time.
        # But commonly for "Positional Encoding", we just pass the time index.
        # Let's support passing the actual features `x` and we learn a representation *of the features* with Sine activation?
        # NO, Time2Vec is for TIME.
        
        # Simplified Implementation:
        # We will assume `x` is the input features (B, L, F).
        # We want to add Time embeddings.
        # We generate time indices (0..L-1) on the fly, embed them, and concatenate.
        
        raise NotImplementedError("Use Time2VecLayer instead for easy integration")

class Time2VecLayer(nn.Module):
    """
    Easy wrapper to add Time2Vec embeddings to input features.
    
    Standard Input: (B, L, F)
    Output: (B, L, F + embed_dim)
    """
    def __init__(self, sequence_length: int, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Model time as local relative position (0 to 1) or indices
        # We will use indices 0..L-1
        self.w0 = nn.Parameter(torch.randn(embed_dim, 1)) # Weights for linear
        self.phi0 = nn.Parameter(torch.randn(embed_dim, 1)) # Bias for linear
        
        self.w = nn.Parameter(torch.randn(embed_dim, embed_dim-1)) # Weights for periodic
        self.phi = nn.Parameter(torch.randn(embed_dim, embed_dim-1)) # Bias for periodic
        
        # Correction: The paper defines T2V for a scalar time \tau.
        # We want a vector output of size `embed_dim` for each time step.
        # So w0 is shape (1,), phi0 is (1,) to produce 1 linear feature? 
        # Actually the linear part is usually 1 dimension, periodic is k. Total k+1.
        
        # Let's strictly follow: 
        # v[0] = w0 * tau + phi0
        # v[i] = sin(wi * tau + phii)
        
        # Parameters for the linear component (output dim 1)
        self.w0 = nn.Parameter(torch.randn(1))
        self.phi0 = nn.Parameter(torch.randn(1))
        
        # Parameters for the periodic component (output dim k-1)
        self.k = embed_dim - 1
        self.w = nn.Parameter(torch.randn(self.k))
        self.phi = nn.Parameter(torch.randn(self.k))
        
    def forward(self, x):
        """
        Args:
            x: (B, L, F)
        Returns:
            (B, L, F + embed_dim)
        """
        B, L, F = x.shape
        device = x.device
        
        # Generate time indices
        # tau: (L) -> (B, L, 1)
        tau = torch.arange(L, dtype=torch.float32, device=device).view(1, L, 1)
        tau = tau.expand(B, -1, -1)
        
        # Linear component: w0 * tau + phi0
        # (B, L, 1) * (1) + (1) -> (B, L, 1)
        linear = self.w0 * tau + self.phi0
        
        # Periodic component: sin(w * tau + phi)
        # w: (k) -> (1, 1, k)
        # tau: (B, L, 1)
        # Result: (B, L, k)
        periodic = torch.sin(tau * self.w.view(1, 1, -1) + self.phi.view(1, 1, -1))
        
        # Concatenate time embedding
        time_embedding = torch.cat([linear, periodic], dim=-1) # (B, L, embed_dim)
        
        # Concatenate with input features
        out = torch.cat([x, time_embedding], dim=-1) # (B, L, F + embed_dim)
        
        return out
