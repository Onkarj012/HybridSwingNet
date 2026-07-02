"""
Main StockXpert model combining all components.

Includes:
- StockXpertModel: Original model
- StockXpertModelV2: Enhanced model with regime detection, probabilistic output, quantile regression
- RegimeDetector: Market regime classification head
- ProbabilisticMagnitudeHead: Predict mean + variance
- QuantileMagnitudeHead: Predict multiple quantiles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .encoders import ResNet1DEncoder, ResNLSEncoder, BiGRUEncoder, BiLSTMEncoder, ContextEncoder
from .sentiment_encoder import SentimentEncoder
from .fusion import AttentionFusion, HorizonMixer


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return F.relu(x + self.block(x))


class RegimeDetector(nn.Module):
    """
    Detects market regime (bull/bear/sideways) to condition predictions.
    
    Regimes:
    - 0: Bear (downtrend)
    - 1: Sideways (range-bound)
    - 2: Bull (uptrend)
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3 regimes
        )
        
    def forward(self, global_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            global_fused: (B, hidden_dim) global representation
        
        Returns:
            regime_logits: (B, 3) raw logits
            regime_probs: (B, 3) softmax probabilities
        """
        regime_logits = self.regime_head(global_fused)
        regime_probs = F.softmax(regime_logits, dim=-1)
        return regime_logits, regime_probs


class ProbabilisticMagnitudeHead(nn.Module):
    """
    Predicts mean AND variance of price change.
    Enables uncertainty-aware trading decisions.
    """
    
    def __init__(self, input_dim: int, num_horizons: int, dropout: float = 0.15):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.mean_head = nn.Linear(input_dim, num_horizons)
        self.log_var_head = nn.Linear(input_dim, num_horizons)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim)
        
        Returns:
            mean: (B, num_horizons)
            std: (B, num_horizons) - positive via softplus
        """
        shared_out = self.shared(x)
        # Bound mean to [-5, +5]
        mean = torch.tanh(self.mean_head(shared_out)) * 5.0
        log_var = self.log_var_head(shared_out)
        std = F.softplus(log_var)  # Ensure positive
        return mean, std


class QuantileMagnitudeHead(nn.Module):
    """
    Predicts multiple quantiles (e.g., 10%, 50%, 90%) for risk assessment.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_horizons: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        dropout: float = 0.15
    ):
        super().__init__()
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # One head per quantile
        self.quantile_heads = nn.ModuleList([
            nn.Linear(input_dim, num_horizons) for _ in quantiles
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, input_dim)
        
        Returns:
            List of (B, num_horizons) tensors, one per quantile
        """
        shared_out = self.shared(x)
        return [head(shared_out) for head in self.quantile_heads]


class StockXpertModel(nn.Module):
    """
    Complete StockXpert model architecture.
    
    Pipeline:
    1. Encode multi-scale inputs separately (ResNet, BiGRU, BiLSTM, MLP)
    2. Fuse using multi-head attention
    3. Add stock embedding
    4. Predict direction (logits) and magnitude (delta log)
    """
    
    def __init__(
        self,
        num_stocks: int,
        short_dim: int,
        mid_dim: int,
        long_dim: int,
        context_dim: int,
        num_horizons: int = 5,
        hidden_dim: int = 96,
        stock_embed_dim: int = 16,
        attn_heads: int = 4,
        dropout: float = 0.15,
        bottleneck_dim: int = 64
    ):
        """
        Args:
            num_stocks: Number of unique stocks for embedding
            short_dim: Number of features in short window
            mid_dim: Number of features in mid window
            long_dim: Number of features in long window
            context_dim: Number of context features
            num_horizons: Number of prediction horizons
            hidden_dim: Hidden dimension size
            stock_embed_dim: Stock embedding dimension
            attn_heads: Number of attention heads
            dropout: Dropout rate
            bottleneck_dim: Dimension of head bottleneck layer
        """
        super().__init__()
        
        # Encoders
        self.short_encoder = ResNLSEncoder(short_dim, hidden_dim, dropout)
        self.mid_encoder = BiGRUEncoder(mid_dim, hidden_dim, dropout=dropout)
        self.long_encoder = BiLSTMEncoder(long_dim, hidden_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(context_dim, hidden_dim, dropout)
        
        # Sentiment Encoder (input dim fixed to 9 as per schema)
        self.sentiment_encoder = SentimentEncoder(9, hidden_dim, dropout)
        
        # Fusion
        self.fusion = AttentionFusion(hidden_dim, attn_heads, dropout, num_horizons)
        
        # Horizon Mixer
        self.mixer = HorizonMixer(hidden_dim, num_horizons, n_layers=2, dropout=dropout)
        
        # Stock embedding
        self.stock_embedding = nn.Embedding(num_stocks, stock_embed_dim)
        
        # Combined dimension after fusion + stock embedding
        combined_dim = hidden_dim + stock_embed_dim
        
        # Prediction heads
        def make_bottleneck_head(out_features=1):
            return nn.Sequential(
                nn.Linear(combined_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                ResidualBlock(bottleneck_dim, dropout),
                nn.Linear(bottleneck_dim, 32),
                nn.ReLU(),
                nn.Linear(32, out_features)
            )
            
        self.direction_heads = nn.ModuleList([make_bottleneck_head() for _ in range(num_horizons)])
        self.magnitude_heads = nn.ModuleList([make_bottleneck_head() for _ in range(num_horizons)])
        self.level_heads = nn.ModuleList([make_bottleneck_head(3) for _ in range(num_horizons)])
        
        # Shared heads for auxiliary tasks
        # Per-horizon heads for auxiliary tasks (formerly shared)
        self.confidence_heads = nn.ModuleList([
            nn.Linear(combined_dim, 1) for _ in range(num_horizons)
        ])
        self.target_zone_heads = nn.ModuleList([
            nn.Linear(combined_dim, 7) for _ in range(num_horizons)
        ])

        
    def forward(self, X_short, X_mid, X_long, X_context, X_sentiment, stock_idx):
        """
        Forward pass.
        
        Args:
            X_short: (B, win_short, short_dim)
            X_mid: (B, win_mid, mid_dim)
            X_long: (B, win_long, long_dim)
            X_context: (B, context_dim)
            X_sentiment: (B, sentiment_dim)
            stock_idx: (B,) - Long tensor of stock indices
        
        Returns:
            direction_logits: (B, num_horizons)
            magnitude_pred: (B, num_horizons)
            confidence: (B, num_horizons)
            target_zone_logits: (B, num_horizons, 7)
            aux_trend_logit: (B, 1) or None
            level_pred: (B, num_horizons, 3)
            attention_dict: Dict with attention weights
        """
        # Encode each scale
        short_enc = self.short_encoder(X_short)
        mid_enc = self.mid_encoder(X_mid)
        long_enc = self.long_encoder(X_long)
        context_enc = self.context_encoder(X_context)
        sent_enc = self.sentiment_encoder(X_sentiment)
        
        # Fuse
        global_fused, fused_per_horizon, attention_dict = self.fusion(
            short_enc, mid_enc, long_enc, context_enc, sent_enc
        )
        
        # Apply Horizon Mixer
        enhanced_horizons, aux_trend_logit = self.mixer(fused_per_horizon)
        fused_per_horizon = enhanced_horizons
        
        # Stock embedding
        stock_emb = self.stock_embedding(stock_idx)
        stock_emb = F.dropout(stock_emb, p=0.1, training=self.training)
        
        # Combinations
        global_combined = torch.cat([global_fused, stock_emb], dim=1)
        
        # Predict
        dir_logits_list = []
        mag_pred_list = []
        level_pred_list = []
        conf_list = []
        tz_list = []
        
        for i in range(len(self.direction_heads)):
            horizon_rep = fused_per_horizon[:, i, :]
            # Phase 5: Graduated feature routing per horizon
            # Each horizon gets a weighted blend of encoder outputs
            # H1(i=0): 70% short + 30% mid
            # H3(i=1): 30% short + 70% mid
            # H5(i=2): 50% mid + 50% long
            # H7(i=3): 30% mid + 70% long
            # H10(i=4): 100% long
            num_h = len(self.direction_heads)
            if num_h == 5:
                blend_weights = [
                    (0.7, 0.3, 0.0),  # H1: short-heavy
                    (0.3, 0.7, 0.0),  # H3: mid-heavy
                    (0.0, 0.5, 0.5),  # H5: mid+long
                    (0.0, 0.3, 0.7),  # H7: long-heavy
                    (0.0, 0.0, 1.0),  # H10: long only
                ]
                w_s, w_m, w_l = blend_weights[i]
                encoder_blend = w_s * short_enc + w_m * mid_enc + w_l * long_enc
            else:
                # Fallback for non-standard horizon counts
                encoder_blend = short_enc if i == 0 else long_enc
            horizon_rep = horizon_rep + encoder_blend

            ctx = torch.cat([horizon_rep, stock_emb], dim=1)

            dir_logits_list.append(self.direction_heads[i](ctx))
            mag_pred_list.append(self.magnitude_heads[i](ctx))
            level_pred_list.append(self.level_heads[i](ctx))
            
            # Phase 3: Per-horizon confidence and zone
            conf_logit = self.confidence_heads[i](ctx) # (B, 1)
            conf_list.append(conf_logit)
            
            tz_logit = self.target_zone_heads[i](ctx) # (B, 7)
            tz_list.append(tz_logit)
            
        direction_logits = torch.cat(dir_logits_list, dim=1)
        # Bound magnitude to [-5, +5] matching target clip range
        magnitude_pred = torch.tanh(torch.cat(mag_pred_list, dim=1)) * 5.0
        level_pred = torch.stack(level_pred_list, dim=1)
        
        # Combine lists
        confidence = torch.sigmoid(torch.cat(conf_list, dim=1)) # (B, H)
        target_zone_logits = torch.stack(tz_list, dim=1) # (B, H, 7)
        
        return direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, attention_dict


class StockXpertModelV2(nn.Module):
    """
    Enhanced StockXpert model with:
    - Regime detection
    - Probabilistic magnitude output
    - Quantile regression for risk assessment
    - Adaptive confidence calibration
    """
    
    def __init__(
        self,
        num_stocks: int,
        short_dim: int,
        mid_dim: int,
        long_dim: int,
        context_dim: int,
        num_horizons: int = 5,
        hidden_dim: int = 96,
        stock_embed_dim: int = 16,
        attn_heads: int = 4,
        dropout: float = 0.15,
        bottleneck_dim: int = 64,
        use_probabilistic: bool = True,
        use_quantile: bool = True,
        use_regime: bool = True,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        
        self.num_horizons = num_horizons
        self.use_probabilistic = use_probabilistic
        self.use_quantile = use_quantile
        self.use_regime = use_regime
        self.quantiles = quantiles
        
        # Encoders
        self.short_encoder = ResNLSEncoder(short_dim, hidden_dim, dropout)
        self.mid_encoder = BiGRUEncoder(mid_dim, hidden_dim, dropout=dropout)
        self.long_encoder = BiLSTMEncoder(long_dim, hidden_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(context_dim, hidden_dim, dropout)
        self.sentiment_encoder = SentimentEncoder(9, hidden_dim, dropout)
        
        # Fusion
        self.fusion = AttentionFusion(hidden_dim, attn_heads, dropout, num_horizons)
        self.mixer = HorizonMixer(hidden_dim, num_horizons, n_layers=2, dropout=dropout)
        
        # Stock embedding
        self.stock_embedding = nn.Embedding(num_stocks, stock_embed_dim)
        
        combined_dim = hidden_dim + stock_embed_dim
        
        # Regime detection
        if use_regime:
            self.regime_detector = RegimeDetector(hidden_dim, dropout)
        
        # Direction heads (same as before)
        def make_bottleneck_head(out_features=1):
            return nn.Sequential(
                nn.Linear(combined_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                ResidualBlock(bottleneck_dim, dropout),
                nn.Linear(bottleneck_dim, 32),
                nn.ReLU(),
                nn.Linear(32, out_features)
            )
            
        self.direction_heads = nn.ModuleList([make_bottleneck_head() for _ in range(num_horizons)])
        
        # Magnitude heads - probabilistic or standard
        if use_probabilistic:
            self.prob_magnitude_head = ProbabilisticMagnitudeHead(
                combined_dim, num_horizons, dropout
            )
        else:
            self.magnitude_heads = nn.ModuleList([
                make_bottleneck_head() for _ in range(num_horizons)
            ])
        
        # Quantile heads for risk assessment
        if use_quantile:
            self.quantile_head = QuantileMagnitudeHead(
                combined_dim, num_horizons, quantiles, dropout
            )
        
        # Level heads
        self.level_heads = nn.ModuleList([make_bottleneck_head(3) for _ in range(num_horizons)])
        
        # Shared heads
        self.confidence_head = nn.Linear(combined_dim, num_horizons)
        self.target_zone_head = nn.Linear(combined_dim, num_horizons * 7)
        
        # Regime-conditioned direction adjustment
        if use_regime:
            self.regime_direction_gate = nn.Linear(3, num_horizons)
        
    def forward(
        self,
        X_short: torch.Tensor,
        X_mid: torch.Tensor,
        X_long: torch.Tensor,
        X_context: torch.Tensor,
        X_sentiment: torch.Tensor,
        stock_idx: torch.Tensor
    ) -> Dict:
        """
        Forward pass returning comprehensive predictions.
        
        Returns:
            Dict with:
                - direction_logits: (B, num_horizons)
                - magnitude_pred: (B, num_horizons)
                - magnitude_std: (B, num_horizons) or None
                - magnitude_quantiles: List of (B, num_horizons) or None
                - confidence: (B, num_horizons)
                - target_zone_logits: (B, num_horizons, 7)
                - aux_trend_logit: (B, 1)
                - level_pred: (B, num_horizons, 3)
                - regime_logits: (B, 3) or None
                - regime_probs: (B, 3) or None
                - attention_dict: Dict
        """
        # Encode
        short_enc = self.short_encoder(X_short)
        mid_enc = self.mid_encoder(X_mid)
        long_enc = self.long_encoder(X_long)
        context_enc = self.context_encoder(X_context)
        sent_enc = self.sentiment_encoder(X_sentiment)
        
        # Fuse
        global_fused, fused_per_horizon, attention_dict = self.fusion(
            short_enc, mid_enc, long_enc, context_enc, sent_enc
        )
        
        # Horizon Mixer
        enhanced_horizons, aux_trend_logit = self.mixer(fused_per_horizon)
        fused_per_horizon = enhanced_horizons
        
        # Stock embedding
        stock_emb = self.stock_embedding(stock_idx)
        stock_emb = F.dropout(stock_emb, p=0.1, training=self.training)
        
        global_combined = torch.cat([global_fused, stock_emb], dim=1)
        
        # Regime detection
        regime_logits = None
        regime_probs = None
        if self.use_regime:
            regime_logits, regime_probs = self.regime_detector(global_fused)
        
        # Direction prediction
        dir_logits_list = []
        for i in range(len(self.direction_heads)):
            horizon_rep = fused_per_horizon[:, i, :]
            if i == 0:
                horizon_rep = horizon_rep + short_enc
            elif i >= 3:
                horizon_rep = horizon_rep + long_enc
            
            ctx = torch.cat([horizon_rep, stock_emb], dim=1)
            dir_logits_list.append(self.direction_heads[i](ctx))
        
        direction_logits = torch.cat(dir_logits_list, dim=1)
        
        # Apply regime conditioning to direction
        if self.use_regime and regime_probs is not None:
            regime_adj = self.regime_direction_gate(regime_probs)
            # Gated adjustment: small influence from regime
            direction_logits = direction_logits + 0.1 * regime_adj
        
        # Magnitude prediction
        magnitude_std = None
        magnitude_quantiles = None
        
        if self.use_probabilistic:
            magnitude_pred, magnitude_std = self.prob_magnitude_head(global_combined)
        else:
            mag_pred_list = []
            for i in range(len(self.magnitude_heads)):
                horizon_rep = fused_per_horizon[:, i, :]
                if i == 0:
                    horizon_rep = horizon_rep + short_enc
                elif i >= 3:
                    horizon_rep = horizon_rep + long_enc
                ctx = torch.cat([horizon_rep, stock_emb], dim=1)
                mag_pred_list.append(self.magnitude_heads[i](ctx))
            magnitude_pred = torch.cat(mag_pred_list, dim=1)
        
        # Quantile prediction
        if self.use_quantile:
            magnitude_quantiles = self.quantile_head(global_combined)
        
        # Level prediction
        level_pred_list = []
        for i in range(len(self.level_heads)):
            horizon_rep = fused_per_horizon[:, i, :]
            ctx = torch.cat([horizon_rep, stock_emb], dim=1)
            level_pred_list.append(self.level_heads[i](ctx))
        level_pred = torch.stack(level_pred_list, dim=1)
        
        # Shared predictions
        confidence = torch.sigmoid(self.confidence_head(global_combined))
        tz_flat = self.target_zone_head(global_combined)
        target_zone_logits = tz_flat.view(-1, self.num_horizons, 7)
        
        return {
            'direction_logits': direction_logits,
            'magnitude_pred': magnitude_pred,
            'magnitude_std': magnitude_std,
            'magnitude_quantiles': magnitude_quantiles,
            'confidence': confidence,
            'target_zone_logits': target_zone_logits,
            'aux_trend_logit': aux_trend_logit,
            'level_pred': level_pred,
            'regime_logits': regime_logits,
            'regime_probs': regime_probs,
            'attention_dict': attention_dict
        }
    
    def predict_with_uncertainty(
        self,
        X_short: torch.Tensor,
        X_mid: torch.Tensor,
        X_long: torch.Tensor,
        X_context: torch.Tensor,
        X_sentiment: torch.Tensor,
        stock_idx: torch.Tensor,
        n_samples: int = 100
    ) -> Dict:
        """
        Generate predictions with Monte Carlo uncertainty estimation.
        
        Uses MC Dropout for epistemic uncertainty.
        """
        self.train()  # Enable dropout
        
        direction_samples = []
        magnitude_samples = []
        
        for _ in range(n_samples):
            output = self.forward(
                X_short, X_mid, X_long, X_context, X_sentiment, stock_idx
            )
            direction_samples.append(torch.sigmoid(output['direction_logits']))
            magnitude_samples.append(output['magnitude_pred'])
        
        direction_samples = torch.stack(direction_samples, dim=0)
        magnitude_samples = torch.stack(magnitude_samples, dim=0)
        
        self.eval()
        
        return {
            'direction_mean': direction_samples.mean(dim=0),
            'direction_std': direction_samples.std(dim=0),
            'magnitude_mean': magnitude_samples.mean(dim=0),
            'magnitude_std': magnitude_samples.std(dim=0),
            'direction_lower': direction_samples.quantile(0.1, dim=0),
            'direction_upper': direction_samples.quantile(0.9, dim=0),
            'magnitude_lower': magnitude_samples.quantile(0.1, dim=0),
            'magnitude_upper': magnitude_samples.quantile(0.9, dim=0)
        }
