"""
Unified recommendation engine for StockXpert.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from ..fundamental.scorer import StockFundamentalScore

logger = logging.getLogger("stockxpert.recommender.engine")

from .strategy import TradingStrategy, TradeSetup

@dataclass
class Recommendation:
    symbol: str
    action: str  # CONVICTION BUY, STRONG BUY, MODERATE BUY, SPECULATIVE, HOLD, AVOID
    horizon: str
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    fundamental_score: float
    position_size: float # Recommended portfolio allocation %
    risk_reward: float
    rationale: List[str]

class RecommendationEngine:
    """
    Synthesizes all model outputs and fundamental scores into actionable trading advice.
    """
    
    def __init__(self, strategy: TradingStrategy = None):
        self.strategy = strategy or TradingStrategy()
        
    def generate_recommendations(
        self,
        prediction_df: pd.DataFrame,
        fund_scores: Dict[str, StockFundamentalScore],
        target_horizon: int = 10
    ) -> List[Recommendation]:
        """
        Generate final recommendations for the latest available date.
        """
        recs = []
        
        # 1. Filter for latest date
        # Ensure dates are datetime
        if not pd.api.types.is_datetime64_any_dtype(prediction_df['date']):
            prediction_df['date'] = pd.to_datetime(prediction_df['date'], format='mixed', utc=True)
            
        if prediction_df['date'].dt.tz is not None:
            prediction_df['date'] = prediction_df['date'].dt.tz_convert(None)
        
        latest_date = prediction_df['date'].max()
        df_latest = prediction_df[prediction_df['date'] == latest_date].copy()
        
        for _, row in df_latest.iterrows():
            symbol = row['symbol']
            
            # Extract technical signals
            # We focus on the target horizon (e.g. 10 days)
            prefix = f"p_up_h{target_horizon}"
            if prefix not in row: continue
            
            p_up = row[prefix]
            conf = row.get(f"confidence_h{target_horizon}", 0.5)
            
            # Extract ATR for risk management
            # If not in row, estimate (2% of price)
            atr = row.get('atr_14', row['close_t'] * 0.02)
            
            # Get fundamental score
            fund_obj = fund_scores.get(symbol)
            fund_score = fund_obj.total_score if fund_obj else 50.0
            
            # Analyze setup using Strategy
            setup = self.strategy.analyze_setup(
                symbol=symbol, 
                current_price=row['close_t'], 
                atr=atr, 
                confidence=conf, 
                fund_score=fund_score,
                p_up=p_up
            )
            
            # Construct Rationale
            rationale = []
            if "BUY" in setup.action:
                rationale.append(f"{setup.action}: High conviction ({setup.position_size_pct*100:.1f}% alloc)")
                rationale.append(f"Model Probability: {p_up:.1%} (Target H{target_horizon})")
                
                if fund_score > 70:
                    rationale.append(f"Strong Fundamentals (Score: {fund_score:.0f})")
                elif fund_score < 40:
                    rationale.append(f"Weak Fundamentals (Score: {fund_score:.0f}) - Speculative")
                    
                rationale.append(f"Risk/Reward: {setup.risk_reward_ratio:.1f}x")
            else:
                 rationale.append(f"{setup.action}: Low conviction or neutral signal")
            
            recs.append(Recommendation(
                symbol=symbol,
                action=setup.action,
                horizon=f"{target_horizon} Days",
                entry_price=setup.entry_price,
                target_price=setup.take_profit,
                stop_loss=setup.stop_loss,
                confidence=float(conf),
                fundamental_score=float(fund_score),
                position_size=float(setup.position_size_pct),
                risk_reward=float(setup.risk_reward_ratio),
                rationale=rationale
            ))
            
        # Sort by Position Size (highest allocation first)
        recs.sort(key=lambda x: x.position_size, reverse=True)
        return recs
