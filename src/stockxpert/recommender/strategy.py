"""
Trading Strategy Module.
Handles position sizing, risk management, and execution logic.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TradeSetup:
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    risk_reward_ratio: float

class TradingStrategy:
    """
    Implements a risk-managed trading strategy.
    
    Philosophy:
    - Survive first: Strict stop losses based on volatility (ATR).
    - Earn second: Position sizing based on conviction and Kelly Criterion.
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,  # risk 2% of capital per trade
        max_position_size: float = 0.20, # max 20% in one stock
        atr_multiplier_stop: float = 2.0,
        atr_multiplier_profit: float = 4.0,
        win_rate_estimate: float = 0.52
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.atr_multiplier_stop = atr_multiplier_stop
        self.atr_multiplier_profit = atr_multiplier_profit
        self.win_rate = win_rate_estimate
        
    def calculate_position_size(self, entry: float, stop: float, conviction: float) -> float:
        """
        Calculate optimal position size using a modified Kelly Criterion.
        
        Kelly Fraction = W - (1-W)/R
        where W = Win Rate, R = Reward/Risk Ratio
        
        We use 'Half Kelly' or 'Quarter Kelly' for safety, scaled by conviction.
        """
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0: return 0.0
        
        # Reward/Risk Ratio
        # We assume we target 2x our risk (R=2) minimum
        R = 2.0 
        
        # Kelly Formula
        W = self.win_rate
        # K% = W - (1-W)/R
        kelly_pct = W - (1 - W) / R
        
        # Safety scaling (Half Kelly is standard conservative approach)
        safe_kelly = kelly_pct * 0.5
        
        # Conviction scaling (0.0 to 1.0)
        # If conviction is low, reduce size further
        final_pct = safe_kelly * conviction
        
        # Cap at max position size
        final_pct = min(final_pct, self.max_position_size)
        
        # Cap at max risk per trade
        # Risk % = Position % * (Risk/Entry)
        # Position % = Risk % / (Risk/Entry)
        risk_pct_of_entry = risk_per_share / entry
        risk_based_cap = self.risk_per_trade / risk_pct_of_entry
        
        final_pct = min(final_pct, risk_based_cap)
        
        # Ensure non-negative
        return max(0.0, final_pct)
    
    def analyze_setup(self, symbol: str, current_price: float, atr: float, confidence: float, fund_score: float, p_up: float) -> TradeSetup:
        """
        Analyze a potential trade setup.
        Args:
            p_up: Probability of upward movement (0 to 1).
        """
        # Determine dynamic stop loss
        stop_dist = atr * self.atr_multiplier_stop
        stop_loss = current_price - stop_dist
        
        # Determine target
        profit_dist = atr * self.atr_multiplier_profit
        take_profit = current_price + profit_dist
        
        # Reward Risk
        rr = profit_dist / stop_dist
        
        # Conviction metric (0 to 1)
        # Combine technical confidence (0-1) and fundamental score (0-100)
        # We normalize fund score to 0-1
        fund_normalized = fund_score / 100.0
        
        # Base conviction on confidence
        base_conviction = (confidence * 0.6) + (fund_normalized * 0.4)
        
        # Directional Filter
        # If model is bearish (p_up < 0.5), we shouldn't buy regardless of confidence
        if p_up < 0.5:
             # Bearish signal
             action = "AVOID" if confidence > 0.6 else "HOLD"
             return TradeSetup(
                symbol=symbol,
                action=action,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=0.0,
                risk_reward_ratio=rr
            )
             
        # Scale conviction by probability strength
        # We want p_up=0.70 to be full conviction (1.0 multiplier)
        # We want p_up=0.55 to be weak conviction (0.25 multiplier)
        # Formula: (p_up - 0.5) * 5
        # 0.50 -> 0.0
        # 0.60 -> 0.5
        # 0.70 -> 1.0
        prob_scaler = min(1.2, max(0.0, (p_up - 0.5) * 5.0))
        
        conviction = base_conviction * prob_scaler
        
        # Determine action category
        action = "AVOID"
        if conviction > 0.70:
            action = "CONVICTION BUY"
        elif conviction > 0.50:
            action = "STRONG BUY"
        elif conviction > 0.30:
            action = "MODERATE BUY"
        elif conviction > 0.20:
             # If fundamentals are bad but tech is good -> Speculative
             if fund_normalized < 0.4:
                 action = "SPECULATIVE BUY"
             else:
                 action = "HOLD"
        else:
             action = "HOLD"
        
        # Position Sizing
        pos_size = 0.0
        if "BUY" in action:
            # Recalculate size using the raw conviction for sizing
            pos_size = self.calculate_position_size(current_price, stop_loss, conviction)
            
        return TradeSetup(
            symbol=symbol,
            action=action,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=pos_size,
            risk_reward_ratio=rr
        )
