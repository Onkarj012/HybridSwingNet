"""
Intraday signal generation engine.
Implements rule-based strategies for intraday trading:
1. VWAP Pullback / Rejection
2. Opening Range Breakout (ORB)
3. Momentum Continuation
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class IntradaySignalGenerator:
    """
    Generates trading signals based on intraday price action.
    """
    
    def __init__(self, 
                 orb_minutes: int = 15,
                 vwap_threshold_pct: float = 0.2, # 0.2% tolerance
                 min_volume_mult: float = 1.5): # Volume > 1.5x average
        self.orb_minutes = orb_minutes
        self.vwap_threshold_pct = vwap_threshold_pct
        self.min_volume_mult = min_volume_mult

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate intraday VWAP."""
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp * v).cumsum() / v.cumsum()
        
    def get_orb_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get Opening Range Breakout levels."""
        # Assuming df is indexed by datetime
        # Get first N minutes of data
        if df.empty:
            return {}
            
        start_time = df.index[0]
        end_orb = start_time + pd.Timedelta(minutes=self.orb_minutes)
        orb_data = df[df.index <= end_orb]
        
        if orb_data.empty:
            return {}
            
        return {
            'orb_high': orb_data['High'].max(),
            'orb_low': orb_data['Low'].min(),
            'orb_volume': orb_data['Volume'].sum()
        }

    def generate_signals(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Generate signals for a single stock dataframe.
        Args:
            symbol: Stock symbol
            df: Intraday dataframe (e.g. 5-minute bars)
        Returns:
            List of signal dictionaries
        """
        signals = []
        if df.empty or len(df) < 5:
            return signals

        # Calculate Indicators
        df['vwap'] = self.calculate_vwap(df)
        df['ema_9'] = df['Close'].ewm(span=9).mean()
        df['ema_20'] = df['Close'].ewm(span=20).mean()
        
        # Scan last 3 bars for signals (to catch recent setups)
        recent_window = 3
        if len(df) < recent_window + 1:
            recent_window = len(df) - 1
            
        for i in range(len(df) - recent_window, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            timestamp = curr.name
            
            # 1. ORB Logic
            orb = self.get_orb_levels(df)
            if orb:
                # Check for breakout
                if curr['Close'] > orb['orb_high'] and prev['Close'] <= orb['orb_high']:
                    signals.append({
                        'symbol': symbol,
                        'strategy': 'ORB_Breakout_Long',
                        'side': 'LONG',
                        'entry': curr['Close'],
                        'stop_loss': orb['orb_low'],
                        'timestamp': timestamp,
                        'confidence': 0.7
                    })
                elif curr['Close'] < orb['orb_low'] and prev['Close'] >= orb['orb_low']:
                    signals.append({
                        'symbol': symbol,
                        'strategy': 'ORB_Breakdown_Short',
                        'side': 'SHORT',
                        'entry': curr['Close'],
                        'stop_loss': orb['orb_high'],
                        'timestamp': timestamp,
                        'confidence': 0.7
                    })
            
            # 2. VWAP Strategy
            if curr['Close'] > curr['vwap'] and prev['Close'] <= prev['vwap']:
                signals.append({
                    'symbol': symbol,
                    'strategy': 'VWAP_Crossover_Long',
                    'side': 'LONG',
                    'entry': curr['Close'],
                    'stop_loss': curr['Low'] * 0.995,
                    'timestamp': timestamp,
                    'confidence': 0.6
                })
            elif curr['Close'] < curr['vwap'] and prev['Close'] >= prev['vwap']:
                signals.append({
                    'symbol': symbol,
                    'strategy': 'VWAP_Crossover_Short',
                    'side': 'SHORT',
                    'entry': curr['Close'],
                    'stop_loss': curr['High'] * 1.005,
                    'timestamp': timestamp,
                    'confidence': 0.6
                })

        return signals
