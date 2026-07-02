"""
Fundamental scoring module for StockXpert.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("stockxpert.fundamental.scorer")

@dataclass
class StockFundamentalScore:
    symbol: str
    total_score: float  # 0-100
    value_score: float
    growth_score: float
    momentum_score: float
    quality_score: float
    factors: Dict[str, float]

class FundamentalScorer:
    """
    Scores stocks based on qualitative and quantitative fundamental factors.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Weights for each category (Value, Growth, Momentum, Quality)
        """
        self.weights = weights or {
            'value': 0.30,
            'growth': 0.25,
            'momentum': 0.20,
            'quality': 0.25
        }
        
    def _fetch_fundamentals(self, symbol: str) -> Dict[str, float]:
        """Fetch fundamental data from yfinance."""
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract safely with defaults
            data = {
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'dividend_yield': info.get('dividendYield', 0) or 0,
                'held_percent_institutions': info.get('heldPercentInstitutions', 0) or 0,
                'price_52w_high': info.get('fiftyTwoWeekHigh', 0),
                'current_price': info.get('currentPrice', 0)
            }
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {str(e)}")
            return {}

    def compute_scores(self, feature_data: Dict[str, pd.DataFrame]) -> Dict[str, StockFundamentalScore]:
        """
        Compute scores for all stocks based on the last available data point.
        Uses yfinance for live data if available, otherwise falls back to technical proxies.
        """
        results = {}
        
        for symbol, df in feature_data.items():
            if df.empty: continue
            
            # Fetch live fundamentals
            # Note: In a real production pipeline, this should be cached or batched
            fund_data = self._fetch_fundamentals(symbol)
            
            # Use dataframe inputs as backup or complement
            latest = df.iloc[-1]
            
            scores = {}
            factors = {}
            
            # --- 1. VALUE SCORE (30%) ---
            # PE: Lower is better (0-30 range ideal)
            pe = fund_data.get('pe_ratio', 0)
            if pe > 0:
                # Score 100 for PE=10, Score 0 for PE>100
                val_pe = max(0, min(100, 100 - (pe - 10))) 
            else:
                val_pe = 50 # Neutral if unknown or negative earnings
            
            # PB: Lower is better (0-5 ideal)
            pb = fund_data.get('pb_ratio', 0)
            if pb > 0:
                val_pb = max(0, min(100, 100 - (pb - 1) * 10))
            else:
                val_pb = 50
                
            scores['value'] = (val_pe * 0.6 + val_pb * 0.4)
            
            # --- 2. GROWTH SCORE (25%) ---
            # Rev Growth: Higher is better
            rev_g = fund_data.get('revenue_growth', 0)
            # 20%+ growth = 100 score, 0% = 50 score
            val_rev = max(0, min(100, 50 + rev_g * 250))
            
            # PEG proxy logic could go here
            scores['growth'] = val_rev
            
            # --- 3. MOMENTUM SCORE (20%) ---
            # 52w High Proximity from fundamentals
            curr = fund_data.get('current_price', 0)
            high_52 = fund_data.get('price_52w_high', 0)
            if high_52 > 0:
                prox_52 = (curr / high_52) * 100 # Closer to 100 is better momentum usually
            else:
                prox_52 = 50
                
            # Technical RS (Relative Strength) from features
            rs_score = 50
            if 'rs_proxy' in df.columns:
                # Z-score normalization roughly
                rs_score = max(0, min(100, 50 + latest['rs_proxy'] * 1000)) # Scale adjusted
            
            scores['momentum'] = (prox_52 * 0.4 + rs_score * 0.6)
            
            # --- 4. QUALITY SCORE (25%) ---
            # ROE: >15% is great
            roe = fund_data.get('roe', 0)
            val_roe = max(0, min(100, roe * 500)) # 0.20 ROE -> 100 score
            
            # Margins: >20% is great
            margin = fund_data.get('profit_margins', 0)
            val_margin = max(0, min(100, margin * 500))
            
            # Debt/Eq: <0.5 is great, >2 is bad
            de = fund_data.get('debt_to_equity', 0)
            if de >= 0:
                val_de = max(0, min(100, 100 - (de * 50)))
            else:
                val_de = 50
                
            # Institutional Holding: Validation from smart money
            inst = fund_data.get('held_percent_institutions', 0)
            val_inst = max(0, min(100, inst * 100)) # 1.0 (100%) -> 100
            
            scores['quality'] = (val_roe * 0.3 + val_margin * 0.3 + val_de * 0.2 + val_inst * 0.2)
            
            # --- TOTAL ---
            total = (
                self.weights['value'] * scores['value'] +
                self.weights['growth'] * scores['growth'] +
                self.weights['momentum'] * scores['momentum'] +
                self.weights['quality'] * scores['quality']
            )
            
            results[symbol] = StockFundamentalScore(
                symbol=symbol,
                total_score=float(total),
                value_score=float(scores['value']),
                growth_score=float(scores['growth']),
                momentum_score=float(scores['momentum']),
                quality_score=float(scores['quality']),
                factors=fund_data
            )
            
        return results
