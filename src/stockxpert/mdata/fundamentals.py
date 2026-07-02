"""
Fundamental data extraction using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("stockxpert.mdata.fundamentals")

class FundamentalDownloader:
    """
    Downloads and processes fundamental data point-in-time for backtesting.
    Note: yfinance quarterly data provides the most recent 4-5 quarters.
    For true historical backtesting, a premium source would be needed,
    but we will approximate using what yfinance provides.
    """
    
    def __init__(self, cache_dir: str = "data/cache/fundamentals"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_fundamentals(self, symbol: str) -> pd.DataFrame:
        """
        Get fundamental features for a symbol.
        Returns a DataFrame indexed by Date with fundamental columns.
        """
        cache_file = self.cache_dir / f"{symbol}.csv"
        
        # In a real system, we'd check if cache is too old
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
        logger.info(f"Downloading fundamentals for {symbol}")
        ticker = yf.Ticker(symbol)
        
        # 1. Income Statement (Quarterly)
        qf = ticker.quarterly_financials
        if qf.empty:
            logger.warning(f"No quarterly financials for {symbol}")
            return pd.DataFrame()
            
        # Transpose so dates are rows
        df_q = qf.T.sort_index()
        
        # Calculate some growth metrics if possible
        # We need specific keys which might vary slightly
        keys = {
            'revenue': ['Total Revenue', 'Revenue'],
            'net_income': ['Net Income'],
            'eps': ['Basic EPS', 'Diluted EPS']
        }
        
        processed = pd.DataFrame(index=df_q.index)
        
        for feat, possible_keys in keys.items():
            for pk in possible_keys:
                if pk in df_q.columns:
                    processed[feat] = df_q[pk]
                    break
        
        # Current info basics (Snapshot)
        info = ticker.info
        curr_pe = info.get('trailingPE', np.nan)
        curr_eps = info.get('trailingEps', np.nan)
        
        # Map snapshots back to dates (Approximation)
        processed['pe_ratio'] = curr_pe
        processed['eps_trailing'] = processed.get('eps', curr_eps)
        
        # Save to cache
        processed.index.name = 'Date'
        processed.to_csv(cache_file)
        return processed

    def map_to_prices(self, price_df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill fundamental data onto daily price dates.
        """
        if fundamental_df.empty:
            # Return empty but with correct index
            return pd.DataFrame(index=price_df.index)
            
        # Reindex to match prices and forward fill
        # We shift fundamental dates by 30 days to account for reporting lag
        fund_shifted = fundamental_df.copy()
        fund_shifted.index = fund_shifted.index + pd.Timedelta(days=45) 
        
        # Align timezones before reindexing
        if price_df.index.tz is not None and fund_shifted.index.tz is None:
             # Localize to match price_df (e.g. UTC or Asia/Kolkata)
             # Note: simple tz_localize might be ambiguous for DST, but acceptable here
             fund_shifted.index = fund_shifted.index.tz_localize(price_df.index.tz)
        elif price_df.index.tz is None and fund_shifted.index.tz is not None:
             fund_shifted.index = fund_shifted.index.tz_localize(None)

        mapped = fund_shifted.reindex(price_df.index).ffill()
        
        # Fill NaNs at the beginning if any
        mapped = mapped.bfill()
        
        return mapped
