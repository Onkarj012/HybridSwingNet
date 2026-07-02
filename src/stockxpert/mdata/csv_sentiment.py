"""
CSV Sentiment Client - loads and aggregates sentiment from CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger("stockxpert.mdata.csv_sentiment")


class CSVSentimentClient:
    """
    Client to load sentiment data from CSV file.
    
    Expected CSV format:
    Company Name, Symbol, Headline, Publish Date, Sentiment
    """
    
    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Path to CSV file with sentiment data
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        logger.info(f"Loading sentiment data from {csv_path}")
        self.raw_df = pd.read_csv(csv_path)
        
        # Clean column names
        self.raw_df.columns = self.raw_df.columns.str.strip()
        
        # Handle different column names (Case-insensitive check)
        # Standardize to 'sentiment_score', 'symbol', 'date'
        
        # 1. Sentiment Score
        if 'sentiment_score' in self.raw_df.columns:
             pass 
        elif 'Sentiment' in self.raw_df.columns:
            # Map from categorical
            sentiment_map = {
                'Positive': 1.0, 
                'Neutral': 0.0,
                'Negative': -1.0
            }
            self.raw_df['sentiment_score'] = self.raw_df['Sentiment'].map(sentiment_map)
        else:
            # Try to find a column that looks like sentiment
            found = False
            for col in self.raw_df.columns:
                if col.lower() in ['sentiment', 'score', 'sentiment_score']:
                    self.raw_df['sentiment_score'] = pd.to_numeric(self.raw_df[col], errors='coerce')
                    found = True
                    break
            
            if not found:
                logger.warning(f"No sentiment column found in {csv_path}. Columns: {self.raw_df.columns}")
                self.raw_df['sentiment_score'] = 0.0
                
        # 2. Date
        if 'date' in self.raw_df.columns:
            self.raw_df['date'] = pd.to_datetime(self.raw_df['date'], errors='coerce', utc=True)
        elif 'Publish Date' in self.raw_df.columns:
             self.raw_df['date'] = pd.to_datetime(self.raw_df['Publish Date'], errors='coerce', utc=True)
        else:
             # Try case-insensitive lookup
            found = False
            for col in self.raw_df.columns:
                if col.lower() in ['date', 'publish date', 'published', 'timestamp', 'datetime']:
                    self.raw_df['date'] = pd.to_datetime(self.raw_df[col], errors='coerce', utc=True)
                    found = True
                    break
            if not found:
                 logger.warning(f"No date column found in {csv_path}")

        # 3. Symbol
        if 'symbol' in self.raw_df.columns:
            # Rename to Symbol for internal consistency if needed, or keep as symbol
            # But the get_daily_sentiment method expects 'Symbol' (uppercase) in logic below? 
            # Actually let's standardize to 'Symbol' (Title case) to match rest of class logic
            self.raw_df.rename(columns={'symbol': 'Symbol'}, inplace=True)
        elif 'Symbol' in self.raw_df.columns:
            pass
        else:
             # Try case-insensitive lookup
            for col in self.raw_df.columns:
                if col.lower() == 'symbol':
                    self.raw_df.rename(columns={col: 'Symbol'}, inplace=True)
                    break

            
        # Drop rows with invalid dates or sentiment
        self.raw_df = self.raw_df.dropna(subset=['date', 'sentiment_score'])
        
        logger.info(f"Loaded {len(self.raw_df)} valid sentiment records")
    
    def get_daily_sentiment(
        self, 
        symbols: list,
        trading_calendar: pd.DatetimeIndex,
        tz: str = "Asia/Kolkata"
    ) -> pd.DataFrame:
        """
        Aggregate sentiment to daily level per symbol.
        
        Args:
            symbols: List of stock symbols
            trading_calendar: Trading date index
            tz: Timezone for date alignment
            
        Returns:
            DataFrame with columns: date, symbol, sentiment_mean, sentiment_count, sentiment_std
        """
        logger.info(f"Building daily sentiment for {len(symbols)} symbols")
        
        # Filter to relevant symbols - handle .NS suffix
        # Strip suffixes from search list and compare
        clean_symbols = [s.split('.')[0] for s in symbols]
        df = self.raw_df[self.raw_df['Symbol'].isin(clean_symbols)].copy()
        
        # Map back to original symbols with suffix for consistency with the rest of the app
        sym_map = {s.split('.')[0]: s for s in symbols}
        df['Symbol'] = df['Symbol'].map(sym_map)
        
        if len(df) == 0:
            logger.warning("No sentiment data found for specified symbols")
            return pd.DataFrame(columns=['date', 'symbol', 'sentiment_mean', 'sentiment_count', 'sentiment_std'])
        
        # Align dates to trading calendar
        df['date'] = df['date'].dt.tz_localize(None)
        
        # Aggregate by (date, symbol)
        agg_df = df.groupby([df['date'].dt.date, 'Symbol'])['sentiment_score'].agg([
            ('sentiment_mean', 'mean'),
            ('sentiment_count', 'count'),
            ('sentiment_std', 'std')
        ]).reset_index()
        
        agg_df.columns = ['date', 'symbol', 'sentiment_mean', 'sentiment_count', 'sentiment_std']
        agg_df['date'] = pd.to_datetime(agg_df['date'])
        
        # Fill NaN std with 0 (happens when count=1)
        agg_df['sentiment_std'] = agg_df['sentiment_std'].fillna(0.0)
        
        logger.info(f"Aggregated to {len(agg_df)} daily sentiment records")
        
        return agg_df
    
    def fetch_for_symbols(
        self,
        symbols: list,
        start_date: datetime,
        end_date: datetime,
        trading_calendar: pd.DatetimeIndex,
        tz: str = "Asia/Kolkata"
    ) -> pd.DataFrame:
        """
        Main interface compatible with other sentiment clients.
        
        Returns:
            DataFrame with sentiment data ready to merge with price data
        """
        daily_sent = self.get_daily_sentiment(symbols, trading_calendar, tz)
        
        # Filter to date range
        daily_sent = daily_sent[
            (daily_sent['date'] >= start_date) & 
            (daily_sent['date'] <= end_date)
        ]
        
        return daily_sent
