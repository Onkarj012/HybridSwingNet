"""
Sentiment aggregation and alignment with trading days.
"""

import pandas as pd
import pytz
from datetime import datetime, time
from typing import Dict, Optional
import logging
from .trading_calendar import TradingCalendar

logger = logging.getLogger("stockxpert.mdata.sentiment")


def build_daily_sentiment(
    news_df: pd.DataFrame,
    trading_calendar: TradingCalendar,
    tz: str = "Asia/Kolkata",
    market_close_time: str = "15:30:00"
) -> pd.DataFrame:
    """
    Build daily sentiment table aligned with trading days.
    
    Key logic:
    - News published BEFORE market close (15:30 IST) -> assigned to SAME trading day
    - News published AFTER market close -> assigned to NEXT trading day
    
    Args:
        news_df: DataFrame with columns [datetime, sentiment_score, symbol]
        trading_calendar: TradingCalendar instance
        tz: Timezone string
        market_close_time: Market close time (HH:MM:SS)
    
    Returns:
        DataFrame with columns [date, symbol, sentiment_mean, sentiment_count, sentiment_std]
    """
    if news_df.empty:
        logger.warning("Empty news DataFrame, returning empty sentiment table")
        return pd.DataFrame(columns=['date', 'symbol', 'sentiment_mean', 'sentiment_count', 'sentiment_std'])
    
    # Ensure datetime column is timezone-aware
    df = news_df.copy()
    
    # Convert to target timezone
    tz_obj = pytz.timezone(tz)
    if df['datetime'].dt.tz is None:
        # Assume UTC if no timezone
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    
    df['datetime'] = df['datetime'].dt.tz_convert(tz_obj)
    
    # Parse market close time
    close_hour, close_min, close_sec = map(int, market_close_time.split(':'))
    market_close = time(close_hour, close_min, close_sec)
    
    # Assign trading date
    def assign_trading_date(row):
        dt = row['datetime']
        symbol = row['symbol']
        
        if pd.isna(dt):
            return None
        
        date = dt.date()
        time_of_day = dt.time()
        
        # If published after market close, assign to next trading day
        if time_of_day > market_close:
            date = trading_calendar.next_trading_day(date, symbol)
        
        # If not a trading day, move to next
        if not trading_calendar.is_trading_day(date, symbol):
            date = trading_calendar.next_trading_day(date, symbol)
        
        return date
    
    df['trading_date'] = df.apply(assign_trading_date, axis=1)
    
    # Drop rows with no valid trading date
    df = df.dropna(subset=['trading_date'])
    
    if df.empty:
        logger.warning("No valid trading dates after alignment")
        return pd.DataFrame(columns=['date', 'symbol', 'sentiment_mean', 'sentiment_count', 'sentiment_std'])
    
    # Aggregate by trading date and symbol
    agg_df = df.groupby(['trading_date', 'symbol']).agg({
        'sentiment_score': ['mean', 'count', 'std']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['date', 'symbol', 'sentiment_mean', 'sentiment_count', 'sentiment_std']
    
    # Fill NaN std with 0
    agg_df['sentiment_std'] = agg_df['sentiment_std'].fillna(0.0)
    
    logger.info(f"Built daily sentiment table: {len(agg_df)} rows across {agg_df['symbol'].nunique()} symbols")
    
    return agg_df


def merge_sentiment_with_prices(
    price_data: Dict[str, pd.DataFrame],
    sentiment_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Merge sentiment data with price DataFrames.
    
    Args:
        price_data: Dict of symbol -> DataFrame (with Date index)
        sentiment_df: Sentiment table with [date, symbol, sentiment_mean, ...]
    
    Returns:
        Dict of symbol -> DataFrame with added sentiment columns
    """
    result = {}
    
    for symbol, df in price_data.items():
        df = df.copy()
        
        # Get sentiment for this symbol
        sent = sentiment_df[sentiment_df['symbol'] == symbol].copy()
        
        if sent.empty:
            # No sentiment available, fill with defaults
            df['sentiment_mean'] = 0.0
            df['sentiment_count'] = 0
            df['sentiment_std'] = 0.0
            
            # Ensure index is tz-naive for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            # Merge on date
            sent['date'] = pd.to_datetime(sent['date']).dt.tz_localize(None)
            sent = sent.set_index('date')
            
            # Reset index to merge
            df = df.reset_index()
            # Ensure price date is also tz-naive before merge
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            df = df.merge(
                sent[['sentiment_mean', 'sentiment_count', 'sentiment_std']],
                left_on='Date',
                right_index=True,
                how='left'
            )
            
            # Fill missing sentiment with 0
            df['sentiment_mean'] = df['sentiment_mean'].fillna(0.0)
            df['sentiment_count'] = df['sentiment_count'].fillna(0).astype(int)
            df['sentiment_std'] = df['sentiment_std'].fillna(0.0)
            
            # Set index back
            df = df.set_index('Date')
        
        result[symbol] = df
    
    logger.info(f"Merged sentiment for {len(result)} symbols")
    return result
