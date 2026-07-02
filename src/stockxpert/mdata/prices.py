"""
Price data download using yfinance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger("stockxpert.mdata.prices")


def download_prices(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: Path,
    auto_adjust: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data for multiple symbols using yfinance.
    
    Args:
        symbols: List of ticker symbols (e.g., ['HDFCBANK.NS'])
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Directory to cache CSVs
        auto_adjust: Use adjusted prices
    
    Returns:
        Dict mapping symbol -> DataFrame with columns [Date, Open, High, Low, Close, Volume]
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for symbol in symbols:
        cache_file = cache_dir / f"{symbol}.csv"
        
        # Check cache
        if cache_file.exists():
            logger.info(f"Loading {symbol} from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
            df.set_index('Date', inplace=True)
            results[symbol] = df
            continue
        
        # Download
        logger.info(f"Downloading {symbol} from {start} to {end}")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=auto_adjust)
            
            if df.empty:
                logger.warning(f"No data downloaded for {symbol}")
                continue
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Reset index to have Date as column for CSV
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Ensure Date column exists (yfinance returns different index names)
            if 'Date' not in df.columns and df.index.name == 'Date':
                df.reset_index(inplace=True)
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved {symbol} to cache ({len(df)} rows)")
            
            # Set index back to Date for return
            df.set_index('Date', inplace=True)
            results[symbol] = df
            
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            continue
    
    logger.info(f"Downloaded {len(results)}/{len(symbols)} symbols successfully")
    return results
