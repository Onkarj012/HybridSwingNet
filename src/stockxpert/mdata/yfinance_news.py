"""
YFinance News Fetcher Module.
"""
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("stockxpert.mdata.yfinance_news")

class YFinanceNewsClient:
    """Client to fetch news from Yahoo Finance."""
    
    def __init__(self):
        pass

    def fetch_for_symbols(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        cache_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Fetch news for a list of symbols.
        
        Args:
            symbols: List of ticker symbols (e.g. ['HDFCBANK.NS'])
            start_date: Start date (datetime)
            end_date: End date (datetime)
            cache_dir: Directory to cache results (ignored for now as YF is real-time)
            
        Returns:
            DataFrame with columns ['date', 'symbol', 'title']
        """
        all_news = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching news for {symbol} via yfinance...")
                ticker = yf.Ticker(symbol)
                news_items = ticker.news
                
                count = 0
                for item in news_items:
                    # Parse timestamp
                    # YF returns 'providerPublishTime' as unix timestamp
                    if 'providerPublishTime' in item:
                        pub_time = datetime.fromtimestamp(item['providerPublishTime'])
                        
                        # Filter by date range
                        if start_date <= pub_time <= end_date:
                            all_news.append({
                                'date': pub_time,
                                'symbol': symbol,
                                'title': item.get('title', '')
                            })
                            count += 1
                
                logger.debug(f"Found {count} news items for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                
        if not all_news:
            logger.warning("No news found for any symbol in the specified range.")
            return pd.DataFrame(columns=['date', 'symbol', 'title'])
            
        df = pd.DataFrame(all_news)
        # Sort by date
        df = df.sort_values('date')
        
        logger.info(f"Total fetched news items: {len(df)}")
        return df
