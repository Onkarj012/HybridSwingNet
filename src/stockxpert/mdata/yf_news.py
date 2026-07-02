"""
yfinance news provider for fetching stock headlines.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger("stockxpert.mdata.yf_news")

class YFinanceNewsClient:
    """
    Client for fetching news via yfinance.
    Note: yfinance usually return only a few recent articles (e.g., last 10-20).
    """
    
    def __init__(self):
        pass

    def fetch_news(
        self,
        symbol: str
    ) -> pd.DataFrame:
        """
        Fetch recent news for a single symbol.
        
        Args:
            symbol: Stock ticker (e.g., HDFCBANK.NS)
            
        Returns:
            DataFrame with [datetime, title, url, source, symbol]
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.get_news()
            
            if not news:
                logger.warning(f"No news found for {symbol} via yfinance")
                return pd.DataFrame(columns=['datetime', 'title', 'url', 'source', 'symbol'])
            
            articles = []
            for item in news:
                # Check for nested structure (new yfinance/yahoo api format)
                content = item.get('content', {})
                
                # Try to extract title
                title = item.get('title')
                if not title:
                    title = content.get('title', '')
                
                # Try to extract date
                # Old format: 'providerPublishTime' (timestamp)
                # New format: 'pubDate' (ISO string) in content
                pub_time = item.get('providerPublishTime')
                if pub_time:
                    dt = datetime.fromtimestamp(pub_time)
                else:
                    # Try parsing pubDate from content
                    pub_date_str = content.get('pubDate')
                    if pub_date_str:
                        try:
                            # Format: '2026-02-17T03:57:49Z'
                            # Handle Z for UTC
                            dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            dt = dt.replace(tzinfo=None) # Make naive for simplicity
                        except ValueError:
                            dt = datetime.now()
                    else:
                        dt = datetime.now()

                # URL/Link
                url = item.get('link')
                if not url:
                    url = content.get('clickThroughUrl', {}).get('url')
                    if not url:
                        url = content.get('canonicalUrl', {}).get('url')
                
                # Publisher
                publisher = item.get('publisher')
                if not publisher:
                    publisher = content.get('provider', {}).get('displayName')

                articles.append({
                    'datetime': dt,
                    'title': title,
                    'url': url,
                    'source': publisher,
                    'symbol': symbol
                })
            
            df = pd.DataFrame(articles)
            logger.info(f"Fetched {len(df)} articles for {symbol} via yfinance")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch yfinance news for {symbol}: {e}")
            return pd.DataFrame(columns=['datetime', 'title', 'url', 'source', 'symbol'])

    def fetch_for_symbols(
        self,
        symbols: List[str],
        cache_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Fetch news for multiple symbols.
        """
        all_data = []
        
        for symbol in symbols:
            # Check cache logic could be added here if needed, 
            # but yfinance news is ephemeral and usually only recent.
            df_symbol = self.fetch_news(symbol)
            if not df_symbol.empty:
                all_data.append(df_symbol)
        
        if not all_data:
            return pd.DataFrame(columns=['datetime', 'title', 'url', 'source', 'symbol'])
            
        return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    client = YFinanceNewsClient()
    df = client.fetch_news("HDFCBANK.NS")
    print(df.head())
