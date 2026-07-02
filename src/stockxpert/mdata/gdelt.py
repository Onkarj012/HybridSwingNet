"""
GDELT 2.1 API client for news fetching.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import time
import logging
from pathlib import Path

logger = logging.getLogger("stockxpert.mdata.gdelt")


class GDELTClient:
    """
    Client for GDELT 2.1 DOC API.
    Fetches news articles matching keywords.
    """
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def __init__(
        self,
        max_records: int = 250,
        sleep_seconds: float = 1.0
    ):
        """
        Args:
            max_records: Max records per query
            sleep_seconds: Sleep between queries for rate limiting
        """
        self.max_records = max_records
        self.sleep_seconds = sleep_seconds
    
    def fetch_news(
        self,
        keywords: List[str],
        start_date: datetime,
        end_date: datetime,
        language: str = "English"
    ) -> pd.DataFrame:
        """
        Fetch news articles for given keywords and date range.
        
        Args:
            keywords: List of search terms (OR logic)
            start_date: Start datetime
            end_date: End datetime
            language: Language filter
        
        Returns:
            DataFrame with columns: [datetime, title, url, source, query_used]
        """
        all_articles = []
        
        # Split into monthly chunks to avoid overwhelming API
        current = start_date
        
        while current < end_date:
            chunk_end = min(current + timedelta(days=30), end_date)
            
            # Build query
            query = " OR ".join(f'"{kw}"' for kw in keywords)
            
            params = {
                'query': query,
                'mode': 'artlist',
                'maxrecords': self.max_records,
                'format': 'json',
                'timespan': self._format_timespan(current, chunk_end),
                'sourcelang': language
            }
            
            logger.debug(f"Querying GDELT: {current.date()} to {chunk_end.date()}")
            
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get('articles', [])
                
                for article in articles:
                    all_articles.append({
                        'datetime': article.get('seendate', ''),
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'source': article.get('domain', ''),
                        'query_used': query
                    })
                
                logger.info(f"Fetched {len(articles)} articles for period {current.date()} - {chunk_end.date()}")
                
            except Exception as e:
                logger.warning(f"GDELT query failed for {current.date()}: {e}")
            
            # Rate limiting
            time.sleep(self.sleep_seconds)
            current = chunk_end
        
        # Create DataFrame
        if not all_articles:
            logger.warning("No articles fetched")
            return pd.DataFrame(columns=['datetime', 'title', 'url', 'source', 'query_used'])
        
        df = pd.DataFrame(all_articles)
        
        # Parse datetime (GDELT format: YYYYMMDDHHmmss)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S', errors='coerce')
        
        # Deduplicate by URL
        df = df.drop_duplicates(subset=['url'], keep='first')
        
        logger.info(f"Total unique articles: {len(df)}")
        return df
    
    def _format_timespan(self, start: datetime, end: datetime) -> str:
        """Format timespan for GDELT API (e.g., '20200101-20200131')."""
        return f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    
    def fetch_for_symbols(
        self,
        query_map: dict,
        start_date: datetime,
        end_date: datetime,
        language: str = "English",
        cache_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Fetch news for multiple symbols using query map.
        
        Args:
            query_map: Dict of symbol -> list of keywords
            start_date: Start datetime
            end_date: End datetime
            language: Language filter
            cache_dir: Optional cache directory
        
        Returns:
            DataFrame with additional 'symbol' column
        """
        all_data = []
        
        for symbol, keywords in query_map.items():
            logger.info(f"Fetching news for {symbol} (keywords: {keywords})")
            
            # Check cache
            if cache_dir:
                cache_file = cache_dir / f"gdelt_{symbol}.parquet"
                if cache_file.exists():
                    logger.info(f"Loading {symbol} news from cache")
                    df_symbol = pd.read_parquet(cache_file)
                    all_data.append(df_symbol)
                    continue
            
            # Fetch
            df_symbol = self.fetch_news(keywords, start_date, end_date, language)
            
            if not df_symbol.empty:
                df_symbol['symbol'] = symbol
                
                # Save to cache
                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_dir / f"gdelt_{symbol}.parquet"
                    df_symbol.to_parquet(cache_file)
                    logger.info(f"Cached {symbol} news ({len(df_symbol)} articles)")
                
                all_data.append(df_symbol)
        
        if not all_data:
            return pd.DataFrame(columns=['datetime', 'title', 'url', 'source', 'query_used', 'symbol'])
        
        return pd.concat(all_data, ignore_index=True)
