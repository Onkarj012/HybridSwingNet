"""
Trading calendar utilities derived from actual price data.
"""

import pandas as pd
from typing import Dict, Set
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("stockxpert.mdata.calendar")


class TradingCalendar:
    """
    Trading calendar derived from actual price data.
    Handles per-symbol trading days and holiday detection.
    """
    
    def __init__(self, price_data: Dict[str, pd.DataFrame]):
        """
        Args:
            price_data: Dict of symbol -> DataFrame with DatetimeIndex
        """
        self.symbol_dates = {}
        
        for symbol, df in price_data.items():
            # Get trading dates as set
            dates = set(pd.to_datetime(df.index).date)
            self.symbol_dates[symbol] = sorted(dates)
        
        # Union of all trading dates
        all_dates = set()
        for dates in self.symbol_dates.values():
            all_dates.update(dates)
        self.all_trading_dates = sorted(all_dates)
        
        logger.info(f"Initialized calendar with {len(self.all_trading_dates)} total trading days")
    
    def is_trading_day(self, date: datetime.date, symbol: str = None) -> bool:
        """Check if a date is a trading day."""
        if symbol:
            return date in self.symbol_dates.get(symbol, set())
        return date in set(self.all_trading_dates)
    
    def next_trading_day(self, date: datetime.date, symbol: str = None) -> datetime.date:
        """
        Get the next trading day after the given date.
        
        Args:
            date: Current date
            symbol: If provided, use symbol-specific calendar
        
        Returns:
            Next trading date
        """
        dates = self.symbol_dates.get(symbol, self.all_trading_dates) if symbol else self.all_trading_dates
        
        # Find next date in sorted list
        for td in dates:
            if td > date:
                return td
        
        # If no future date, return last + 1 day (fallback)
        return dates[-1] + timedelta(days=1)
    
    def get_trading_dates_between(
        self,
        start: datetime.date,
        end: datetime.date,
        symbol: str = None
    ) -> list:
        """Get all trading dates in a range."""
        dates = self.symbol_dates.get(symbol, self.all_trading_dates) if symbol else self.all_trading_dates
        return [d for d in dates if start <= d <= end]
