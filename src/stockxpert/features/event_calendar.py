
import pandas as pd
import numpy as np

def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flags for important market events:
    1. Monthly Expiry (Last Thursday of the month)
    2. Earnings Season (April, July, Oct, Jan)
    3. RBI Policy Months (Feb, April, June, Aug, Oct, Dec - approximate)
    """
    df = df.copy()
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    else:
        dates = pd.to_datetime(df.index, utc=True).tz_localize(None)
        
    # 1. Expiry Week (Week containing last Thursday)
    # Vectorized approach to find last Thursday
    # Last day of month
    is_expiry_week = []
    
    for d in dates:
        # Check if date is in the last 7 days of the month
        if d.days_in_month - d.day < 7:
            is_expiry_week.append(1.0) # Approx expiry week
        else:
            is_expiry_week.append(0.0)
            
    df['is_expiry_week'] = is_expiry_week
    
    # 2. Earnings Season (Quarterly results usually come in month following quarter end)
    # Q1 end Mar -> Results Apr/May
    # Q2 end Jun -> Results Jul/Aug
    # Q3 end Sep -> Results Oct/Nov
    # Q4 end Dec -> Results Jan/Feb
    earnings_months = [1, 2, 4, 5, 7, 8, 10, 11]
    df['is_earnings_season'] = dates.month.isin(earnings_months).astype(float)
    
    # 3. Policy Months (Bi-monthly usually)
    policy_months = [2, 4, 6, 8, 10, 12]
    df['is_policy_month'] = dates.month.isin(policy_months).astype(float)
    
    return df
