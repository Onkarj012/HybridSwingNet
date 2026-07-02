"""
Gap Up/Gap Down Predictor for StockXpert.

Predicts next-day opening gaps using:
  1. Historical gap patterns (magnitude, frequency, volatility)
  2. Overnight news sentiment (via YFinanceNewsClient + TextBlob)
  3. Technical context (ATR, trend, volatility regime)

Usage:
    predictor = GapPredictor()
    result = predictor.predict(df, symbol, news_df=news_df)
    # result.predicted_gap_pct, result.gap_risk_score, ...
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("stockxpert.features.gap_predictor")

# ─── High-impact keywords for gap magnitude boosting ─────────────────────────
BULLISH_KEYWORDS = [
    "upgrade", "beats", "exceeds", "strong results", "record profit",
    "fda approval", "buyback", "dividend", "outperform", "buy rating",
    "acquisition", "deal", "partnership", "expansion", "bullish",
    "surprise", "above estimate", "raised guidance", "positive",
    "growth", "rally", "strong demand",
]
BEARISH_KEYWORDS = [
    "downgrade", "misses", "disappoints", "weak results", "loss",
    "fda rejection", "fraud", "scam", "lawsuit", "sell rating",
    "underperform", "cut guidance", "warning", "negative", "decline",
    "crash", "below estimate", "downfall", "slump", "bearish",
    "investigation", "probe", "default", "debt concern",
]


@dataclass
class GapPrediction:
    """Result of gap prediction for a single stock."""
    symbol: str
    current_close: float

    # Historical gap features
    avg_gap_pct_5d: float = 0.0
    avg_gap_pct_10d: float = 0.0
    avg_gap_pct_21d: float = 0.0
    gap_volatility_10d: float = 0.0
    gap_up_freq_10d: float = 0.0
    gap_down_freq_10d: float = 0.0
    max_gap_up_21d: float = 0.0
    max_gap_down_21d: float = 0.0
    close_to_open_corr_21d: float = 0.0

    # News-based features
    overnight_sentiment: float = 0.0     # -1 to +1
    overnight_news_count: int = 0
    high_impact_detected: bool = False
    news_gap_pressure: float = 0.0       # weighted sentiment × count × recency

    # Combined prediction
    predicted_gap_pct: float = 0.0       # e.g. +0.012 = +1.2% gap up
    predicted_open: float = 0.0
    gap_risk_score: int = 0              # 0–100
    gap_confidence: str = "LOW"          # LOW / MED / HIGH
    gap_direction: str = "FLAT"          # UP / DOWN / FLAT
    news_catalysts: List[str] = field(default_factory=list)


class GapPredictor:
    """
    Predicts next-day opening gap using historical patterns + news sentiment.

    The prediction combines:
      - Historical gap behaviour of the stock (60% weight)
      - Overnight news sentiment (30% weight)
      - Technical context / volatility regime (10% weight)
    """

    # Weight configuration
    HIST_WEIGHT = 0.60
    NEWS_WEIGHT = 0.30
    TECH_WEIGHT = 0.10

    def __init__(self):
        self._sentiment_scorer = None  # lazy-loaded TextBlob

    # ─── Public API ───────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        news_df: Optional[pd.DataFrame] = None,
    ) -> GapPrediction:
        """
        Predict the next-day opening gap for a stock.

        Args:
            df: OHLCV DataFrame for the stock (must have Open, High, Low, Close, Volume)
            symbol: Stock ticker (e.g. 'COALINDIA.NS')
            news_df: Optional DataFrame of recent news with columns [datetime, title, symbol]

        Returns:
            GapPrediction dataclass with all features and predicted gap
        """
        if len(df) < 30:
            logger.warning(f"{symbol}: Insufficient data ({len(df)} rows) for gap prediction")
            return GapPrediction(
                symbol=symbol,
                current_close=float(df["Close"].iloc[-1]) if len(df) > 0 else 0.0,
            )

        close = float(df["Close"].iloc[-1])
        result = GapPrediction(symbol=symbol, current_close=close)

        # 1. Historical gap features
        self._compute_historical_features(df, result)

        # 2. News-based gap scoring
        if news_df is not None and not news_df.empty:
            sym_news = news_df[news_df["symbol"] == symbol] if "symbol" in news_df.columns else news_df
            self._compute_news_features(sym_news, result)

        # 3. Technical context
        tech_bias = self._compute_technical_bias(df)

        # 4. Combine into predicted gap
        hist_gap = result.avg_gap_pct_5d  # Recent 5-day avg gap direction
        news_gap = result.news_gap_pressure
        combined_gap = (
            self.HIST_WEIGHT * hist_gap
            + self.NEWS_WEIGHT * news_gap
            + self.TECH_WEIGHT * tech_bias
        )

        result.predicted_gap_pct = combined_gap

        # Sanity clip: no stock should predict > 2% gap from TA alone
        result.predicted_gap_pct = np.clip(result.predicted_gap_pct, -0.02, 0.02)

        result.predicted_open = close * (1 + result.predicted_gap_pct)

        # Determine direction
        if combined_gap > 0.002:
            result.gap_direction = "UP"
        elif combined_gap < -0.002:
            result.gap_direction = "DOWN"
        else:
            result.gap_direction = "FLAT"

        # Compute gap risk score
        result.gap_risk_score = self._compute_risk_score(result)

        # Confidence level
        if result.gap_risk_score >= 70:
            result.gap_confidence = "HIGH"
        elif result.gap_risk_score >= 40:
            result.gap_confidence = "MED"
        else:
            result.gap_confidence = "LOW"

        return result

    def predict_batch(
        self,
        data: Dict[str, pd.DataFrame],
        news_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, GapPrediction]:
        """Predict gaps for multiple stocks."""
        results = {}
        for symbol, df in data.items():
            results[symbol] = self.predict(df, symbol, news_df)
        return results

    # ─── Historical Gap Features ──────────────────────────────────────────────

    def _compute_historical_features(self, df: pd.DataFrame, result: GapPrediction):
        """Compute gap features from historical OHLCV data."""
        # Daily gap = (Open_t - Close_{t-1}) / Close_{t-1}
        gaps = (df["Open"] / df["Close"].shift(1) - 1).dropna()

        if len(gaps) < 5:
            return

        # Rolling averages
        result.avg_gap_pct_5d = float(gaps.tail(5).mean())
        result.avg_gap_pct_10d = float(gaps.tail(10).mean()) if len(gaps) >= 10 else result.avg_gap_pct_5d
        result.avg_gap_pct_21d = float(gaps.tail(21).mean()) if len(gaps) >= 21 else result.avg_gap_pct_10d

        # Gap volatility (std dev of recent gaps)
        result.gap_volatility_10d = float(gaps.tail(10).std()) if len(gaps) >= 10 else float(gaps.std())

        # Gap frequency
        recent_10 = gaps.tail(10)
        result.gap_up_freq_10d = float((recent_10 > 0.001).mean())
        result.gap_down_freq_10d = float((recent_10 < -0.001).mean())

        # Max gaps (worst case scenarios)
        recent_21 = gaps.tail(21) if len(gaps) >= 21 else gaps
        result.max_gap_up_21d = float(recent_21.max())
        result.max_gap_down_21d = float(recent_21.min())

        # Correlation: previous day's return → next day's gap
        if len(df) >= 22:
            prev_returns = df["Close"].pct_change().iloc[-22:-1]
            next_gaps = gaps.iloc[-21:]
            min_len = min(len(prev_returns), len(next_gaps))
            if min_len >= 10:
                corr = prev_returns.iloc[-min_len:].reset_index(drop=True).corr(
                    next_gaps.iloc[-min_len:].reset_index(drop=True)
                )
                result.close_to_open_corr_21d = float(corr) if not np.isnan(corr) else 0.0

    # ─── News-Based Gap Scoring ───────────────────────────────────────────────

    def _compute_news_features(self, news_df: pd.DataFrame, result: GapPrediction):
        """Score overnight news for gap prediction."""
        if news_df.empty:
            return

        # Filter to recent articles (last 24h ideally)
        articles = news_df.copy()
        if "datetime" in articles.columns:
            articles["datetime"] = pd.to_datetime(articles["datetime"], errors="coerce")
            articles = articles.dropna(subset=["datetime"])
            # Sort by recency
            articles = articles.sort_values("datetime", ascending=False)

        if articles.empty:
            return

        result.overnight_news_count = len(articles)

        # Score each headline
        sentiments = []
        catalysts = []
        for _, row in articles.iterrows():
            title = str(row.get("title", ""))
            if not title:
                continue

            score = self._score_headline(title)
            sentiments.append(score)

            # Check for high-impact keywords
            title_lower = title.lower()
            for kw in BULLISH_KEYWORDS + BEARISH_KEYWORDS:
                if kw in title_lower:
                    result.high_impact_detected = True
                    catalysts.append(title[:80])
                    break

        if sentiments:
            result.overnight_sentiment = float(np.mean(sentiments))
            result.news_catalysts = catalysts[:3]  # top 3 catalysts

            # Compute news gap pressure:
            # sentiment × sqrt(count) × impact_multiplier
            impact_mult = 2.0 if result.high_impact_detected else 1.0
            count_factor = min(np.sqrt(len(sentiments)), 3.0)  # cap at sqrt(9)=3
            result.news_gap_pressure = result.overnight_sentiment * count_factor * impact_mult

            # Normalize to reasonable range (clip to ±1%)
            # Most NSE stocks gap 0.1-0.5% on average; >1% is rare
            result.news_gap_pressure = np.clip(result.news_gap_pressure * 0.005, -0.01, 0.01)

    def _score_headline(self, title: str) -> float:
        """
        Score a headline for sentiment.
        Uses keyword matching first, then TextBlob as fallback.
        Returns score in [-1, +1].
        """
        title_lower = title.lower()
        score = 0.0

        # Keyword-based scoring (fast, domain-specific)
        for kw in BULLISH_KEYWORDS:
            if kw in title_lower:
                score += 0.3
        for kw in BEARISH_KEYWORDS:
            if kw in title_lower:
                score -= 0.3

        if abs(score) > 0:
            return np.clip(score, -1.0, 1.0)

        # TextBlob fallback for non-keyword headlines
        try:
            if self._sentiment_scorer is None:
                from textblob import TextBlob
                self._sentiment_scorer = TextBlob
            blob = self._sentiment_scorer(title)
            return float(blob.sentiment.polarity)
        except ImportError:
            logger.debug("TextBlob not available, using neutral sentiment")
            return 0.0
        except Exception:
            return 0.0

    # ─── Technical Bias ───────────────────────────────────────────────────────

    def _compute_technical_bias(self, df: pd.DataFrame) -> float:
        """
        Compute a technical bias factor for gap direction.
        Returns a value in approximately [-0.02, +0.02].
        """
        close = df["Close"]

        # 1. Overnight return trend (recent 5-day average of gaps)
        if "Open" in df.columns and len(df) > 5:
            recent_gaps = (df["Open"] / close.shift(1) - 1).tail(5)
            trend = float(recent_gaps.mean())
        else:
            trend = 0.0

        # 2. Momentum context: 3-day price momentum
        if len(df) >= 4:
            mom_3d = float(close.iloc[-1] / close.iloc[-4] - 1)
        else:
            mom_3d = 0.0

        # 3. Volatility regime: high vol → larger potential gaps
        if len(df) >= 21:
            recent_vol = float(close.pct_change().tail(21).std())
            long_vol = float(close.pct_change().tail(60).std()) if len(df) >= 60 else recent_vol
            vol_expansion = recent_vol / (long_vol + 1e-10)
        else:
            vol_expansion = 1.0

        # Combine: momentum direction weighted by volatility expansion
        # Clipped tight — technical context is a minor input
        bias = mom_3d * 0.1 * min(vol_expansion, 1.5)
        return np.clip(bias, -0.005, 0.005)

    # ─── Risk Score ───────────────────────────────────────────────────────────

    def _compute_risk_score(self, result: GapPrediction) -> int:
        """
        Compute gap risk score (0–100).

        Higher score = more likely the gap will be large enough to
        invalidate the entry/target range.

        Factors:
          - Gap magnitude (predicted + historical volatility)
          - News impact
          - Historical gap unpredictability
        """
        score = 0

        # 1. Predicted gap magnitude (0-30 pts)
        abs_gap = abs(result.predicted_gap_pct) * 100  # as percentage
        if abs_gap > 2.0:
            score += 30
        elif abs_gap > 1.0:
            score += 20
        elif abs_gap > 0.5:
            score += 10
        elif abs_gap > 0.2:
            score += 5

        # 2. Historical gap volatility (0-25 pts)
        gap_vol = result.gap_volatility_10d * 100
        if gap_vol > 2.0:
            score += 25
        elif gap_vol > 1.0:
            score += 15
        elif gap_vol > 0.5:
            score += 8

        # 3. News impact (0-25 pts)
        if result.high_impact_detected:
            score += 20
        if result.overnight_news_count >= 5:
            score += 5
        elif result.overnight_news_count >= 2:
            score += 3

        # 4. Historical max gap severity (0-20 pts)
        max_abs_gap = max(abs(result.max_gap_up_21d), abs(result.max_gap_down_21d)) * 100
        if max_abs_gap > 3.0:
            score += 20
        elif max_abs_gap > 2.0:
            score += 12
        elif max_abs_gap > 1.0:
            score += 5

        return min(score, 100)


# ─── Convenience Functions ────────────────────────────────────────────────────

def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gap-related columns to a DataFrame (for feature engineering).
    Can be used standalone without the full GapPredictor class.
    """
    result = df.copy()

    # Basic gap
    result["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1).fillna(0)
    result["gap_abs"] = result["gap_pct"].abs()

    # Rolling gap statistics
    for window in [5, 10, 21]:
        result[f"avg_gap_{window}d"] = result["gap_pct"].rolling(window).mean().fillna(0)
        result[f"gap_vol_{window}d"] = result["gap_pct"].rolling(window).std().fillna(0)
        result[f"max_gap_up_{window}d"] = result["gap_pct"].rolling(window).max().fillna(0)
        result[f"max_gap_down_{window}d"] = result["gap_pct"].rolling(window).min().fillna(0)

    # Gap frequency (% of days with gap > 0.1%)
    result["gap_up_freq_10d"] = (
        (result["gap_pct"] > 0.001).rolling(10).mean().fillna(0)
    )
    result["gap_down_freq_10d"] = (
        (result["gap_pct"] < -0.001).rolling(10).mean().fillna(0)
    )

    # Gap fill ratio: how often does the gap get filled within the day?
    # Gap up filled if Low <= prev Close; gap down filled if High >= prev Close
    prev_close = df["Close"].shift(1)
    gap_up_mask = result["gap_pct"] > 0.001
    gap_down_mask = result["gap_pct"] < -0.001
    gap_filled = pd.Series(0.0, index=df.index)
    gap_filled[gap_up_mask] = (df["Low"][gap_up_mask] <= prev_close[gap_up_mask]).astype(float)
    gap_filled[gap_down_mask] = (df["High"][gap_down_mask] >= prev_close[gap_down_mask]).astype(float)
    result["gap_fill_rate_10d"] = gap_filled.rolling(10).mean().fillna(0)

    return result
