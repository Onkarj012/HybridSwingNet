"""
Analyst Target Price Scraper for StockXpert.

Fetches consensus analyst target prices from:
  1. yfinance (primary) — via Ticker.info fields
  2. Fallback to cached data if fetching fails

Caches results per stock with 24-hour TTL.

Usage:
    scraper = TargetScraper(cache_dir="cache/targets")
    targets = scraper.fetch_targets(["HDFCBANK.NS", "RELIANCE.NS"])
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("stockxpert.mdata.target_scraper")

CACHE_TTL_HOURS = 24


@dataclass
class AnalystTarget:
    """Consensus analyst target for a single stock."""
    symbol: str
    current_price: float = 0.0
    consensus_target: float = 0.0       # Mean target price
    target_high: float = 0.0            # Highest analyst target
    target_low: float = 0.0             # Lowest analyst target
    target_upside_pct: float = 0.0      # % upside to consensus target
    analyst_count: int = 0              # Number of analysts covering
    recommendation: str = "N/A"         # buy / hold / sell / strong_buy / etc.
    fetched_at: str = ""                # ISO timestamp
    source: str = "yfinance"


class TargetScraper:
    """
    Fetches and caches analyst consensus target prices.

    Primary source: yfinance Ticker.info (fields: targetMeanPrice,
    targetHighPrice, targetLowPrice, numberOfAnalystOpinions, recommendationKey)
    """

    def __init__(self, cache_dir: str = "cache/targets"):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            self.cache_dir = Path("/tmp/stockxpert_targets")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache dir fallback to {self.cache_dir}")

    def fetch_targets(
        self,
        symbols: List[str],
        force_refresh: bool = False,
    ) -> Dict[str, AnalystTarget]:
        """
        Fetch analyst targets for a list of symbols.

        Args:
            symbols: List of stock tickers (e.g. ['HDFCBANK.NS'])
            force_refresh: If True, bypass cache

        Returns:
            Dict of symbol → AnalystTarget
        """
        results = {}
        to_fetch = []

        for sym in symbols:
            if not force_refresh:
                cached = self._load_cache(sym)
                if cached is not None:
                    results[sym] = cached
                    continue
            to_fetch.append(sym)

        if to_fetch:
            logger.info(f"Fetching analyst targets for {len(to_fetch)} stocks...")
            for sym in to_fetch:
                target = self._fetch_yfinance(sym)
                if target is not None:
                    results[sym] = target
                    self._save_cache(sym, target)
                # Small delay to avoid rate limiting
                time.sleep(0.1)

        logger.info(f"Analyst targets: {len(results)}/{len(symbols)} stocks covered")
        return results

    def _fetch_yfinance(self, symbol: str) -> Optional[AnalystTarget]:
        """Fetch analyst target from yfinance Ticker.info."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                logger.warning(f"{symbol}: No info available from yfinance")
                return None

            current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            consensus = info.get("targetMeanPrice", 0)
            target_high = info.get("targetHighPrice", 0)
            target_low = info.get("targetLowPrice", 0)
            analyst_count = info.get("numberOfAnalystOpinions", 0)
            rec_key = info.get("recommendationKey", "N/A")

            if not consensus or consensus == 0:
                logger.debug(f"{symbol}: No consensus target available")
                return None

            upside = (consensus / current_price - 1) * 100 if current_price > 0 else 0

            target = AnalystTarget(
                symbol=symbol,
                current_price=float(current_price or 0),
                consensus_target=float(consensus),
                target_high=float(target_high or consensus),
                target_low=float(target_low or consensus),
                target_upside_pct=round(upside, 2),
                analyst_count=int(analyst_count or 0),
                recommendation=str(rec_key),
                fetched_at=datetime.now().isoformat(),
                source="yfinance",
            )

            logger.debug(
                f"{symbol}: Target ₹{consensus:.0f} ({upside:+.1f}%), "
                f"{analyst_count} analysts, rec={rec_key}"
            )
            return target

        except Exception as e:
            logger.error(f"{symbol}: Failed to fetch yfinance targets: {e}")
            return None

    def _cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        safe_name = symbol.replace(".", "_").replace("&", "_")
        return self.cache_dir / f"{safe_name}.json"

    def _load_cache(self, symbol: str) -> Optional[AnalystTarget]:
        """Load cached target if valid (within TTL)."""
        path = self._cache_path(symbol)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
            if datetime.now() - fetched_at > timedelta(hours=CACHE_TTL_HOURS):
                logger.debug(f"{symbol}: Cache expired")
                return None

            return AnalystTarget(**data)
        except Exception as e:
            logger.debug(f"{symbol}: Cache read error: {e}")
            return None

    def _save_cache(self, symbol: str, target: AnalystTarget):
        """Save target to cache."""
        try:
            path = self._cache_path(symbol)
            path.write_text(json.dumps(asdict(target), indent=2))
        except Exception as e:
            logger.debug(f"{symbol}: Cache write error: {e}")


# ─── Convenience ──────────────────────────────────────────────────────────────

def format_analyst_table(targets: Dict[str, AnalystTarget]) -> str:
    """Format analyst targets as a markdown table for reports."""
    if not targets:
        return "_No analyst consensus data available._\n"

    # Sort by upside descending
    sorted_targets = sorted(
        targets.values(),
        key=lambda t: t.target_upside_pct,
        reverse=True,
    )

    lines = [
        "| Stock | Current | Consensus Target | Upside | Analysts | Rec |",
        "| ----- | ------: | ---------------: | -----: | :------: | :-: |",
    ]

    for t in sorted_targets:
        if t.analyst_count == 0:
            continue
        name = t.symbol.replace(".NS", "")
        rec_emoji = {
            "buy": "🟢", "strong_buy": "🟢🟢",
            "hold": "🟡", "sell": "🔴", "strong_sell": "🔴🔴",
        }.get(t.recommendation, "⚪")

        lines.append(
            f"| **{name}** | ₹{t.current_price:,.0f} | ₹{t.consensus_target:,.0f} "
            f"| {t.target_upside_pct:+.1f}% | {t.analyst_count} | {rec_emoji} {t.recommendation} |"
        )

    return "\n".join(lines) + "\n"
