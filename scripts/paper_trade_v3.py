"""V3 paper-trading runner skeleton."""

from __future__ import annotations

import argparse

from stockxpert.v3.config import load_v3_config
from stockxpert.v3.live import PaperTradeConfig, PaperTradingPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_nifty100.yaml")
    args = parser.parse_args()

    cfg = load_v3_config(args.config)
    pipeline = PaperTradingPipeline(
        PaperTradeConfig(
            threshold=cfg.live.paper_trading_threshold,
            sector_cap_pct=cfg.backtest.sector_cap_pct,
            drawdown_pause_pct=cfg.backtest.drawdown_pause_pct,
        )
    )
    print(f"Initialized V3 paper-trading pipeline at {pipeline.log_dir}")


if __name__ == "__main__":
    main()

