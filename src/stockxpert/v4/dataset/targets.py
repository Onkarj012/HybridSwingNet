"""Cost-adjusted triple-barrier targets for V4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class TradeAction(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = 2


@dataclass(frozen=True)
class IndiaDeliveryCosts:
    brokerage_bps: float = 0.0
    stt_buy_bps: float = 0.0
    stt_sell_bps: float = 10.0
    exchange_bps: float = 0.345
    sebi_bps: float = 0.01
    stamp_buy_bps: float = 1.5
    gst_rate: float = 0.18
    slippage_bps: float = 2.0

    def round_trip_bps(self, side: str = "LONG") -> float:
        taxable_bps = (2.0 * self.brokerage_bps) + (2.0 * self.exchange_bps) + (2.0 * self.sebi_bps)
        gst_bps = taxable_bps * self.gst_rate
        stt_bps = self.stt_buy_bps + self.stt_sell_bps
        stamp_bps = self.stamp_buy_bps if side.upper() == "LONG" else 0.0
        return float(stt_bps + stamp_bps + taxable_bps + gst_bps + (2.0 * self.slippage_bps))


DEFAULT_INDIA_DELIVERY_COSTS = IndiaDeliveryCosts()


@dataclass(frozen=True)
class TripleBarrierTarget:
    raw_return: float
    market_excess_return: float
    net_return_long: float
    net_return_short: float
    trade_action: TradeAction
    mfe: float
    mae: float
    cost_bps: float
    sample_weight: float
    exit_index: int
    exit_reason: str


def _require_ohlc(frame: pd.DataFrame) -> None:
    missing = [col for col in ("Open", "High", "Low", "Close") if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns for V4 target construction: {missing}")


def _atr_at(frame: pd.DataFrame, t: int, fallback_pct: float = 0.02) -> float:
    close = float(frame.iloc[t]["Close"])
    if "atr_14" in frame.columns and pd.notna(frame.iloc[t]["atr_14"]):
        atr = float(frame.iloc[t]["atr_14"])
        if atr > 0:
            return atr
    return close * fallback_pct


def cost_adjusted_triple_barrier(
    frame: pd.DataFrame,
    t: int,
    horizon: int,
    target_atr: float = 2.5,
    stop_atr: float = 1.5,
    flat_cost_buffer_bps: float = 20.0,
    costs: IndiaDeliveryCosts = DEFAULT_INDIA_DELIVERY_COSTS,
    market_frame: pd.DataFrame | None = None,
) -> TripleBarrierTarget:
    """Build a target using only bars from ``t+1`` through ``t+horizon``.

    The sample date ``t`` is the prediction date. Entry is the next session's
    open, and all barrier checks are made on and after that entry bar.
    """
    _require_ohlc(frame)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if t + horizon >= len(frame):
        raise IndexError("not enough future bars for requested V4 horizon")

    entry_idx = t + 1
    end_idx = t + horizon
    entry_open = float(frame.iloc[entry_idx]["Open"])
    atr = _atr_at(frame, t)
    long_target = entry_open + target_atr * atr
    long_stop = entry_open - stop_atr * atr
    short_target = entry_open - target_atr * atr
    short_stop = entry_open + stop_atr * atr

    exit_idx = end_idx
    exit_reason = "horizon"
    exit_price = float(frame.iloc[end_idx]["Close"])

    future = frame.iloc[entry_idx : end_idx + 1]
    for idx, row in future.iterrows():
        pos = frame.index.get_loc(idx)
        high = float(row["High"])
        low = float(row["Low"])
        if high >= long_target:
            exit_idx = int(pos)
            exit_reason = "long_target"
            exit_price = long_target
            break
        if low <= long_stop:
            exit_idx = int(pos)
            exit_reason = "long_stop"
            exit_price = long_stop
            break

    long_gross = (exit_price / entry_open) - 1.0

    short_exit_idx = end_idx
    short_exit_reason = "horizon"
    short_exit_price = float(frame.iloc[end_idx]["Close"])
    for idx, row in future.iterrows():
        pos = frame.index.get_loc(idx)
        high = float(row["High"])
        low = float(row["Low"])
        if low <= short_target:
            short_exit_idx = int(pos)
            short_exit_reason = "short_target"
            short_exit_price = short_target
            break
        if high >= short_stop:
            short_exit_idx = int(pos)
            short_exit_reason = "short_stop"
            short_exit_price = short_stop
            break

    short_gross = (entry_open / short_exit_price) - 1.0
    cost_long = costs.round_trip_bps("LONG") / 10_000.0
    cost_short = costs.round_trip_bps("SHORT") / 10_000.0
    net_long = long_gross - cost_long
    net_short = short_gross - cost_short

    highs = future["High"].to_numpy(dtype=np.float64)
    lows = future["Low"].to_numpy(dtype=np.float64)
    mfe = float(np.max(highs / entry_open - 1.0))
    mae = float(np.min(lows / entry_open - 1.0))

    raw_return = float(frame.iloc[end_idx]["Close"] / entry_open - 1.0)
    market_excess = raw_return
    if market_frame is not None:
        _require_ohlc(market_frame)
        market_entry = float(market_frame.iloc[entry_idx]["Open"])
        market_exit = float(market_frame.iloc[end_idx]["Close"])
        market_excess = raw_return - (market_exit / market_entry - 1.0)

    flat_buffer = flat_cost_buffer_bps / 10_000.0
    if max(net_long, net_short) <= flat_buffer:
        action = TradeAction.FLAT
    elif net_long >= net_short:
        action = TradeAction.LONG
        exit_idx = exit_idx
    else:
        action = TradeAction.SHORT
        exit_idx = short_exit_idx
        exit_reason = short_exit_reason

    return TripleBarrierTarget(
        raw_return=float(raw_return),
        market_excess_return=float(market_excess),
        net_return_long=float(net_long),
        net_return_short=float(net_short),
        trade_action=action,
        mfe=mfe,
        mae=mae,
        cost_bps=costs.round_trip_bps("LONG"),
        sample_weight=float(1.0 + min(abs(max(net_long, net_short)) * 20.0, 4.0)),
        exit_index=int(exit_idx),
        exit_reason=exit_reason,
    )


def attach_v4_targets(
    frame: pd.DataFrame,
    horizon: int,
    target_atr: float = 2.5,
    stop_atr: float = 1.5,
    flat_cost_buffer_bps: float = 20.0,
    costs: IndiaDeliveryCosts = DEFAULT_INDIA_DELIVERY_COSTS,
    market_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a copy with V4 target columns attached."""
    out = frame.copy()
    columns = {
        "raw_return": [],
        "market_excess_return": [],
        "net_return_long": [],
        "net_return_short": [],
        "trade_action": [],
        "mfe": [],
        "mae": [],
        "cost_bps": [],
        "sample_weight": [],
    }
    idxs = []
    for t in range(0, len(frame) - horizon):
        target = cost_adjusted_triple_barrier(
            frame,
            t,
            horizon=horizon,
            target_atr=target_atr,
            stop_atr=stop_atr,
            flat_cost_buffer_bps=flat_cost_buffer_bps,
            costs=costs,
            market_frame=market_frame,
        )
        idxs.append(frame.index[t])
        for key in columns:
            value = getattr(target, key)
            columns[key].append(int(value) if key == "trade_action" else value)

    for key, values in columns.items():
        out[key] = np.nan
        out.loc[idxs, key] = values
    return out
