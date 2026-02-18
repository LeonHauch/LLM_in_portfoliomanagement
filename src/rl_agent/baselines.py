# src/rl_agent/baselines.py
"""
Classical portfolio baselines for benchmarking the PPO agent.

All baselines operate on the same data and produce the same metrics
so results are directly comparable.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def _load_returns(data_path: str, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Load parquet and compute log-return slice for the evaluation period."""
    raw = pd.read_parquet(data_path)
    adj_cols = sorted([c for c in raw.columns if c.startswith("Adj Close_")])
    prices = raw[adj_cols].copy()
    prices.columns = [c.replace("Adj Close_", "") for c in adj_cols]
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    return log_returns.iloc[start_idx:end_idx + 1]


def _portfolio_metrics(
    daily_returns: np.ndarray,
    initial_capital: float = 100_000.0,
    trading_days: int = 252,
) -> Dict:
    """Compute standard portfolio performance metrics."""
    cumulative = np.cumprod(1 + daily_returns)
    portfolio_values = initial_capital * cumulative

    total_return = float(cumulative[-1] - 1)
    n_years = len(daily_returns) / trading_days
    ann_return = float((1 + total_return) ** (1 / max(n_years, 1e-6)) - 1)
    ann_vol = float(np.std(daily_returns) * np.sqrt(trading_days))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # Sortino (downside deviation)
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = float(np.std(neg_returns) * np.sqrt(trading_days)) if len(neg_returns) > 0 else 1e-6
    sortino = ann_return / downside_std

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "portfolio_values": portfolio_values.tolist(),
        "n_periods": len(daily_returns),
    }


def equal_weight_baseline(
    data_path: str,
    start_idx: int,
    end_idx: int,
    transaction_cost: float = 0.001,
    rebalance_freq: int = 1,
    initial_capital: float = 100_000.0,
) -> Dict:
    """
    1/N equal-weight portfolio, rebalanced every `rebalance_freq` days.

    Args:
        data_path: Path to preprocessed parquet file.
        start_idx: First row index (inclusive).
        end_idx: Last row index (inclusive).
        transaction_cost: Fractional cost per unit of turnover.
        rebalance_freq: Rebalance every N days (1 = daily).
        initial_capital: Starting capital.
    """
    returns_df = _load_returns(data_path, start_idx, end_idx)
    n_assets = returns_df.shape[1]
    target_w = np.ones(n_assets) / n_assets

    daily_returns = []
    current_w = target_w.copy()

    for day in range(len(returns_df)):
        asset_returns = returns_df.iloc[day].values
        port_ret = float(np.dot(current_w, asset_returns))

        # Rebalance?
        if rebalance_freq > 0 and day % rebalance_freq == 0 and day > 0:
            turnover = float(np.sum(np.abs(current_w - target_w)))
            tc = turnover * transaction_cost
            port_ret -= tc
            current_w = target_w.copy()
        else:
            # Weights drift with returns
            new_w = current_w * np.exp(asset_returns)
            current_w = new_w / new_w.sum()

        daily_returns.append(port_ret)

    metrics = _portfolio_metrics(np.array(daily_returns), initial_capital)
    metrics["strategy"] = "equal_weight"
    metrics["rebalance_freq"] = rebalance_freq
    logger.info(
        f"Equal-weight baseline: Sharpe={metrics['sharpe_ratio']:.3f}, "
        f"Return={metrics['total_return']:.2%}"
    )
    return metrics


def buy_and_hold_baseline(
    data_path: str,
    start_idx: int,
    end_idx: int,
    initial_capital: float = 100_000.0,
) -> Dict:
    """
    Buy-and-hold equal-weight portfolio (no rebalancing).
    Equivalent to buying equal amounts at day 0 and holding.
    """
    returns_df = _load_returns(data_path, start_idx, end_idx)
    n_assets = returns_df.shape[1]
    current_w = np.ones(n_assets) / n_assets

    daily_returns = []
    for day in range(len(returns_df)):
        asset_returns = returns_df.iloc[day].values
        port_ret = float(np.dot(current_w, asset_returns))
        daily_returns.append(port_ret)
        # Drift
        new_w = current_w * np.exp(asset_returns)
        current_w = new_w / new_w.sum()

    metrics = _portfolio_metrics(np.array(daily_returns), initial_capital)
    metrics["strategy"] = "buy_and_hold"
    logger.info(
        f"Buy-and-hold baseline: Sharpe={metrics['sharpe_ratio']:.3f}, "
        f"Return={metrics['total_return']:.2%}"
    )
    return metrics


def inverse_volatility_baseline(
    data_path: str,
    start_idx: int,
    end_idx: int,
    vol_lookback: int = 60,
    rebalance_freq: int = 20,
    transaction_cost: float = 0.001,
    initial_capital: float = 100_000.0,
) -> Dict:
    """
    Inverse-volatility weighting (simple risk-parity proxy).
    Weights proportional to 1/sigma, rebalanced periodically.
    """
    raw = pd.read_parquet(data_path)
    adj_cols = sorted([c for c in raw.columns if c.startswith("Adj Close_")])
    prices = raw[adj_cols].copy()
    prices.columns = [c.replace("Adj Close_", "") for c in adj_cols]
    log_returns = np.log(prices / prices.shift(1)).fillna(0)

    # Need lookback before start for vol estimation
    safe_start = max(0, start_idx - vol_lookback)
    full_returns = log_returns.iloc[safe_start:end_idx + 1]

    eval_returns = log_returns.iloc[start_idx:end_idx + 1]
    n_assets = eval_returns.shape[1]

    # Initial weights: inverse vol from lookback period
    init_vol = full_returns.iloc[:vol_lookback].std().values
    init_vol = np.where(init_vol > 0, init_vol, 1e-6)
    current_w = (1 / init_vol) / (1 / init_vol).sum()

    daily_returns = []
    for day in range(len(eval_returns)):
        asset_rets = eval_returns.iloc[day].values
        port_ret = float(np.dot(current_w, asset_rets))

        if rebalance_freq > 0 and day % rebalance_freq == 0 and day > 0:
            # Recompute weights
            lookback_start = max(0, start_idx + day - vol_lookback - safe_start)
            lookback_end = start_idx + day - safe_start
            recent_vol = full_returns.iloc[lookback_start:lookback_end].std().values
            recent_vol = np.where(recent_vol > 0, recent_vol, 1e-6)
            target_w = (1 / recent_vol) / (1 / recent_vol).sum()

            turnover = float(np.sum(np.abs(current_w - target_w)))
            tc = turnover * transaction_cost
            port_ret -= tc
            current_w = target_w.copy()
        else:
            new_w = current_w * np.exp(asset_rets)
            current_w = new_w / new_w.sum()

        daily_returns.append(port_ret)

    metrics = _portfolio_metrics(np.array(daily_returns), initial_capital)
    metrics["strategy"] = "inverse_volatility"
    metrics["rebalance_freq"] = rebalance_freq
    logger.info(
        f"Inverse-vol baseline: Sharpe={metrics['sharpe_ratio']:.3f}, "
        f"Return={metrics['total_return']:.2%}"
    )
    return metrics


def run_all_baselines(
    data_path: str,
    start_idx: int,
    end_idx: int,
    transaction_cost: float = 0.001,
    initial_capital: float = 100_000.0,
) -> Dict[str, Dict]:
    """Run all baseline strategies and return a dict of results."""
    results = {}
    results["equal_weight_daily"] = equal_weight_baseline(
        data_path, start_idx, end_idx,
        transaction_cost=transaction_cost,
        rebalance_freq=1,
        initial_capital=initial_capital,
    )
    results["equal_weight_monthly"] = equal_weight_baseline(
        data_path, start_idx, end_idx,
        transaction_cost=transaction_cost,
        rebalance_freq=20,
        initial_capital=initial_capital,
    )
    results["buy_and_hold"] = buy_and_hold_baseline(
        data_path, start_idx, end_idx,
        initial_capital=initial_capital,
    )
    results["inverse_volatility"] = inverse_volatility_baseline(
        data_path, start_idx, end_idx,
        transaction_cost=transaction_cost,
        initial_capital=initial_capital,
    )
    return results
