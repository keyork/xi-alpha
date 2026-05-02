"""核心回测引擎：分组因子评估。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, spearmanr

from .. import config


@dataclass
class BacktestResult:
    """回测结果数据类。"""

    group_returns: dict[int, np.ndarray]
    group_nav: dict[int, np.ndarray]
    long_short: np.ndarray
    long_short_nav: np.ndarray
    ic_series: np.ndarray
    dates: np.ndarray
    n_groups: int
    group_labels: np.ndarray


def run_backtest(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    dates: np.ndarray,
    n_groups: int = config.N_GROUPS,
) -> BacktestResult:
    """运行分组回测。

    Args:
        factor_values: 因子值矩阵 (T, N)
        forward_returns: 远期收益率矩阵 (T, N)
        dates: 日期序列 (T,)
        n_groups: 分组数，默认 5

    Returns:
        BacktestResult: 回测结果
    """
    T, N = factor_values.shape
    invalid = np.isnan(factor_values) | np.isnan(forward_returns)
    valid = ~invalid

    percentile = np.full((T, N), np.nan)
    group_labels = np.full((T, N), -1, dtype=np.int32)

    for t in range(T):
        row_valid = valid[t]
        n_valid = row_valid.sum()
        if n_valid < config.MIN_STOCKS_FOR_GROUP:
            continue
        ranked = rankdata(factor_values[t, row_valid], method="average")
        pct = (ranked - 1.0) / (n_valid - 1.0)
        percentile[t, row_valid] = pct

        # 反转，使第 0 组对应因子值最高（排名最高）
        groups = np.floor((1.0 - pct) * n_groups).astype(np.int32)
        groups = np.clip(groups, 0, n_groups - 1)
        group_labels[t, row_valid] = groups

    group_returns: dict[int, np.ndarray] = {}
    group_nav: dict[int, np.ndarray] = {}
    for g in range(n_groups):
        mask = group_labels == g  # (T, N) bool
        ret = np.full(T, np.nan)
        for t in range(T):
            sel = mask[t]
            if sel.any():
                ret[t] = np.mean(forward_returns[t, sel])
        group_returns[g] = ret
        nav_ret = np.where(np.isnan(ret), 0.0, ret)
        group_nav[g] = np.cumprod(1.0 + nav_ret)

    long_short = np.where(
        np.isnan(group_returns[0]) | np.isnan(group_returns[n_groups - 1]),
        np.nan,
        group_returns[0] - group_returns[n_groups - 1],
    )
    long_short_nav = group_nav[0] / group_nav[n_groups - 1]

    ic_series = np.full(T, np.nan)
    for t in range(T):
        row_valid = valid[t]
        n_valid = row_valid.sum()
        if n_valid < config.MIN_STOCKS_FOR_IC:
            continue
        corr, _ = spearmanr(
            factor_values[t, row_valid], forward_returns[t, row_valid]
        )
        ic_series[t] = corr

    return BacktestResult(
        group_returns=group_returns,
        group_nav=group_nav,
        long_short=long_short,
        long_short_nav=long_short_nav,
        ic_series=ic_series,
        dates=dates,
        n_groups=n_groups,
        group_labels=group_labels,
    )
