"""回测指标计算模块。"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .. import config
from .engine import BacktestResult


def _max_drawdown(nav: np.ndarray) -> float:
    """计算最大回撤。"""
    valid = nav[~np.isnan(nav)]
    if len(valid) == 0:
        return 0.0
    cummax = np.maximum.accumulate(valid)
    drawdowns = 1.0 - valid / cummax
    return float(np.max(drawdowns))


def _max_dd_duration(nav: np.ndarray) -> int:
    """计算最大回撤持续天数。"""
    valid = nav[~np.isnan(nav)]
    if len(valid) == 0:
        return 0
    cummax = np.maximum.accumulate(valid)
    underwater = valid < cummax
    max_streak = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


def _daily_turnover(group_labels: np.ndarray) -> float:
    """计算日均换手率。"""
    T, N = group_labels.shape
    turnovers = []
    for t in range(1, T):
        valid_today = group_labels[t] >= 0
        valid_yesterday = group_labels[t - 1] >= 0
        both_valid = valid_today & valid_yesterday
        n_both = both_valid.sum()
        if n_both == 0:
            continue
        changed = group_labels[t, both_valid] != group_labels[t - 1, both_valid]
        turnovers.append(changed.sum() / n_both)
    return float(np.mean(turnovers)) if turnovers else 0.0


def _group_monotonicity(group_returns: dict[int, np.ndarray], n_groups: int) -> bool:
    """检查分组收益是否单调递减（即高分组收益高于低分组）。"""
    means = []
    for g in range(n_groups):
        r = group_returns[g]
        valid = r[~np.isnan(r)]
        means.append(np.mean(valid) if len(valid) > 0 else 0.0)
    for i in range(len(means) - 1):
        if means[i] < means[i + 1]:
            return False
    return True


def calc_metrics(result: BacktestResult) -> dict:
    """计算回测指标。

    参数:
        result: BacktestResult 对象

    返回值:
        包含 13 个指标的字典:
        - ic_mean: IC 均值
        - ic_std: IC 标准差
        - ir: 信息比率
        - ic_positive_ratio: IC 正值占比
        - annual_return: 年化收益
        - annual_vol: 年化波动率
        - sharpe: 夏普比率
        - max_drawdown: 最大回撤
        - max_dd_duration: 最大回撤持续天数
        - daily_turnover: 日均换手率
        - t_stat: t 统计量
        - t_pvalue: t 检验 p 值
        - group_monotonicity: 分组单调性
    """
    ic = result.ic_series
    ls = result.long_short

    ic_mean = float(np.nanmean(ic))
    ic_std = float(np.nanstd(ic))
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_positive_ratio = float(np.nanmean(ic > 0))

    annual_return = float(np.nanmean(ls)) * config.TRADING_DAYS_PER_YEAR
    annual_vol = float(np.nanstd(ls)) * np.sqrt(config.TRADING_DAYS_PER_YEAR)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    max_dd = _max_drawdown(result.long_short_nav)
    max_dd_dur = _max_dd_duration(result.long_short_nav)

    turnover = _daily_turnover(result.group_labels)

    valid_ls = ls[~np.isnan(ls)]
    if len(valid_ls) > 1:
        t_stat_result = stats.ttest_1samp(valid_ls, 0)
        t_stat = float(t_stat_result.statistic)
        t_pvalue = float(t_stat_result.pvalue)
    else:
        t_stat = 0.0
        t_pvalue = 1.0

    mono = _group_monotonicity(result.group_returns, result.n_groups)

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "ic_positive_ratio": ic_positive_ratio,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_dd_duration": max_dd_dur,
        "daily_turnover": turnover,
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "group_monotonicity": mono,
    }
