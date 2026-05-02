"""并行批量因子评估模块。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .. import config
from ..backend.base import BackendBase
from ..factor.compiler import compile_factor

from .engine import BacktestResult, run_backtest
from .metrics import calc_metrics

if TYPE_CHECKING:
    from ..data.loader import StockData


def _worker_backtest(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    dates: np.ndarray,
    n_groups: int,
) -> tuple[BacktestResult, dict]:
    """工作函数，在子进程中执行单因子的回测和指标计算。"""
    result = run_backtest(factor_values, forward_returns, dates, n_groups)
    metrics = calc_metrics(result)
    return result, metrics


def batch_evaluate(
    factor_list: list[tuple[str, str]],
    stock_data: StockData,
    backend: BackendBase,
    forward_returns: np.ndarray,
    dates: np.ndarray,
    n_groups: int = config.N_GROUPS,
    n_jobs: int = config.N_JOBS,
    progress_callback: Optional[Callable[[], None]] = None,
) -> tuple[list[tuple[str, BacktestResult, dict]], np.ndarray]:
    """批量评估多个因子。

    Args:
        factor_list: 因子列表，每个元素为 (名称, 表达式) 元组。
        stock_data: 股票数据对象。
        backend: 计算后端实例。
        forward_returns: 前向收益率矩阵。
        dates: 日期序列。
        n_groups: 分组数，默认 5。
        n_jobs: 并行进程数，默认 4。
        progress_callback: 每完成一个因子后调用的回调函数。

    Returns:
        (排序后的因子结果列表, 因子间相关系数矩阵)。
    """
    compiled: list[tuple[str, Callable, np.ndarray]] = []
    for name, expr in factor_list:
        fn = compile_factor(expr)
        factor_values = fn(stock_data, backend)
        compiled.append((name, fn, factor_values))
        if progress_callback:
            progress_callback()

    worker_args = [
        (fv, forward_returns, dates, n_groups)
        for _, _, fv in compiled
    ]

    if n_jobs <= 1 or len(worker_args) <= 1:
        results = []
        for args in worker_args:
            results.append(_worker_backtest(*args))
            if progress_callback:
                progress_callback()
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(_worker_backtest, *args) for args in worker_args
            ]
            results = []
            for f in futures:
                results.append(f.result())
                if progress_callback:
                    progress_callback()

    factor_values_list = [fv for _, _, fv in compiled]
    names = [name for name, _, _ in compiled]

    scored: list[tuple[str, BacktestResult, dict]] = []
    for i, (result, metrics) in enumerate(results):
        scored.append((names[i], result, metrics))

    scored.sort(key=lambda x: x[2]["ir"], reverse=True)

    corr_matrix = _build_corr_matrix(factor_values_list)

    return scored, corr_matrix


def _build_corr_matrix(factor_values_list: list[np.ndarray]) -> np.ndarray:
    """构建因子间逐期截面相关系数矩阵。"""
    n = len(factor_values_list)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr = _mean_cross_corr(factor_values_list[i], factor_values_list[j])
            matrix[i, j] = corr
            matrix[j, i] = corr
    return matrix


def _mean_cross_corr(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个因子向量的逐期截面相关系数均值。"""
    T = a.shape[0]
    corrs = []
    for t in range(T):
        valid = ~np.isnan(a[t]) & ~np.isnan(b[t])
        n_valid = valid.sum()
        if n_valid < 3:
            continue
        ai = a[t, valid]
        bi = b[t, valid]
        std_a = np.std(ai)
        std_b = np.std(bi)
        if std_a == 0 or std_b == 0:
            continue
        c = np.corrcoef(ai, bi)[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0
