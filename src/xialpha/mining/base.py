"""BaseMiner 抽象基类：所有因子挖掘器的共享评估管线。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.loader import StockData
    from ..backend.base import BackendBase

logger = logging.getLogger(__name__)


class BaseMiner(ABC):
    """所有因子挖掘器的父类，提供共享的因子编译→回测→指标评估管线。"""

    def __init__(self, stock_data: StockData, backend: BackendBase) -> None:
        self.stock_data = stock_data
        self.backend = backend
        self._forward_returns: np.ndarray = backend.shift(stock_data.returns, -1)
        self._seen_expressions: set[str] = set()

    # ── 单因子评估 ──────────────────────────────────────────────────

    def _evaluate_factor(self, expression: str) -> dict | None:
        """编译并回测单个因子表达式，返回指标字典或 *None*（失败时）。

        流程:
        1. 检查去重（``self._seen_expressions``）
        2. 调用 ``compile_factor(expression)`` 编译
        3. 调用编译后的函数获取因子值
        4. 调用 ``run_backtest()`` 运行回测
        5. 调用 ``calc_metrics()`` 计算指标
        6. 返回指标字典

        失败情况返回 *None*：编译错误、回测异常、因子全 NaN。
        """
        if expression in self._seen_expressions:
            return None
        self._seen_expressions.add(expression)

        try:
            from ..factor.compiler import compile_factor

            compiled = compile_factor(expression)
            factor_values = compiled(self.stock_data, self.backend)

            if np.all(np.isnan(factor_values)):
                logger.warning("Factor values all NaN: %s", expression)
                return None

            from ..backtest.engine import run_backtest

            result = run_backtest(
                factor_values,
                self._forward_returns,
                self.stock_data.dates,
            )

            from ..backtest.metrics import calc_metrics

            metrics = calc_metrics(result)
            return metrics

        except Exception:
            logger.warning("Factor evaluation failed: %s", expression, exc_info=True)
            return None

    # ── 批量评估 ────────────────────────────────────────────────────

    def _evaluate_batch(
        self, expressions: list[str]
    ) -> list[tuple[str, dict | None]]:
        return [(expr, self._evaluate_factor(expr)) for expr in expressions]

    # ── 子类必须实现 ────────────────────────────────────────────────

    @abstractmethod
    def mine(self) -> list[tuple[str, dict]]:
        ...
