"""向量化的 (T×N) 数组操作抽象后端接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BackendBase(ABC):
    """二维数组操作 (T×N) 的抽象后端接口。"""

    # ── 时序算子（axis=0，时间轴）─────────────────────────────────────

    @abstractmethod
    def rolling_mean(self, x: np.ndarray, window: int) -> np.ndarray: ...

    @abstractmethod
    def rolling_std(self, x: np.ndarray, window: int) -> np.ndarray: ...

    @abstractmethod
    def rolling_max(self, x: np.ndarray, window: int) -> np.ndarray: ...

    @abstractmethod
    def rolling_min(self, x: np.ndarray, window: int) -> np.ndarray: ...

    @abstractmethod
    def rolling_sum(self, x: np.ndarray, window: int) -> np.ndarray: ...

    @abstractmethod
    def rolling_corr(
        self, x: np.ndarray, y: np.ndarray, window: int
    ) -> np.ndarray: ...

    @abstractmethod
    def shift(self, x: np.ndarray, periods: int) -> np.ndarray: ...

    @abstractmethod
    def pct_change(self, x: np.ndarray, periods: int) -> np.ndarray: ...

    @abstractmethod
    def diff(self, x: np.ndarray, periods: int) -> np.ndarray: ...

    @abstractmethod
    def cumsum(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def cumprod(self, x: np.ndarray) -> np.ndarray: ...

    # ── 截面算子（axis=1，股票轴）──────────────────────────────────────

    @abstractmethod
    def rank(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def zscore(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def demean(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def clip(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray: ...

    # ── 逐元素运算 ───────────────────────────────────────────────────

    @abstractmethod
    def add(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray: ...

    @abstractmethod
    def sub(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray: ...

    @abstractmethod
    def mul(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray: ...

    @abstractmethod
    def div(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray: ...

    @abstractmethod
    def log(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def abs(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def sign(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def where(
        self, condition: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray: ...

    @abstractmethod
    def nanmean(self, x: np.ndarray) -> np.ndarray: ...

    # ── 相关性 ───────────────────────────────────────────────────────

    @abstractmethod
    def cross_corr(
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray: ...
