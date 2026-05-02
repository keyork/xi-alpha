"""NumPy 后端：NaN 安全的滚动、截面和逐元素运算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .base import BackendBase


class NumPyBackend(BackendBase):

    # ── 时序算子（axis=0）────────────────────────────────────────────

    def rolling_mean(self, x: np.ndarray, window: int) -> np.ndarray:
        return self._rolling_agg(x, window, "mean")

    def rolling_std(self, x: np.ndarray, window: int) -> np.ndarray:
        return self._rolling_agg(x, window, "std")

    def rolling_max(self, x: np.ndarray, window: int) -> np.ndarray:
        return self._rolling_agg(x, window, "max")

    def rolling_min(self, x: np.ndarray, window: int) -> np.ndarray:
        return self._rolling_agg(x, window, "min")

    def rolling_sum(self, x: np.ndarray, window: int) -> np.ndarray:
        return self._rolling_agg(x, window, "sum")

    def rolling_corr(
        self, x: np.ndarray, y: np.ndarray, window: int
    ) -> np.ndarray:
        T, N = x.shape
        out = np.full((T, N), np.nan)
        for j in range(N):
            sx = pd.Series(x[:, j])
            sy = pd.Series(y[:, j])
            out[:, j] = sx.rolling(window).corr(sy).values
        return out

    def shift(self, x: np.ndarray, periods: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        if periods > 0:
            if periods < x.shape[0]:
                out[periods:] = x[:-periods]
        elif periods < 0:
            p = -periods
            if p < x.shape[0]:
                out[:x.shape[0] - p] = x[p:]
        else:
            out[:] = x
        return out

    def pct_change(self, x: np.ndarray, periods: int) -> np.ndarray:
        shifted = self.shift(x, periods)
        return self.div(x, shifted) - 1.0

    def diff(self, x: np.ndarray, periods: int) -> np.ndarray:
        return x - self.shift(x, periods)

    def cumsum(self, x: np.ndarray) -> np.ndarray:
        return np.nancumsum(x, axis=0)

    def cumprod(self, x: np.ndarray) -> np.ndarray:
        return np.nancumprod(x, axis=0)

    # ── 截面算子（axis=1）────────────────────────────────────────────

    def rank(self, x: np.ndarray) -> np.ndarray:
        T, N = x.shape
        out = np.full((T, N), np.nan)
        for i in range(T):
            row = x[i]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                continue
            r = rankdata(row[valid], method="average")
            out[i, valid] = (r - 1.0) / (valid.sum() - 1.0)
        return out

    def zscore(self, x: np.ndarray) -> np.ndarray:
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, ddof=1, keepdims=True)
        std = np.where(std == 0, np.nan, std)
        return (x - mean) / std

    def demean(self, x: np.ndarray) -> np.ndarray:
        mean = np.nanmean(x, axis=1, keepdims=True)
        return x - mean

    def clip(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(x, lo, hi)

    # ── 逐元素运算 ───────────────────────────────────────────────────

    def add(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
        return x + np.asarray(y)

    def sub(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
        return x - np.asarray(y)

    def mul(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
        return x * np.asarray(y)

    def div(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
        y_arr = np.asarray(y, dtype=x.dtype)
        out = np.full_like(x, np.nan)
        mask = y_arr != 0
        np.divide(x, y_arr, out=out, where=mask)
        return out

    def log(self, x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, np.nan)
        valid = x > 0
        out[valid] = np.log(x[valid])
        return out

    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    def sign(self, x: np.ndarray) -> np.ndarray:
        out = np.sign(x)
        out[np.isnan(x)] = np.nan
        return out

    def where(
        self, condition: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return np.where(condition, x, y)

    def nanmean(self, x: np.ndarray) -> np.ndarray:
        return np.nanmean(x, axis=1)

    # ── 相关性 ───────────────────────────────────────────────────────

    def cross_corr(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        out = np.full(T, np.nan)
        for i in range(T):
            mask = ~np.isnan(x[i]) & ~np.isnan(y[i])
            if mask.sum() < 2:
                continue
            corr = np.corrcoef(x[i, mask], y[i, mask])
            out[i] = corr[0, 1]
        return out

    # ── 辅助方法 ─────────────────────────────────────────────────────

    def _rolling_agg(
        self, x: np.ndarray, window: int, func: str
    ) -> np.ndarray:
        T, N = x.shape
        out = np.full((T, N), np.nan)
        for j in range(N):
            s = pd.Series(x[:, j])
            out[:, j] = getattr(s.rolling(window), func)().values
        return out
