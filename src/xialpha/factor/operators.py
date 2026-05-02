"""算子注册表：算子名 → 后端方法的薄包装。"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..backend.base import BackendBase

# ── 公共注册表 ─────────────────────────────────────────────────────

OPERATOR_REGISTRY: dict[str, Callable] = {}

DATA_FIELDS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "returns",
]


def _register(name: str):
    def _decorator(fn: Callable) -> Callable:
        OPERATOR_REGISTRY[name] = fn
        return fn

    return _decorator


# ── 滚动算子（数据, 窗口）─────────────────────────────────────────


@_register("rolling_mean")
def _rolling_mean(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, window = args
    return backend.rolling_mean(x, int(window))


@_register("rolling_std")
def _rolling_std(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, window = args
    return backend.rolling_std(x, int(window))


@_register("rolling_max")
def _rolling_max(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, window = args
    return backend.rolling_max(x, int(window))


@_register("rolling_min")
def _rolling_min(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, window = args
    return backend.rolling_min(x, int(window))


@_register("rolling_sum")
def _rolling_sum(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, window = args
    return backend.rolling_sum(x, int(window))


@_register("rolling_corr")
def _rolling_corr(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, y, window = args
    return backend.rolling_corr(x, y, int(window))


# ── 平移 / 变化算子（数据, 周期）──────────────────────────────────


@_register("shift")
def _shift(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, periods = args
    return backend.shift(x, int(periods))


@_register("pct_change")
def _pct_change(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, periods = args
    return backend.pct_change(x, int(periods))


@_register("diff")
def _diff(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    x, periods = args
    return backend.diff(x, int(periods))


# ── 截面算子（数据）──────────────────────────────────────────────


@_register("rank")
def _rank(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    (x,) = args
    return backend.rank(x)


@_register("zscore")
def _zscore(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    (x,) = args
    return backend.zscore(x)


@_register("demean")
def _demean(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    (x,) = args
    return backend.demean(x)


# ── 逐元素算子（数据）─────────────────────────────────────────────


@_register("log")
def _log(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    (x,) = args
    return backend.log(x)


@_register("abs")
def _abs(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    (x,) = args
    return backend.abs(x)


@_register("sign")
def _sign(*args, stock_data=None, backend: BackendBase | None = None) -> np.ndarray:
    if backend is None:
            raise TypeError("backend is required")
    (x,) = args
    return backend.sign(x)
