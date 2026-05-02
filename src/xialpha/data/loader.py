"""A 股日线数据加载，基于 AKShare，自动对齐交易日历。"""

import logging
import time

import akshare as ak
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .. import config

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    """多股票对齐面板数据容器。"""

    open: np.ndarray    # (T, N) float64
    high: np.ndarray    # (T, N) float64
    low: np.ndarray     # (T, N) float64
    close: np.ndarray   # (T, N) float64
    volume: np.ndarray  # (T, N) float64
    amount: np.ndarray  # (T, N) float64
    returns: np.ndarray  # (T, N) float64, 首行为 NaN
    dates: np.ndarray   # (T,) datetime-like
    symbols: np.ndarray  # (N,) str


def _to_daily_symbol(symbol: str) -> str:
    """将 6 位纯数字代码转换为 stock_zh_a_daily 所需的带前缀格式。

    6 开头 → sh, 其余（0/3 开头）→ sz。
    """
    if symbol.startswith("6"):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _fetch_single_stock(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = config.MAX_RETRIES,
    base_delay: float = config.BASE_DELAY,
) -> pd.DataFrame | None:
    """拉取单只股票日线数据，带指数退避重试。

    优先使用 stock_zh_a_daily 接口（东方财富），列名已为英文，
    无需额外映射。若失败则自动降级到 stock_zh_a_hist（新浪）。

    Parameters
    ----------
    symbol : str
        6 位股票代码。
    start_date : str
        起始日期，格式 "YYYY-MM-DD"。
    end_date : str
        截止日期，格式 "YYYY-MM-DD"。
    max_retries : int
        最大重试次数（含首次请求）。
    base_delay : float
        首次退避等待秒数，后续每次翻倍。
    """
    apis = [
        ("daily", _fetch_via_daily),
        ("hist", _fetch_via_hist),
    ]
    for api_name, fetcher in apis:
        for attempt in range(1, max_retries + 1):
            try:
                df = fetcher(symbol, start_date, end_date)
                if df is None or df.empty:
                    logger.warning("Empty data for %s via %s, skipping", symbol, api_name)
                    return None
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
                return df[["open", "high", "low", "close", "volume", "amount"]]
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Failed to fetch %s via %s (attempt %d/%d): %s — retrying in %.1fs",
                        symbol, api_name, attempt, max_retries, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "Failed to fetch %s via %s after %d attempts — trying next API",
                        symbol, api_name, max_retries,
                    )
    logger.warning("All APIs exhausted for %s — skipping", symbol)
    return None


def _fetch_via_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """通过 stock_zh_a_daily 接口拉取（东方财富数据源）。"""
    return ak.stock_zh_a_daily(
        symbol=_to_daily_symbol(symbol),
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=config.ADJUST,
    )


# stock_zh_a_hist 中文列名 → 英文
_HIST_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def _fetch_via_hist(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """通过 stock_zh_a_hist 接口拉取（新浪数据源，降级备用）。"""
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=config.ADJUST,
    )
    return df.rename(columns=_HIST_COL_MAP)


def load_stock_data(
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    stock_pool: list[str] | None = None,
) -> StockData:
    """加载并对齐一池 A 股的日线数据。

    Parameters
    ----------
    start_date : str
        格式 "YYYY-MM-DD"。
    end_date : str
        格式 "YYYY-MM-DD"。
    stock_pool : list[str] | None
        6 位股票代码列表，默认使用 config.STOCK_POOL。

    Returns
    -------
    StockData
        对齐后的面板数据，停牌日用 NaN 填充。
    """
    if stock_pool is None:
        stock_pool = config.STOCK_POOL

    n_stocks = len(stock_pool)
    frames: dict[str, pd.DataFrame] = {}

    for i, code in enumerate(stock_pool, 1):
        logger.info("Loading stock %d/%d: %s...", i, n_stocks, code)
        df = _fetch_single_stock(code, start_date, end_date)
        if df is not None:
            frames[code] = df
        if i < n_stocks:
            time.sleep(config.REQUEST_INTERVAL)

    if not frames:
        raise RuntimeError("No stock data could be fetched")

    all_dates: pd.DatetimeIndex = pd.DatetimeIndex(sorted(
        {d for df in frames.values() for d in df.index}
    ))

    aligned: dict[str, pd.DataFrame] = {}
    for code, df in frames.items():
        aligned[code] = df.reindex(all_dates)

    fields = ["open", "high", "low", "close", "volume", "amount"]
    panels: dict[str, np.ndarray] = {}
    for field in fields:
        cols = [aligned[code][field].to_numpy(dtype=np.float64) for code in aligned]
        panels[field] = np.column_stack(cols)

    # returns[t] = close[t] / close[t-1] - 1，首行为 NaN
    close = panels["close"]
    returns = np.full_like(close, np.nan)
    returns[1:] = close[1:] / close[:-1] - 1.0

    codes = list(aligned.keys())

    return StockData(
        open=panels["open"],
        high=panels["high"],
        low=panels["low"],
        close=panels["close"],
        volume=panels["volume"],
        amount=panels["amount"],
        returns=returns,
        dates=all_dates.values,
        symbols=np.array(codes, dtype=str),
    )
