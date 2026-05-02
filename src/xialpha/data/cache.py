"""StockData 本地 pickle 缓存，避免重复拉取 AKShare。"""

import hashlib
import pickle

from .. import config
from .loader import StockData, load_stock_data


def _cache_key(start_date: str, end_date: str, stock_pool: list[str] | None) -> str:
    pool_hash = hashlib.md5(
        ",".join(sorted(stock_pool)).encode()
    ).hexdigest()[:12] if stock_pool else "default"
    return f"{start_date}_{end_date}_{pool_hash}.pkl"


def get_cached_or_load(
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    stock_pool: list[str] | None = None,
) -> StockData:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = _cache_key(start_date, end_date, stock_pool)
    path = config.CACHE_DIR / key

    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    data = load_stock_data(start_date, end_date, stock_pool)

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return data
