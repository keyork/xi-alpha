"""xi-alpha 全局配置加载器。

从项目根目录的 xi-alpha.toml 读取所有配置项，提供模块级属性供各子模块使用。
支持通过 XIALPHA_CONFIG 环境变量指定自定义配置文件路径。
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent

_CONFIG_SEARCH_PATHS = [
    Path.cwd() / "xi-alpha.toml",
    ROOT_DIR.parent.parent / "xi-alpha.toml",
]


def _find_config_path() -> Path:
    env = os.environ.get("XIALPHA_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"XIALPHA_CONFIG 指定的配置文件不存在: {p}")

    for p in _CONFIG_SEARCH_PATHS:
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"未找到 xi-alpha.toml，已搜索: {[str(p) for p in _CONFIG_SEARCH_PATHS]}"
    )


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _defaults() -> dict[str, Any]:
    return {
        "data": {
            "start_date": "2021-01-01",
            "end_date": "2024-12-31",
            "request_interval": 1.0,
            "max_retries": 3,
            "base_delay": 2.0,
            "adjust": "qfq",
            "cache": {"dir": ".cache"},
            "stock_pool": {"symbols": []},
        },
        "factors": {
            "default_window": 20,
            "classic": {},
        },
        "backtest": {
            "n_groups": 5,
            "trading_days_per_year": 252,
            "n_jobs": 4,
        },
        "report": {
            "output_dir": "output",
            "dpi": 150,
            "top_n_display": 5,
            "figure": {"single": [16, 10], "batch": [18, 14]},
            "filename": {"single": "factor_report.png", "batch": "batch_report.png"},
        },
        "backend": {"type": "numpy"},
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        },
    }


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or _find_config_path()
    logger.info("加载配置文件: %s", config_path)
    with open(config_path, "rb") as f:
        user_cfg = tomllib.load(f)
    return _deep_merge(_defaults(), user_cfg)


_CFG = load_config()


def _get(*keys: str) -> Any:
    node = _CFG
    for k in keys:
        node = node[k]
    return node


def reload(path: Path | None = None) -> None:
    global _CFG
    _CFG = load_config(path)


# ── data ───────────────────────────────────────────────────────────────
START_DATE: str = _get("data", "start_date")
END_DATE: str = _get("data", "end_date")
REQUEST_INTERVAL: float = _get("data", "request_interval")
MAX_RETRIES: int = _get("data", "max_retries")
BASE_DELAY: float = _get("data", "base_delay")
ADJUST: str = _get("data", "adjust")
CACHE_DIR: Path = ROOT_DIR / _get("data", "cache", "dir")
STOCK_POOL: list[str] = _get("data", "stock_pool", "symbols")

# ── factors ────────────────────────────────────────────────────────────
DEFAULT_WINDOW: int = _get("factors", "default_window")
CLASSIC_FACTORS: dict[str, str] = _get("factors", "classic")

# ── backtest ───────────────────────────────────────────────────────────
N_GROUPS: int = _get("backtest", "n_groups")
TRADING_DAYS_PER_YEAR: int = _get("backtest", "trading_days_per_year")
N_JOBS: int = _get("backtest", "n_jobs")

# ── report ─────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = ROOT_DIR / _get("report", "output_dir")
DPI: int = _get("report", "dpi")
TOP_N_DISPLAY: int = _get("report", "top_n_display")
FIGSIZE_SINGLE: tuple[int, int] = tuple(_get("report", "figure", "single"))
FIGSIZE_BATCH: tuple[int, int] = tuple(_get("report", "figure", "batch"))
FILENAME_SINGLE: str = _get("report", "filename", "single")
FILENAME_BATCH: str = _get("report", "filename", "batch")

# ── backend ────────────────────────────────────────────────────────────
BACKEND_TYPE: str = _get("backend", "type")

# ── logging ────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("logging", "level")
LOG_FORMAT: str = _get("logging", "format")
