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

_PACKAGE_DIR = Path(__file__).resolve().parent


def _find_config_path() -> Path:
    env = os.environ.get("XIALPHA_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"XIALPHA_CONFIG 指定的配置文件不存在: {p}")

    search_paths = [
        Path.cwd() / "xi-alpha.toml",
        _PACKAGE_DIR.parent.parent / "xi-alpha.toml",
    ]

    for p in search_paths:
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"未找到 xi-alpha.toml，已搜索: {[str(p) for p in search_paths]}"
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
            "min_stocks_for_group": 2,
            "min_stocks_for_ic": 3,
            "ic_decay_days": 10,
        },
        "report": {
            "output_dir": "output",
            "dpi": 150,
            "top_n_display": 5,
            "figure": {"single": [16, 10], "batch": [18, 14]},
            "filename": {"single": "factor_report.png", "batch": "batch_report.png"},
            "style": {
                "colormap_groups": "RdYlGn",
                "colormap_corr": "coolwarm",
                "colormap_nav": "tab10",
                "line_color": "steelblue",
            },
        },
        "backend": {"type": "numpy"},
        "mining": {
            "gp": {
                "population_size": 100,
                "n_generations": 50,
                "crossover_prob": 0.8,
                "mutation_prob": 0.15,
                "max_depth": 4,
                "tournament_size": 3,
                "elites": 5,
                "objectives": ["ir", "sharpe"],
                "seed": 42,
            },
            "llm": {
                "api_url": "http://localhost:39001/v1/chat/completions",
                "model_name": "glm-5.1",
                "api_key": "",
                "max_iterations": 10,
                "factors_per_iteration": 5,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        },
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
PROJECT_ROOT: Path = _find_config_path().parent


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
CACHE_DIR: Path = PROJECT_ROOT / _get("data", "cache", "dir")
STOCK_POOL: list[str] = _get("data", "stock_pool", "symbols")

# ── factors ────────────────────────────────────────────────────────────
DEFAULT_WINDOW: int = _get("factors", "default_window")
CLASSIC_FACTORS: dict[str, str] = _get("factors", "classic")

# ── backtest ───────────────────────────────────────────────────────────
N_GROUPS: int = _get("backtest", "n_groups")
TRADING_DAYS_PER_YEAR: int = _get("backtest", "trading_days_per_year")
N_JOBS: int = _get("backtest", "n_jobs")
MIN_STOCKS_FOR_GROUP: int = _get("backtest", "min_stocks_for_group")
MIN_STOCKS_FOR_IC: int = _get("backtest", "min_stocks_for_ic")
IC_DECAY_DAYS: int = _get("backtest", "ic_decay_days")

# ── report ─────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = PROJECT_ROOT / _get("report", "output_dir")
DPI: int = _get("report", "dpi")
TOP_N_DISPLAY: int = _get("report", "top_n_display")
FIGSIZE_SINGLE: tuple[int, int] = tuple(_get("report", "figure", "single"))
FIGSIZE_BATCH: tuple[int, int] = tuple(_get("report", "figure", "batch"))
FILENAME_SINGLE: str = _get("report", "filename", "single")
FILENAME_BATCH: str = _get("report", "filename", "batch")
COLORMAP_GROUPS: str = _get("report", "style", "colormap_groups")
COLORMAP_CORR: str = _get("report", "style", "colormap_corr")
COLORMAP_NAV: str = _get("report", "style", "colormap_nav")
LINE_COLOR: str = _get("report", "style", "line_color")

# ── backend ────────────────────────────────────────────────────────────
BACKEND_TYPE: str = _get("backend", "type")

# ── mining.gp ─────────────────────────────────────────────────────────
GP_POPULATION_SIZE: int = _get("mining", "gp", "population_size")
GP_N_GENERATIONS: int = _get("mining", "gp", "n_generations")
GP_CROSSOVER_PROB: float = _get("mining", "gp", "crossover_prob")
GP_MUTATION_PROB: float = _get("mining", "gp", "mutation_prob")
GP_MAX_DEPTH: int = _get("mining", "gp", "max_depth")
GP_TOURNAMENT_SIZE: int = _get("mining", "gp", "tournament_size")
GP_ELITES: int = _get("mining", "gp", "elites")
GP_OBJECTIVES: list[str] = _get("mining", "gp", "objectives")
GP_SEED: int = _get("mining", "gp", "seed")

# ── mining.llm ────────────────────────────────────────────────────────
LLM_API_URL: str = _get("mining", "llm", "api_url")
LLM_MODEL_NAME: str = _get("mining", "llm", "model_name")
LLM_API_KEY: str = _get("mining", "llm", "api_key")
LLM_MAX_ITERATIONS: int = _get("mining", "llm", "max_iterations")
LLM_FACTORS_PER_ITER: int = _get("mining", "llm", "factors_per_iteration")
LLM_TEMPERATURE: float = _get("mining", "llm", "temperature")
LLM_MAX_TOKENS: int = _get("mining", "llm", "max_tokens")

# ── logging ────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("logging", "level")
LOG_FORMAT: str = _get("logging", "format")
