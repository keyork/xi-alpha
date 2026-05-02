"""Rich 终端显示模块 — 美化 xi-alpha 流水线输出。"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .data.loader import StockData

_console = Console()

_STYLES = {
    "data": "cyan",
    "factor": "magenta",
    "backtest": "yellow",
    "report": "green",
    "done": "bold bright_green",
}

_METRIC_LABELS = {
    "ic_mean": "IC 均值",
    "ir": "IR (信息比率)",
    "sharpe": "Sharpe",
    "max_drawdown": "最大回撤",
    "daily_turnover": "日换手率",
    "t_stat": "t-statistic",
    "t_pvalue": "p-value",
    "n_periods": "有效期数",
}


def setup_logging(level: str = "INFO") -> None:
    handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    if root.handlers:
        root.handlers.clear()
    root.addHandler(handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.captureWarnings(True)
    import warnings as _w
    _w.filterwarnings("ignore", message="pkg_resources is deprecated")


def print_header(stock_data: StockData, n_factors: int, mode: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_stocks = stock_data.close.shape[1]
    n_days = stock_data.close.shape[0]
    dates = stock_data.dates
    start_str = str(dates[0])[:10] if len(dates) else "?"
    end_str = str(dates[-1])[:10] if len(dates) else "?"

    content = Text.from_markup(
        f"[white]📊 股票数 / Stocks:[/]      [bold cyan]{n_stocks}[/]\n"
        f"[white]📅 交易日 / Days:[/]        [bold cyan]{n_days}[/]  "
        f"([dim]{start_str} → {end_str}[/])\n"
        f"[white]🧪 因子数 / Factors:[/]     [bold magenta]{n_factors}[/]  [dim]({mode})[/]\n"
        f"[white]🕐 时间 / Time:[/]          [dim]{now}[/]"
    )
    _console.print(Panel(
        content,
        title="[bold bright_blue]ξ-alpha  向量化因子研究引擎[/]",
        border_style="bright_blue",
        padding=(1, 2),
    ))
    _console.print()


def create_progress(label: str, total: int) -> Progress:
    return Progress(
        SpinnerColumn("dots"),
        TextColumn(f"[bold]{label}"),
        BarColumn(bar_width=32, complete_style="bright_cyan", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    )


def print_step(stage: str, message: str) -> None:
    color = _STYLES.get(stage, "white")
    _console.print(f"  [{color}]▸[/{color}] {message}")


def print_single_summary(factor_expr: str, metrics: dict) -> None:
    _console.print()
    _console.print(Rule("📊 单因子报告 | Single Factor Report", style="bright_cyan"))

    table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
    table.add_column("指标", style="white", min_width=20)
    table.add_column("值", justify="right", style="bold")

    for key, label in _METRIC_LABELS.items():
        val = metrics.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)

        if key in ("ic_mean", "ir", "sharpe"):
            if isinstance(val, float) and val > 0:
                val_str = f"[green]{val_str}[/]"
            elif isinstance(val, float) and val < 0:
                val_str = f"[red]{val_str}[/]"

        if key == "t_pvalue" and isinstance(val, float):
            if val < 0.05:
                val_str = f"[bold green]{val_str} ★[/]"
            else:
                val_str = f"[dim]{val_str}[/]"

        table.add_row(label, val_str)

    _console.print(table)
    _console.print(f"\n  [dim]因子表达式:[/] {factor_expr}")
    _console.print()


def print_batch_summary(results: list[tuple]) -> None:
    _console.print()
    _console.print(Rule("📊 批量因子报告 | Batch Factor Report", style="bright_cyan"))

    table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
    table.add_column("因子 / Factor", style="white", min_width=25)
    table.add_column("IR", justify="right")
    table.add_column("IC 均值", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("最大回撤", justify="right")
    table.add_column("换手率", justify="right")

    for name, _result, m in results:
        ir = m.get("ir", float("nan"))
        ic = m.get("ic_mean", float("nan"))
        sr = m.get("sharpe", float("nan"))
        dd = m.get("max_drawdown", float("nan"))
        to = m.get("daily_turnover", float("nan"))

        ir_s = _fmt_val(ir, "ir")
        ic_s = _fmt_val(ic, "ic")
        sr_s = _fmt_val(sr, "sharpe")
        dd_s = f"{dd:.4f}" if dd >= 0 else f"[red]{dd:.4f}[/]"
        to_s = f"{to:.4f}"

        table.add_row(name[:30], ir_s, ic_s, sr_s, dd_s, to_s)

    _console.print(table)
    _console.print()


def print_done(elapsed: float) -> None:
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{m}m {s}s" if h == 0 else f"{h}h {m}m {s}s"
    _console.print()
    _console.print(Panel(
        f"[bold green]✅ 完成[/]  [dim]耗时 {elapsed_str}[/]",
        border_style="green",
        padding=(0, 2),
    ))


def _fmt_val(v: float, key: str) -> str:
    s = f"{v:.4f}" if not (v != v) else "  N/A"
    if v > 0:
        return f"[green]{s}[/]"
    if v < 0:
        return f"[red]{s}[/]"
    return s


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start
