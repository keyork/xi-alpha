"""xi-alpha 因子研究流水线命令行入口。"""

import argparse
import logging
import sys

from . import config
from .display import Timer, setup_logging, print_header, print_step, print_done

setup_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xi-alpha",
        description="向量化因子研究引擎 — data → compile → backtest → report",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="Single factor expression, e.g. 'rank(rolling_mean(close, 5) / rolling_mean(close, 20))'",
    )
    parser.add_argument(
        "--factors-file",
        type=str,
        default=None,
        help="Path to a file with one factor expression per line (format: 'name: expr' or just 'expr')",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=config.START_DATE,
        help=f"Start date (default: {config.START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=config.END_DATE,
        help=f"End date (default: {config.END_DATE})",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=config.N_GROUPS,
        help=f"Number of quantile groups (default: {config.N_GROUPS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Ignore cache and force re-fetch data",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        default=False,
        help="Run all classic factors from the library",
    )
    parser.add_argument(
        "--mine-gp",
        action="store_true",
        default=False,
        help="GP+NSGA-II 因子挖掘",
    )
    parser.add_argument(
        "--mine-llm",
        action="store_true",
        default=False,
        help="LLM Agent 因子挖掘",
    )
    return parser


def _load_data(args: argparse.Namespace):
    from .data.loader import load_stock_data
    from .data.cache import get_cached_or_load

    if args.no_cache:
        print_step("data", "正在获取数据（缓存已禁用）...")
        return load_stock_data(args.start, args.end)
    print_step("data", "正在加载数据（使用缓存）...")
    return get_cached_or_load(args.start, args.end)


def _parse_factors_file(path: str) -> list[tuple[str, str]]:
    factors: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                name, expr = line.split(":", 1)
                factors.append((name.strip(), expr.strip()))
            else:
                factors.append((line[:40], line))
    return factors


def run_pipeline(args: argparse.Namespace) -> None:
    if args.mine_gp or args.mine_llm:
        _run_mining(args)
        return

    if not any([args.factor, args.factors_file, args.classic]):
        from .display import _console
        _console.print("[bold red]错误: 请提供 --factor, --factors-file, --classic, --mine-gp 或 --mine-llm 之一[/]")
        _console.print("[dim]运行 --help 查看用法[/]")
        sys.exit(1)

    from .backend.auto import get_backend
    from .display import create_progress

    timer = Timer()

    stock_data = _load_data(args)
    backend = get_backend()
    forward_returns = backend.shift(stock_data.returns, -1)

    if args.factor:
        _run_single(args, stock_data, backend, forward_returns)
    else:
        _run_batch(args, stock_data, backend, forward_returns)

    print_done(timer.elapsed)


def _run_single(
    args: argparse.Namespace,
    stock_data,
    backend,
    forward_returns,
) -> None:
    from .factor.compiler import compile_factor
    from .backtest.engine import run_backtest
    from .backtest.metrics import calc_metrics, calc_ic_decay
    from .report.summary import plot_single_factor
    from .display import print_single_summary, create_progress

    print_header(stock_data, 1, "单因子")

    print_step("factor", f"编译因子表达式: [dim]{args.factor}[/]")
    compiled_fn = compile_factor(args.factor)

    print_step("factor", "计算因子值...")
    factor_values = compiled_fn(stock_data, backend)

    print_step("backtest", f"运行回测 ({args.groups} 分组)...")
    result = run_backtest(factor_values, forward_returns, stock_data.dates, args.groups)

    print_step("backtest", "计算指标...")
    metrics = calc_metrics(result)
    metrics["ic_decay"] = calc_ic_decay(factor_values, forward_returns)

    print_step("report", "生成报告...")
    plot_single_factor(result, metrics, title=args.factor, output_dir=args.output)

    print_single_summary(args.factor, metrics)


def _run_batch(
    args: argparse.Namespace,
    stock_data,
    backend,
    forward_returns,
) -> None:
    from .factor.library import get_all_factors
    from .backtest.batch import batch_evaluate
    from .report.summary import plot_batch_summary
    from .display import print_batch_summary, create_progress

    if args.classic:
        factor_list = get_all_factors()
        mode = "经典因子"
    else:
        factor_list = _parse_factors_file(args.factors_file)
        mode = f"文件 ({args.factors_file})"

    print_header(stock_data, len(factor_list), mode)

    print_step("backtest", f"批量评估 {len(factor_list)} 个因子...")

    with create_progress("回测进度", len(factor_list)) as progress:
        task_id = progress.add_task("评估中", total=len(factor_list))
        results, corr_matrix = batch_evaluate(
            factor_list, stock_data, backend, forward_returns,
            stock_data.dates, args.groups,
            progress_callback=lambda: progress.advance(task_id),
        )

    print_batch_summary(results)

    print_step("report", "生成批量报告...")
    factor_names = [name for name, _ in factor_list]
    plot_batch_summary(results, corr_matrix, factor_names, output_dir=args.output)


def _run_mining(args: argparse.Namespace) -> None:
    from .backend.auto import get_backend
    from .mining import GPMiner, LLMMiner
    from .display import _console

    timer = Timer()

    stock_data = _load_data(args)
    backend = get_backend()

    if args.mine_gp:
        print_header(stock_data, 0, "GP+NSGA-II 因子挖掘")
        miner = GPMiner(stock_data, backend)
    else:
        print_header(stock_data, 0, "LLM Agent 因子挖掘")
        miner = LLMMiner(stock_data, backend)

    results = miner.mine()

    if not results:
        _console.print("[bold red]未发现有效因子[/]")
        return

    _console.print(f"\n[bold green]发现 {len(results)} 个有效因子[/]")
    _console.print("[bold]Top 10 因子（按 IR 降序）:[/]")
    for i, (expr, metrics) in enumerate(results[:10], 1):
        ir = metrics.get("ir", 0)
        ic = metrics.get("ic_mean", 0)
        sharpe = metrics.get("sharpe", 0)
        _console.print(f"  {i:>2}. IR=[cyan]{ir:.4f}[/] IC=[green]{ic:.4f}[/] Sharpe=[yellow]{sharpe:.4f}[/] {expr[:60]}")

    import json
    from pathlib import Path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / ("mining_gp.json" if args.mine_gp else "mining_llm.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"expression": expr, **{k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items()}} for expr, m in results],
            f, ensure_ascii=False, indent=2,
        )
    _console.print(f"\n[dim]结果已保存至 {output_file}[/]")

    print_done(timer.elapsed)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
