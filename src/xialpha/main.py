"""xi-alpha 因子研究流水线命令行入口。"""

import argparse
import logging
import sys

from . import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)
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
    return parser


def _load_data(args: argparse.Namespace):
    from .data.loader import load_stock_data
    from .data.cache import get_cached_or_load

    if args.no_cache:
        logger.info("Fetching data (cache disabled) ...")
        return load_stock_data(args.start, args.end)
    logger.info("Loading data (with cache) ...")
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


def _print_single_summary(factor_expr: str, metrics: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Factor: {factor_expr}")
    print("-" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:>25s}: {value:.6f}")
        else:
            print(f"  {key:>25s}: {value}")
    print("=" * 60 + "\n")


def _print_batch_summary(results: list[tuple]) -> None:
    header = f"{'Factor':<35s} {'IR':>8s} {'IC_mean':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Turnover':>8s}"
    print("\n" + "=" * len(header))
    print("Batch Factor Evaluation Summary (sorted by IR)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, _result, m in results:
        ir = m.get("ir", float("nan"))
        ic_mean = m.get("ic_mean", float("nan"))
        sharpe = m.get("sharpe", float("nan"))
        max_dd = m.get("max_drawdown", float("nan"))
        turnover = m.get("daily_turnover", float("nan"))
        print(f"{name:<35s} {ir:>8.4f} {ic_mean:>8.4f} {sharpe:>8.4f} {max_dd:>8.4f} {turnover:>8.4f}")
    print("=" * len(header) + "\n")


def run_pipeline(args: argparse.Namespace) -> None:
    if not any([args.factor, args.factors_file, args.classic]):
        print("Error: must provide at least one of --factor, --factors-file, --classic")
        print("Run with --help for usage information.")
        sys.exit(1)

    from .backend.auto import get_backend

    backend = get_backend()
    stock_data = _load_data(args)
    forward_returns = backend.shift(stock_data.returns, -1)

    if args.factor:
        _run_single(args, stock_data, backend, forward_returns)
    else:
        _run_batch(args, stock_data, backend, forward_returns)


def _run_single(
    args: argparse.Namespace,
    stock_data,
    backend,
    forward_returns,
) -> None:
    from .factor.compiler import compile_factor
    from .backtest.engine import run_backtest
    from .backtest.metrics import calc_metrics
    from .report.summary import plot_single_factor

    logger.info("Compiling factor: %s", args.factor)
    compiled_fn = compile_factor(args.factor)

    logger.info("Computing factor values ...")
    factor_values = compiled_fn(stock_data, backend)

    logger.info("Running backtest (%d groups) ...", args.groups)
    result = run_backtest(factor_values, forward_returns, stock_data.dates, args.groups)

    logger.info("Calculating metrics ...")
    metrics = calc_metrics(result)

    logger.info("Generating report ...")
    plot_single_factor(result, metrics, title=args.factor, output_dir=args.output)

    _print_single_summary(args.factor, metrics)
    logger.info("Done.")


def _run_batch(
    args: argparse.Namespace,
    stock_data,
    backend,
    forward_returns,
) -> None:
    from .factor.library import get_all_factors
    from .backtest.batch import batch_evaluate
    from .report.summary import plot_batch_summary

    if args.classic:
        factor_list = get_all_factors()
        logger.info("Using %d classic factors", len(factor_list))
    else:
        factor_list = _parse_factors_file(args.factors_file)
        logger.info("Loaded %d factors from %s", len(factor_list), args.factors_file)

    logger.info("Running batch evaluation ...")
    results, corr_matrix = batch_evaluate(
        factor_list, stock_data, backend, forward_returns,
        stock_data.dates, args.groups,
    )

    factor_names = [name for name, _ in factor_list]
    _print_batch_summary(results)

    logger.info("Generating batch report ...")
    plot_batch_summary(results, corr_matrix, factor_names, output_dir=args.output)
    logger.info("Done.")


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
