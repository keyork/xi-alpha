"""xi-alpha 报告可视化模块。"""

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，用于保存图片

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .. import config


def _ensure_dir(output_dir: str | Path | None) -> Path:
    """确保输出目录存在，若为 None 则使用默认目录。"""
    d = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_style():
    """配置 matplotlib 中文字体及绘图风格。"""
    import matplotlib.font_manager as fm
    fm._load_fontmanager(try_read_cache=False)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    # style.use 可能覆盖字体，必须在之后设置
    candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen is None:
        chosen = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def plot_single_factor(result, metrics, title="", output_dir=None):
    """绘制单因子回测报告图。

    Parameters
    ----------
    result : BacktestResult
        回测结果对象，包含 group_nav、long_short_nav、ic_series 等属性。
    metrics : dict
        指标字典，包含 ic_mean、ir、sharpe、max_drawdown、daily_turnover、t_stat、t_pvalue、ic_decay 等。
    title : str
        图表标题，默认为"因子报告"。
    output_dir : str | Path | None
        输出目录，默认为 None（使用配置目录）。
    """
    out = _ensure_dir(output_dir)
    _setup_style()

    ic_mean = metrics.get("ic_mean", 0.0)
    ir = metrics.get("ir", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    turnover = metrics.get("daily_turnover", 0.0)
    t_stat = metrics.get("t_stat", 0.0)
    t_pvalue = metrics.get("t_pvalue", 1.0)

    print(
        f"IC={ic_mean:.4f} IR={ir:.4f} Sharpe={sharpe:.4f} "
        f"MaxDD={max_dd:.4f} Turnover={turnover:.4f} "
        f"t-stat={t_stat:.4f} p={t_pvalue:.4f}"
    )

    fig, axes = plt.subplots(2, 2, figsize=config.FIGSIZE_SINGLE)
    fig.suptitle(title or "因子报告", fontsize=14, fontweight="bold")

    ax0 = axes[0, 0]
    n_groups = result.n_groups
    cmap = plt.get_cmap("RdYlGn", n_groups)
    for g in range(n_groups):
        nav = result.group_nav[g]
        ax0.plot(nav, color=cmap(g), label=f"G{g}")
    ax0.set_title("分组累积净值")
    ax0.set_xlabel("交易日")
    ax0.set_ylabel("净值")
    ax0.legend(fontsize=8, loc="best")

    ax1 = axes[0, 1]
    ls_nav = np.asarray(result.long_short_nav)
    ax1.plot(ls_nav, color="steelblue", linewidth=1.2)
    ax1.set_title(f"多空净值  (Sharpe={sharpe:.4f})")
    ax1.set_xlabel("交易日")
    ax1.set_ylabel("净值")

    ax2 = axes[1, 0]
    ic_series = np.asarray(result.ic_series)
    n_days = len(ic_series)
    if n_days > 500:
        ax2.plot(ic_series, color="steelblue", linewidth=0.6)
    else:
        colors = ["green" if not np.isnan(v) and v >= 0 else "red" for v in ic_series]
        ax2.bar(np.arange(n_days), np.nan_to_num(ic_series, nan=0.0), color=colors, width=0.8)
    ax2.axhline(ic_mean, color="black", linestyle="--", linewidth=0.8, label=f"IC 均值={ic_mean:.4f}")
    ax2.set_title("日 IC")
    ax2.set_xlabel("交易日")
    ax2.set_ylabel("IC")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 1]
    ic_decay = metrics.get("ic_decay")
    if ic_decay:
        days_k = [k for k, _ic in ic_decay]
        ics = [_ic for _k, _ic in ic_decay]
        ax3.plot(days_k, ics, marker="o", markersize=3, linewidth=1.2)
        ax3.set_title("IC 衰减")
    else:
        ax3.set_title("IC 衰减（无数据）")
    ax3.set_xlabel("持有天数")
    ax3.set_ylabel("IC")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / config.FILENAME_SINGLE, dpi=config.DPI)
    plt.show()
    plt.close(fig)
    return fig


def save_single_report(fig, name, output_dir=None):
    """将单因子报告图保存为 PNG 文件。"""
    out = _ensure_dir(output_dir)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    return path


def plot_batch_summary(results_list, corr_matrix, factor_names, output_dir=None):
    """生成批量因子汇总报告。

    Parameters
    ----------
    results_list : list of (name, BacktestResult, metrics_dict)
        按 IR 排序的回测结果列表。
    corr_matrix : 2-D ndarray
        因子间相关系数矩阵。
    factor_names : list of str
        因子名称列表。
    """
    out = _ensure_dir(output_dir)
    _setup_style()

    n_factors = len(factor_names)
    fig = plt.figure(figsize=config.FIGSIZE_BATCH)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")
    col_labels = ["因子", "IC 均值", "IR", "Sharpe", "最大回撤", "换手率"]
    cell_text = []
    cell_colors = []
    for name, _result, m in results_list:
        row = [
            name[:30],
            f"{m.get('ic_mean', 0):.4f}",
            f"{m.get('ir', 0):.4f}",
            f"{m.get('sharpe', 0):.4f}",
            f"{m.get('max_drawdown', 0):.4f}",
            f"{m.get('daily_turnover', 0):.4f}",
        ]
        cell_text.append(row)
        row_colors = []
        for val_str, metric_key in zip(
            row[1:],
            ["ic_mean", "ir", "sharpe", "max_drawdown", "daily_turnover"],
        ):
            v = float(val_str)
            if metric_key == "max_drawdown":
                green = max(0.0, min(1.0, 1.0 - abs(v) * 5))
            elif metric_key == "daily_turnover":
                green = 0.9
            else:
                green = max(0.0, min(1.0, 0.5 + v * 5))
            r = 1.0 - green
            g = green
            row_colors.append((r, g, 0.4, 0.25))
        row_colors.insert(0, (1, 1, 1, 0))
        cell_colors.append(row_colors)

    tbl = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax_table.set_title("因子指标汇总（按 IR 排序）", fontsize=12, fontweight="bold", pad=12)

    ax_corr = fig.add_subplot(gs[1, 0])
    corr = np.asarray(corr_matrix)
    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax_corr.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    short_names = [n[:12] for n in factor_names]
    ax_corr.set_xticks(range(n_factors))
    ax_corr.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax_corr.set_yticks(range(n_factors))
    ax_corr.set_yticklabels(short_names, fontsize=7)
    ax_corr.set_title("因子相关性", fontsize=11)

    ax_nav = fig.add_subplot(gs[1, 1])
    top_n = min(config.TOP_N_DISPLAY, len(results_list))
    cmap_nav = plt.get_cmap("tab10", top_n)
    for idx in range(top_n):
        name, result, m = results_list[idx]
        ls_nav = np.asarray(result.long_short_nav)
        ax_nav.plot(ls_nav, color=cmap_nav(idx),
                    label=f"{name[:20]} (SR={m.get('sharpe', 0):.2f})", linewidth=1.0)
    ax_nav.set_title(f"前 {top_n} 个因子 — 多空净值")
    ax_nav.set_xlabel("交易日")
    ax_nav.set_ylabel("净值")
    ax_nav.legend(fontsize=7, loc="best")

    fig.suptitle("批量因子报告", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / config.FILENAME_BATCH, dpi=config.DPI)
    plt.show()
    plt.close(fig)
    return fig
