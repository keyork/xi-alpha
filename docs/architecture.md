# xi-alpha 系统架构文档

## 1. 概述

xi-alpha 是一个面向 A 股市场的向量化因子研究引擎。它提供从数据获取、因子计算、回测评估到可视化报告的完整流水线，目标是让因子研究员用最少的代码完成从因子表达式到回测结果的全流程验证。

### 设计原则

| 原则 | 说明 |
|------|------|
| **模块解耦** | 数据层、计算后端、因子引擎、回测引擎、报告生成、终端显示各自独立，层间仅通过明确的接口交互 |
| **配置驱动** | 因子定义、回测参数、报告配色、缓存路径均通过 TOML 配置文件控制，代码中零硬编码业务参数 |
| **NaN-safe** | 所有数值计算路径默认处理缺失值，backfill、滚动窗口、排名等操作对 NaN 有明确行为 |
| **安全沙箱** | 因子表达式通过 AST 编译执行，白名单机制限定可用算子和函数，杜绝任意代码执行风险 |

### 构建与包管理

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖，构建后端从 setuptools 切换为 **hatchling**。核心配置在 `pyproject.toml` 中：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

安装方式：

```bash
uv sync              # 安装依赖
uv run xi-alpha      # 通过 CLI 入口运行
uv run python main.py  # 通过根目录入口运行
```

### 根目录入口

`main.py` 位于项目根目录，作为快捷入口，一行代码调用 `xialpha.main:main`：

```python
from xialpha.main import main

if __name__ == "__main__":
    main()
```

开发者可以直接 `python main.py` 启动流水线，也可以通过 `uv run xi-alpha` 调用已注册的 CLI 命令。

### 幸存者偏差警告

配置文件 `xi-alpha.toml` 头部包含醒目的幸存者偏差警告：默认股票池由人工挑选的约 50 只大盘蓝筹组成，这些股票已经过市场长期验证，不包含已退市、ST 或长期表现不佳的标的。在此池子上得到的因子 IC / IR 可能显著高于全市场水平。生产环境应替换为指数成分股（如中证 500/1000）。

---

## 2. 系统架构图

```
CLI / main.py (根目录入口)
    │
    ├── 回测模式 ──────────────────────────────────────────────────────────────────────────────────┐
    │                                                                                             │
    ▼                                                                                             │
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   Data   │ →  │ Backend  │ →  │  Factor  │ →  │ Backtest │ →  │  Report  │ →  │ Display  │      │
│  Layer   │    │  Layer   │    │  Layer   │    │  Layer   │    │  Layer   │    │  Layer   |      │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
  loader.py       BackendBase      compiler         engine          summary          display.py    │
  cache.py        numpy_backend    operators.py     metrics.py      matplotlib                     │
                   auto.py          library.py       batch.py        中文字体自适应                 │
                                                                                                   │
    └── 挖掘模式 ─────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                      Mining Layer (挖掘层)                        │
    │                                                                   │
    │  ┌──────────────────────────┐  ┌──────────────────────────────┐  │
    │  │ GP+NSGA-II 挖掘器         │  │ LLM Agent 挖掘器              │  │
    │  │                          │  │                              │  │
    │  │ tree.py                  │  │ prompts.py                   │  │
    │  │   generate_tree()        │  │   SYSTEM_PROMPT              │  │
    │  │   crossover()            │  │   build_initial_prompt()     │  │
    │  │   mutate()               │  │   build_iteration_prompt()   │  │
    │  │                          │  │   parse_llm_response()       │  │
    │  │ nsga.py                  │  │                              │  │
    │  │   fast_non_dominated_    │  │ llm_miner.py                 │  │
    │  │     sort()               │  │   LLMMiner.mine()            │  │
    │  │   nsga2_select()         │  │                              │  │
    │  │                          │  │                              │  │
    │  │ gp_miner.py              │  │                              │  │
    │  │   GPMiner.mine()         │  │                              │  │
    │  └──────────────────────────┘  └──────────────────────────────┘  │
    │                                                                   │
    │  共享基类: base.py → BaseMiner._evaluate_factor()                  │
    │  内部调用: Factor Layer (compile) + Backtest Layer (run_backtest)  │
    └───────────────────────────────────────────────────────────────────┘

回测模式数据流向: 原始行情 → 标准化 ndarray → 因子值序列 → 分组回测结果 → 图表/表格 → Rich 终端美化输出
挖掘模式数据流向: 原始行情 → 标准化 ndarray → [GP 进化 / LLM 迭代] → 因子表达式 → 编译 → 回测 → 筛选 → 按 IR 降序输出
```

### 各层职责

| 层 | 输入 | 输出 | 职责 |
|----|------|------|------|
| Data Layer | 股票代码列表 + 日期范围 | `StockData` (含 OHLCV ndarray) | 从 akshare 拉取行情数据，缓存到本地 `.cache/` |
| Backend Layer | ndarray 运算请求 | ndarray 计算结果 | 抽象数值计算接口，当前提供 NumPy 实现 |
| Factor Layer | 因子表达式字符串 | `(T, N)` 因子值矩阵 | 编译表达式、调度算子、输出因子值 |
| Backtest Layer | 因子值矩阵 + 收益率矩阵 | `BacktestResult` (13 项指标 + IC Decay) | 分组回测、多空比值法 NAV、统计检验 |
| Report Layer | `BacktestResult` + 指标字典 | PNG 图片 | matplotlib 可视化报告生成 |
| Display Layer | `BacktestResult` + 指标字典 | 终端美化输出 | Rich 面板头部、指标表格、进度条、计时器 |
| Mining Layer | `StockData` + `BackendBase` 实例 | `list[(expression, metrics)]` 按 IR 降序 | GP+NSGA-II 自动进化因子表达式，或 LLM Agent 迭代式因子发现 |

### 层间数据流

```
Data → Backend
    StockData.prices (DataFrame) → 通过 Backend 方法计算收益率等衍生指标

Backend → Factor
    Backend 实例注入 Factor Compiler，算子调用 Backend 方法完成实际计算

Factor → Backtest
    factor_values: ndarray(T, N) + returns: ndarray(T, N) → 输入回测引擎

Backtest → Report
    BacktestResult + metrics dict → matplotlib 绑图 → PNG 文件

Backtest → Display
    BacktestResult + metrics dict → Rich Table / Panel → 终端美化输出
```

---

## 3. 模块详解

### 3.1 数据层 (`data/`)

#### `loader.py`

数据加载的核心模块，负责从 akshare 获取 A 股行情数据并标准化为内部格式。

**StockData dataclass**

```python
@dataclass
class StockData:
    open: np.ndarray      # 开盘价 (T, N)
    high: np.ndarray      # 最高价 (T, N)
    low: np.ndarray       # 最低价 (T, N)
    close: np.ndarray     # 收盘价 (T, N)
    volume: np.ndarray    # 成交量 (T, N)
    amount: np.ndarray    # 成交额 (T, N)
    returns: np.ndarray   # 日收益率 (T, N)，基于前复权收盘价计算
    dates: np.ndarray     # 交易日期 (T,)
    symbols: np.ndarray   # 股票代码 (N,)
```

**load_stock_data()**

主入口函数。执行流程：

1. 检查缓存，命中则直接返回
2. 逐一拉取每只股票的行情数据
3. 双 API 降级策略：优先调用 `ak.stock_zh_a_daily()`（东方财富），失败时降级到 `ak.stock_zh_a_hist()`（新浪）
4. 指数退避重试：初始间隔由 `base_delay` 配置决定，每次失败翻倍，最多 `max_retries` 次
5. 交易日对齐：以所有股票交集的交易日为准，缺失交易日填 NaN
6. 写入缓存后返回

#### `cache.py`

基于 pickle 的文件缓存实现。

- 缓存路径：`{CACHE_DIR}/{start}_{end}_{md5_hash}.pkl`
- `CACHE_DIR` 解析到项目根目录下的 `.cache/` 目录（通过 `config.PROJECT_ROOT / ".cache"` 确定）
- 缓存 key：对股票池排序后做 MD5 哈希取前 12 位，拼接日期范围
- 命中缓存时直接 `pickle.load` 返回，跳过全部网络请求

---

### 3.2 后端抽象层 (`backend/`)

后端层将数值计算抽象为统一接口，使得因子表达式与底层计算引擎解耦。

#### `base.py` — BackendBase ABC

定义 28 个抽象方法，覆盖因子计算所需的所有数值操作：

| 方法类别 | 方法 | 说明 |
|----------|------|------|
| 滚动统计 | `rolling_mean`, `rolling_std`, `rolling_max`, `rolling_min`, `rolling_sum`, `rolling_var` | 滚动窗口统计量 |
| 滚动排名 | `rolling_rank` | 滚动窗口内百分位排名 |
| 时序操作 | `delay`, `delta` | 平移和差分 |
| 截面操作 | `cross_sectional_rank`, `cross_sectional_zscore` | 截面标准化 |
| 逐元素操作 | `rank`, `abs`, `log`, `sign`, `power`, `max_`, `min_` | 逐元素变换 |
| 逻辑操作 | `if_else` | 条件选择 |
| 其他 | `fillna`, `corr`, `cov` | 缺失值处理和统计 |

```python
class BackendBase(ABC):
    @abstractmethod
    def rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray: ...

    @abstractmethod
    def cross_sectional_rank(self, data: np.ndarray) -> np.ndarray: ...
    # ... 其余 26 个抽象方法
```

#### `numpy_backend.py` — NumPy 实现

基于 NumPy/pandas/scipy 的参考实现。

- 滚动操作：通过 pandas `Series.rolling()` 桥接实现，利用 pandas 对 NaN 边界的成熟处理
- 截面排名：使用 `scipy.stats.rankdata()` 计算百分位排名
- NaN 处理：所有方法保证输出形状与输入一致，无效位置填 NaN

#### `auto.py` — 工厂函数

当前直接返回 NumPy 后端实例，后续可扩展 JAX/PyTorch 后端：

```python
def get_backend() -> BackendBase:
    return NumPyBackend()
```

---

### 3.3 因子层 (`factor/`)

因子层是系统的核心，负责将人类可读的因子表达式编译为可执行的数值计算。

#### `operators.py` — 算子注册表

采用装饰器注册模式管理所有可用算子。

```python
OPERATOR_REGISTRY: dict[str, Callable] = {}

def _register(name: str):
    def decorator(fn):
        OPERATOR_REGISTRY[name] = fn
        return fn
    return decorator

@_register("rolling_mean")
def _rolling_mean(*args, stock_data=None, backend=None):
    x, window = args
    return backend.rolling_mean(x, int(window))
```

**15 个已注册算子**：

| 类别 | 算子 | 说明 |
|------|------|------|
| 滚动统计 | `rolling_mean(x, d)` | 滚动均值 |
| | `rolling_std(x, d)` | 滚动标准差 |
| | `rolling_max(x, d)` | 滚动最大值 |
| | `rolling_min(x, d)` | 滚动最小值 |
| | `rolling_sum(x, d)` | 滚动求和 |
| | `rolling_corr(x, y, d)` | 滚动相关系数 |
| 时序平移 | `shift(x, d)` | 平移 d 期（正数向后，负数向前） |
| | `pct_change(x, d)` | d 期变化率 |
| | `diff(x, d)` | d 期差值 |
| 截面操作 | `rank(x)` | 截面百分位排名 |
| | `zscore(x)` | 截面 Z-score 标准化 |
| | `demean(x)` | 截面去均值 |
| 逐元素 | `log(x)` | 自然对数 |
| | `abs(x)` | 绝对值 |
| | `sign(x)` | 符号函数 |

#### `compiler.py` — AST 编译器

因子表达式的编译执行引擎。核心流程：

```
表达式字符串
    │
    ▼  ast.parse()
AST 树
    │
    ▼  _validate_ast()   ← 安全白名单校验
校验后的 AST
    │
    ▼  compile()          ← 生成字节码
编译函数
    │
    ▼  eval()             ← 传入数据执行
ndarray 结果
```

**编译缓存**：相同表达式只编译一次，后续直接从 `_compile_cache` 字典取编译结果。

**关键函数**：

```python
def compile_factor(expr: str) -> Callable:
    """将表达式编译为可调用函数 (stock_data, backend) -> ndarray"""
    if expr in _compile_cache:
        return _compile_cache[expr]
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree)
    code = compile(tree, "<factor>", "eval")
    def _compiled(stock_data, backend):
        ns = {}
        for name, op_func in OPERATOR_REGISTRY.items():
            ns[name] = lambda *a, _fn=op_func, _be=backend: _fn(*a, backend=_be)
        for field in DATA_FIELDS:
            ns[field] = getattr(stock_data, field)
        return eval(code, {"__builtins__": {}}, ns)
    _compile_cache[expr] = _compiled
    return _compiled
```

**代码风格**：编译器使用 `raise TypeError` / `raise ValueError` 替代 `assert` 进行类型校验和错误报告，确保生产环境下错误信息完整可靠。

#### `library.py` — 经典因子库

从 TOML 配置文件的 `[factors.classic]` 节加载预定义因子集合，提供开箱即用的经典因子：

```python
CLASSIC_FACTORS: dict[str, str] = {
    "short_term_reversal": "-1 * rolling_sum(returns, 5)",
    "volatility": "-1 * rolling_std(returns, 20)",
    "ma_deviation": "close / rolling_mean(close, 20) - 1",
    ...
}

def get_all_factors() -> list[tuple[str, str]]:
    return list(config.CLASSIC_FACTORS.items())
```

配置文件中的因子定义：

```toml
[factors.classic]
short_term_reversal = "-1 * rolling_sum(returns, 5)"
medium_term_momentum = "rolling_sum(returns, 20)"
volatility = "-1 * rolling_std(returns, 20)"
turnover = "-1 * rolling_mean(volume, 10)"
ma_deviation = "close / rolling_mean(close, 20) - 1"
price_volume_divergence = "-1 * rolling_corr(close, volume, 10)"
```

---

### 3.4 回测层 (`backtest/`)

#### `engine.py` — 回测引擎

分组回测的核心实现。每个横截面上按因子值排序，等分为若干组。

**分组规则**：

- 因子值从大到小排序
- `G0 (高)` 对应因子值最高的一组（多头组）
- `G4 (低)` 对应因子值最低的一组（空头组），当 `n_groups=5` 时
- 多空组合 NAV 采用**比值法**，详见下方

**分组标签**：报告图中分组标签采用 `G0 (高)` / `G1` / `G2` / `G3` / `G4 (低)` 格式，首尾组标注方向含义。

```python
def run_backtest(
    factor_values: np.ndarray,   # (T, N) 因子值
    forward_returns: np.ndarray,  # (T, N) 前向收益率
    dates: np.ndarray,            # (T,) 日期序列
    n_groups: int = config.N_GROUPS,
) -> BacktestResult:
    ...
```

**百分位排名**：因子值先做截面百分位排名，再按排名分组，避免异常值影响分组边界。

**最小截面股票数**：通过 `min_stocks_for_group` 配置项控制，当某期有效股票数低于此值时跳过该期分组。IC 计算也有独立的最小阈值 `min_stocks_for_ic`。

**多空净值比值法**：多空组合净值采用比值法而非累加法：

```python
long_short_nav = group_nav[0] / group_nav[n_groups - 1]
```

这一设计决策的原因：比值法的分子分母都是累积净值，量纲一致，经济含义清晰（多头净值相对空头净值的倍数）。相比之下，累加法（`group_returns[0] - group_returns[N-1]` 再累乘）在两边净值差异悬殊时容易失真。

**多空方向说明**：IC 和多空组合的方向可能不一致。IC 是因子值与下期收益率的 Spearman 相关系数，反映的是单调关系方向；多空组合固定为 G0（因子值最高组）减去 G4（因子值最低组）的收益率。因此，当 IC 为负时，说明因子值越低收益越高，但多空组合仍计算高分组减低分组，两者方向可能相反。这是设计上的有意选择：IC 衡量预测能力，多空衡量极端分组的差异。

#### `metrics.py` — 指标计算

提供 13 项回测评估指标及 IC Decay 计算。

**13 项核心指标**：

| 指标 | 说明 |
|------|------|
| IC 均值 (ic_mean) | 因子值与下期收益率的 Spearman 截面相关系数均值 |
| IC 标准差 (ic_std) | IC 序列标准差 |
| 信息比率 IR (ir) | IC 均值 / IC 标准差 |
| IC 正值占比 (ic_positive_ratio) | IC 为正的期数占比 |
| 年化收益 (annual_return) | 多空组合年化收益率 (x 252) |
| 年化波动率 (annual_vol) | 多空组合年化波动率 (x sqrt(252)) |
| Sharpe 比率 (sharpe) | 年化收益 / 年化波动率 |
| 最大回撤 (max_drawdown) | 多空组合最大回撤 |
| 最大回撤持续天数 (max_dd_duration) | 最大回撤持续交易日数 |
| 日均换手率 (daily_turnover) | 因子分组日均换手率 |
| t 统计量 (t_stat) | 多空收益均值的 t 检验统计量 |
| t 检验 p 值 (t_pvalue) | t 检验 p 值 |
| 分组单调性 (group_monotonicity) | 分组收益是否单调递减 |

**IC Decay 计算**（`calc_ic_decay()`）：

计算因子在不同持有天数下的 IC 衰减曲线。对于第 k 天（k 从 1 到 `ic_decay_days`），将前向收益率向后平移 k 期，然后计算截面 Spearman IC 的均值。结果为 `[(day_k, ic_k), ...]` 列表。

```python
def calc_ic_decay(
    factor_values: np.ndarray,
    forward_returns: np.ndarray,
    max_days: int | None = None,    # 默认取 config.IC_DECAY_DAYS
) -> list[tuple[int, float]]:
    ...
```

IC Decay 反映因子信号的有效持续时间，帮助研究者判断因子的最优持有周期。

#### `batch.py` — 批量回测

支持同时对多个因子进行回测，支持并行执行和进度回调：

```python
def batch_evaluate(
    factor_list: list[tuple[str, str]],   # [(因子名, 表达式), ...]
    stock_data: StockData,
    backend: BackendBase,
    forward_returns: np.ndarray,          # (T, N) 前向收益率
    dates: np.ndarray,                    # (T,) 日期
    n_groups: int = config.N_GROUPS,
    n_jobs: int = config.N_JOBS,
    progress_callback: Callable | None = None,
) -> tuple[list[tuple[str, BacktestResult, dict]], np.ndarray]:
    """批量评估，返回 (按 IR 排序的结果列表, 因子间相关系数矩阵)"""
    ...
```

**并行策略**：使用 `ProcessPoolExecutor` 多进程并行，每个因子的回测在独立进程中执行。当 `n_jobs <= 1` 或因子数不超过 1 时退化为串行执行。

**进度回调**：支持 `progress_callback` 参数，每完成一个因子后调用，用于 Rich 进度条更新。

**额外输出**：
- 因子间相关性矩阵：计算所有因子两两之间的截面 IC 相关性
- IR 排序：按 IR 值对所有因子降序排列，快速筛选有效因子
- 每个因子均计算 IC Decay 数据

---

### 3.5 报告层 (`report/`)

#### `summary.py` — 报告生成器

将回测结果转化为可视化报告。所有配色参数通过配置项控制，零硬编码。

**单因子报告**：2x2 子图布局

```
┌─────────────────────┬─────────────────────┐
│                     │                     │
│   分组累积净值曲线    │    多空净值 NAV 曲线   │
│   (colormap_groups   │    (line_color      │
│    色带 + 标签)       │     steelblue 线)   │
│   G0(高)~G4(低)      │                     │
│                     │                     │
├─────────────────────┼─────────────────────┤
│                     │                     │
│   日 IC 时序图       │   IC Decay 图        │
│   (柱状图/线图 +     │   (持有天数 vs IC)   │
│    均值虚线)         │                     │
│                     │                     │
└─────────────────────┴─────────────────────┘
```

**中文字体自适应**：自动检测系统中可用的中文字体（WenQuanYi Micro Hei、Noto Sans CJK SC、SimHei、Microsoft YaHei 等），无需手动配置。

**批量报告**：对多个因子生成汇总表格，包含因子间 IC 相关性热力图（`colormap_corr` 配色）和前 N 个因子的 NAV 曲线叠加图（`colormap_nav` 配色）。

---

### 3.6 显示层 (`display.py`)

基于 Rich 库的终端美化模块，替代原始 logging 输出，让运行过程一目了然。

**模块级变量 `_METRIC_LABELS`**：将指标名映射到中文标签的字典，用于 Rich Table 显示。

```python
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
```

**模块级变量 `_STYLES`**：各阶段对应的颜色映射。

```python
_STYLES = {
    "data": "cyan",
    "factor": "magenta",
    "backtest": "yellow",
    "report": "green",
    "done": "bold bright_green",
}
```

**`setup_logging()`**：用 `RichHandler` 替代默认 logging handler，日志输出自带时间戳和着色，同时抑制 matplotlib 的冗余日志。

**`print_header()`**：使用 Rich `Panel` 显示运行头部信息，包含股票数、交易日、因子数、运行时间。

**`create_progress()`**：返回 Rich `Progress` 实例，配置了旋转动画、进度条、百分比、计数和计时器。

**`print_single_summary()`**：使用 Rich `Table` 显示单因子指标摘要。正值标绿、负值标红，p 值小于 0.05 加星号标注显著性。

**`print_batch_summary()`**：使用 Rich `Table` 显示批量因子汇总表，包含 IR、IC 均值、Sharpe、最大回撤、换手率列。

**`Timer` 类**：基于 `time.perf_counter()` 的计时器，记录从创建到当前的耗时。

**`print_done()`**：使用 Rich `Panel` 显示完成信息和总耗时。

**数据流**：`BacktestResult` + `metrics dict` → `Display Layer` → 终端美化输出（与 Report Layer 并行，Report 负责图片文件，Display 负责终端交互）。

---

### 3.7 挖掘层 (`mining/`)

挖掘层提供两种自动因子发现策略——基于遗传编程的 GP+NSGA-II 挖掘器，以及基于大语言模型的 LLM Agent 挖掘器。两者共享 `BaseMiner` 基类提供的评估管线，内部调用 Factor Layer（编译）和 Backtest Layer（回测 + 指标计算）。

#### 3.7.1 `base.py` — BaseMiner ABC

所有挖掘器的抽象基类，封装了「编译表达式 → 执行 → 回测 → 指标」的完整评估管线。

**BaseMiner.\_\_init\_\_(stock_data, backend)**

- `stock_data`：`StockData` 实例，包含行情数据
- `backend`：`BackendBase` 实例，用于数值计算
- 预计算前向收益率：`self._forward_returns = backend.shift(stock_data.returns, -1)`，用于回测
- 初始化去重集合：`self._seen_expressions: set[str] = set()`

**\_evaluate\_factor(expression)** — 单因子评估

核心评估方法，执行以下流程：

1. **去重检查**：表达式已在 `_seen_expressions` 中则直接返回 `None`
2. **编译**：调用 `compile_factor(expression)` 将表达式编译为可执行函数
3. **执行**：调用编译后的函数获取因子值 ndarray
4. **全 NaN 检查**：`np.all(np.isnan(factor_values))` 为 True 则返回 `None`
5. **回测**：调用 `run_backtest(factor_values, self._forward_returns, stock_data.dates)`
6. **指标计算**：调用 `calc_metrics(result)` 计算指标
7. 返回指标字典；任何步骤失败均返回 `None`

```python
def _evaluate_factor(self, expression: str) -> dict | None:
    if expression in self._seen_expressions:
        return None
    self._seen_expressions.add(expression)
    try:
        compiled = compile_factor(expression)
        factor_values = compiled(self.stock_data, self.backend)
        if np.all(np.isnan(factor_values)):
            return None
        result = run_backtest(factor_values, self._forward_returns, self.stock_data.dates)
        return calc_metrics(result)
    except Exception:
        return None
```

**\_evaluate\_batch(expressions)** — 批量评估

对表达式列表逐一调用 `_evaluate_factor()`，返回 `[(expression, metrics | None), ...]` 列表。

**mine()** — 抽象方法

子类必须实现的核心挖掘方法，返回 `list[tuple[str, dict]]`——所有成功评估的因子按 IR 降序排列。

#### 3.7.2 `tree.py` — 表达式树操作

为 GP 挖掘器提供表达式树的表示、随机生成、交叉、变异和序列化功能。

**FactorNode dataclass**

```python
@dataclass
class FactorNode:
    node_type: str          # 'field', 'constant', 'unary', 'window', 'corr', 'arithmetic'
    value: str              # 字段名 / 算子名 / 运算符 / 常量字符串
    children: list[FactorNode]
    window: int | None = None  # 仅 window 和 corr 类型使用
```

**算子分类表**

| 类别 | 元素 | 子节点数 |
|------|------|----------|
| 终端-字段 | `open`, `high`, `low`, `close`, `volume`, `amount`, `returns` | 0 |
| 终端-常量 | `-1`, `-0.5`, `0.5`, `1`, `2`, `3`, `5`, `10`, `20`, `60` | 0 |
| 一元算子 | `rank`, `zscore`, `demean`, `log`, `abs`, `sign` | 1 |
| 窗口算子 | `rolling_mean`, `rolling_std`, `rolling_sum`, `rolling_max`, `rolling_min`, `shift`, `pct_change`, `diff` | 1 + window |
| 相关算子 | `rolling_corr` | 2 + window |
| 算术运算 | `+`, `-`, `*`, `/` | 2 |

窗口值候选：`[5, 10, 20, 60]`。

**to\_expression(node)** — 序列化

将表达式树序列化为与 AST 编译器兼容的字符串：

```python
# field → "close"
# constant → "20"
# unary → "rank(close)"
# window → "rolling_mean(close, 20)"
# corr → "rolling_corr(close, volume, 10)"
# arithmetic → "(close) * (volume)"
```

**generate\_tree(max\_depth, rng, method)** — 随机生成

- `method="grow"`：在非叶节点以约 40% 概率选择算子，60% 概率选择终端，生成形态多样的树
- `method="full"`：深度未达上限前只选算子，保证树的结构完整
- 终端生成概率分布：85% 数据字段、15% 常量
- 算子类别选择权重：一元 × 3、窗口 × 3、算术 × 2、corr × 1

**crossover(parent1, parent2, max\_depth, rng)** — 交叉

随机选择两棵树的各一个子树进行交换，返回两个子代（深拷贝，不修改原始树）。**深度保护**：若任一子代深度超过 `max_depth`，则放弃交叉，返回原始树的拷贝。

**mutate(tree, max\_depth, rng)** — 变异

随机选择一个子树，替换为新随机生成的子树。可用深度为 `max(1, max_depth - len(path))`。**深度保护**：变异后深度超过 `max_depth` 则放弃变异，返回原始树的拷贝。

#### 3.7.3 `nsga.py` — NSGA-II 多目标选择

为 GP 挖掘器提供基于 Pareto 支配的多目标选择算法。

**dominates(a, b)**

判断个体 `a` 是否支配个体 `b`。采用最大化语义：`a` 支配 `b` 当且仅当所有目标上 `a >= b`，且至少一个目标上 `a > b`。

**fast\_non\_dominated\_sort(fitnesses)**

对种群进行非支配排序，返回 Pareto 前沿列表。每个前沿是一个索引列表，第一个前沿为 Pareto 最优前沿（不被任何其他个体支配）。

算法流程：
1. 计算每个个体的支配计数和被支配集合
2. 支配计数为 0 的个体组成第一前沿
3. 逐层剥离：对当前前沿中每个个体，将其被支配集合中的个体支配计数减 1，计数归零者进入下一前沿

**crowding\_distance(front\_fitnesses)**

计算单个前沿内个体的拥挤度距离。**边界个体**（每个目标上的最小值和最大值个体）距离设为 `inf`，确保边界个体始终被保留。内部个体的距离为各目标方向上相邻个体间距的归一化之和。

**nsga2\_select(population, fitnesses, n\_select)**

NSGA-II 选择主函数：

1. 对种群执行非支配排序，得到前沿列表
2. 按前沿顺序逐个填入选中集合
3. 当最后一前沿无法全部放入时，按拥挤度距离降序截断

**tournament\_select(population, fitnesses, tournament\_size, rng)**

锦标赛选择：随机抽取 `tournament_size` 个候选个体，选择 Pareto 排名最低（最优）的个体。同排名时以拥挤度距离决胜（距离大者优先）。

#### 3.7.4 `gp_miner.py` — GP+NSGA-II 挖掘器

结合遗传编程和 NSGA-II 多目标选择的因子挖掘器。

**进化主循环**

```
初始化种群 (grow 方法随机生成 population_size 棵树)
    │
    ▼
评估初始种群 (编译 → 回测 → 指标)
    │
    ▼
┌─── 进化循环 (n_generations 代) ──────────────────────────────────┐
│                                                                   │
│   1. 计算适应度: config.GP_OBJECTIVES → metrics 字典取值            │
│   2. NSGA-II 选择: 从当前种群选出 (pop_size - elites) 个个体       │
│   3. 精英保留: 按 IR 降序取前 elites 个有效个体                      │
│   4. 交叉 + 变异: 对选中个体生成子代                                 │
│      - crossover_prob: 交叉概率                                     │
│      - mutation_prob: 变异概率                                      │
│      - 其余: 直接复制                                               │
│   5. 深度保护: 超过 max_depth 则重新生成                            │
│   6. 评估新个体: 对新种群中未见过的表达式进行评估                     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
汇总: 所有历史成功因子按 IR 降序返回
```

**适应度映射**

通过 `config.GP_OBJECTIVES` 配置列表将目标名称映射到 metrics 字典取值。以 `-` 前缀的目标名表示取反（用于需要最小化的指标，如最大回撤）。评估失败的个体适应度为全零元组，在 NSGA-II 排序中自然被淘汰。

```python
def _extract_objective(metrics: dict | None, obj_name: str) -> float:
    if metrics is None:
        return 0.0
    if obj_name.startswith("-"):
        return -metrics.get(obj_name[1:], 0.0)
    return metrics.get(obj_name, 0.0)
```

**精英保留**

每代保留 `elites` 个 IR 最高的有效个体直接进入下一代。若无有效个体，则随机生成替代。

**日志输出**

每代输出：代数、种群大小、有效因子数、最佳 IR 值。

#### 3.7.5 `llm_miner.py` — LLM Agent 挖掘器

基于大语言模型的迭代式因子发现挖掘器。通过 httpx 调用 OpenAI 兼容 API（POST /v1/chat/completions）。

**状态管理**

| 属性 | 容量 | 说明 |
|------|------|------|
| `_best_factors` | top 20 | 历史最佳因子列表，按 IR 降序维护 |
| `_recent_failures` | 最近 10 | 最近评估失败的表达式列表 |
| `_all_results` | 无限制 | 完整尝试记录 |

**迭代流程**

```
第 1 轮: build_initial_prompt(n_factors) → LLM 生成
    │
    ▼
┌─── 迭代循环 (max_iterations 轮) ─────────────────────────────────┐
│                                                                   │
│   1. 构建 prompt:                                                 │
│      - 第 1 轮: build_initial_prompt(n_factors)                    │
│      - 后续轮: build_iteration_prompt(                             │
│          n_factors, best[:5], failures[-5:], iteration, max_iter)  │
│                                                                   │
│   2. 调用 LLM: httpx.post(api_url, payload, timeout=60s)          │
│                                                                   │
│   3. 解析响应: parse_llm_response(content) → 表达式列表             │
│                                                                   │
│   4. 评估: _evaluate_batch(expressions) → results                  │
│                                                                   │
│   5. 更新状态:                                                     │
│      - 成功因子加入 _best_factors (保留 top 20)                    │
│      - 失败表达式加入 _recent_failures (保留最近 10)               │
│      - 记录本轮最佳 IR                                             │
│                                                                   │
│   6. 终止条件检查:                                                 │
│      - 连续 3 轮无改善 (stagnation >= 3)                           │
│      - 达到 max_iterations                                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    │
    ▼
返回: _best_factors (top 20 按 IR 降序)
```

**终止条件**

- 连续 `_STAGNATION_LIMIT = 3` 轮无 IR 改善
- 达到 `config.LLM_MAX_ITERATIONS` 上限
- LLM 返回空响应或无法解析时计入停滞计数

**\_call\_llm(messages)**

通过 httpx 同步调用 LLM API，支持 Bearer Token 认证。超时 60 秒，失败时返回空字符串。

#### 3.7.6 `prompts.py` — LLM 提示词模板

为 LLM Agent 挖掘器提供提示词构建和响应解析功能。

**SYSTEM\_PROMPT**

系统提示词，定义 LLM 的角色和约束：

- **角色定义**：专业的量化因子研究员
- **可用数据字段**：open, high, low, close, volume, amount, returns
- **可用算子**：截面（rank, zscore, demean）、滚动（rolling\_mean/std/sum/max/min）、相关（rolling\_corr）、时序（shift, pct\_change, diff）、逐元素（log, abs, sign）、算术（+, -, *, /）
- **输出格式规则**：JSON 格式 `{"factors": ["表达式1", "表达式2", ...]}`

**build\_initial\_prompt(n)**

构建第一轮迭代的提示词，要求生成 `n` 个因子表达式。

**build\_iteration\_prompt(n\_factors, best\_factors, recent\_failures, iteration, max\_iterations)**

构建后续迭代的提示词，包含历史反馈信息：

- **最佳因子反馈**：展示前 5 个最佳因子的 IR、IC、Sharpe 及表达式
- **失败案例反馈**：展示最近 5 个评估失败的表达式，引导 LLM 避免类似错误
- **迭代进度**：当前轮次 / 总轮次

**parse\_llm\_response(content)** — 响应解析

两层解析策略：

1. **JSON 解析**：尝试 `json.loads(content)`，提取 `factors` 字段中的字符串列表
2. **正则回退**：JSON 解析失败时，使用正则表达式匹配引号内的表达式模式，匹配算子名、数据字段名、数字、运算符、括号等组成的合法因子表达式

---

## 4. 数据流

从命令行输入到最终报告的完整数据流：

```
CLI 输入
  ├── 股票代码列表: list[str]
  ├── 日期范围: (start_date, end_date)
  ├── 因子表达式列表: list[str]
  └── 回测参数: dict (n_groups, ...)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: 数据加载                                                    │
│  输入: codes, start_date, end_date                                   │
│  输出: StockData                                                     │
│  处理: akshare API → DataFrame → 标准化 ndarray (T, N)               │
│  缓存: 项目根目录 .cache/ 下的 pickle 文件                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: 后端初始化                                                   │
│  输入: backend_name (str)                                            │
│  输出: BackendBase 实例                                               │
│  处理: 工厂函数创建后端对象                                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: 因子计算                                                    │
│  输入: expressions (list[str]), StockData, BackendBase               │
│  输出: dict[str, ndarray(T, N)] — 每个因子一个值矩阵                    │
│  处理: AST 编译表达式 → 注入数据字段和算子 → eval 执行                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: 回测执行                                                    │
│  输入: factor_values ndarray(T, N), forward_returns ndarray(T, N)    │
│  输出: BacktestResult (13 项指标 + IC Decay + 分组收益序列)            │
│  处理: 截面排名 → 分组 → 各组净值 → 比值法多空 NAV → 统计指标            │
│  阈值: min_stocks_for_group, min_stocks_for_ic                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: 报告生成 + 终端显示                                           │
│  输入: BacktestResult + metrics dict                                 │
│  输出: PNG 图片文件 + Rich 终端美化表格                                 │
│  处理: matplotlib 绑图 → 项目根目录 output/ 目录                       │
│        Rich Table → 终端指标摘要                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 挖掘模式数据流（Step 6）

挖掘模式是一种独立的顶层运行模式，内部调用 Data → Backend → Factor → Backtest 层完成因子评估，但不经过 Report 和 Display 层。

#### GP+NSGA-II 挖掘模式

```
CLI 输入: --mine-gp
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6a: GP 初始化                                                  │
│  输入: StockData, BackendBase                                        │
│  处理: BaseMiner 预计算前向收益率 → 随机生成 population_size 棵表达式树 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6b: GP 进化循环 (n_generations 代)                             │
│                                                                     │
│  对每棵树: to_expression(tree) → 字符串                               │
│       → compile_factor(expr) → 编译函数                               │
│       → eval(stock_data, backend) → 因子值 ndarray                    │
│       → run_backtest(factor_values, forward_returns, dates)          │
│       → calc_metrics(result) → 指标字典                               │
│                                                                     │
│  适应度计算: objectives → metrics 取值 → (float, ...) 元组            │
│  NSGA-II 选择 → 精英保留 → 交叉 + 变异 → 新种群                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6c: 结果汇总                                                   │
│  输出: list[(expression, metrics)] — 所有历史成功因子按 IR 降序        │
└─────────────────────────────────────────────────────────────────────┘
```

#### LLM Agent 挖掘模式

```
CLI 输入: --mine-llm
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6a: LLM 初始化                                                 │
│  输入: StockData, BackendBase                                        │
│  处理: BaseMiner 预计算前向收益率 → 初始化状态容器                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6b: LLM 迭代循环 (max_iterations 轮)                           │
│                                                                     │
│  第 1 轮: build_initial_prompt(n) → LLM 生成                         │
│  后续轮: build_iteration_prompt(n, best[:5], failures[-5:], ...)     │
│       → LLM 生成                                                    │
│                                                                     │
│  每轮: parse_llm_response(content) → 表达式列表                       │
│       → _evaluate_batch(expressions)                                 │
│         对每个表达式: compile → eval → backtest → metrics             │
│       → 更新 best_factors / recent_failures                          │
│       → 检查停滞条件 (连续 3 轮无改善则终止)                           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 6c: 结果汇总                                                   │
│  输出: list[(expression, metrics)] — top 20 最佳因子按 IR 降序        │
└─────────────────────────────────────────────────────────────────────┘
```

**核心数据类型流转**：

```
list[str] codes
    → StockData (dataclass)
    → dict[str, ndarray]  (数据字段名 → ndarray)
    → ndarray(T, N)       (因子值)
    → BacktestResult      (分组收益 + NAV + IC + 多空)
    → dict                (13 项指标 + ic_decay)
    → PNG + Rich Table    (可视化 + 终端输出)
```

---

## 5. 配置系统

### TOML 配置文件结构

配置文件 `xi-alpha.toml` 位于项目根目录。Python 3.11+ 使用内置 `tomllib` 解析，3.10 通过 `tomli` 兼容。

```toml
# xi-alpha 向量化因子研究引擎配置文件
# 所有可调参数集中在此，代码中不再出现任何硬编码业务参数。

# ── 数据源 ──
[data]
start_date = "2021-01-01"
end_date = "2024-12-31"
request_interval = 1.0
max_retries = 3
base_delay = 2.0
adjust = "qfq"

# ⚠️ 幸存者偏差警告
# 默认股票池由人工挑选的约 50 只大盘蓝筹组成...

[data.cache]
dir = ".cache"

[data.stock_pool]
symbols = ["600519", "000858", ...]

# ── 因子 ──
[factors]
default_window = 20

[factors.classic]
short_term_reversal = "-1 * rolling_sum(returns, 5)"
volatility = "-1 * rolling_std(returns, 20)"
ma_deviation = "close / rolling_mean(close, 20) - 1"

# ── 回测 ──
[backtest]
n_groups = 5
trading_days_per_year = 252
n_jobs = 4
min_stocks_for_group = 2
min_stocks_for_ic = 3
ic_decay_days = 10

# ── 报告 ──
[report]
output_dir = "output"
dpi = 150
top_n_display = 5

[report.figure]
single = [16, 10]
batch = [18, 14]

[report.filename]
single = "factor_report.png"
batch = "batch_report.png"

[report.style]
colormap_groups = "RdYlGn"
colormap_corr = "coolwarm"
colormap_nav = "tab10"
line_color = "steelblue"

# ── 后端 ──
[backend]
type = "numpy"

# ── 日志 ──
[logging]
level = "INFO"
format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

### config.py 加载机制

```python
def load_config(path: Path | None = None) -> dict:
    """
    配置加载流程:
    1. 确定配置文件路径（优先级从高到低）:
       a. 显式传入的 path 参数
       b. XIALPHA_CONFIG 环境变量指定的路径
       c. 当前工作目录下的 xi-alpha.toml
       d. 项目根目录下的 xi-alpha.toml
    2. 读取 TOML 文件，与内置默认值深度合并（用户配置覆盖默认值）
    3. 返回完整配置字典
    """
    ...
```

**搜索路径**（按优先级排列）：

1. `path` 参数显式指定
2. `XIALPHA_CONFIG` 环境变量
3. `./xi-alpha.toml`（当前工作目录）
4. 项目根目录下的 `xi-alpha.toml`

**深度合并**：用户配置中的嵌套字段会逐层覆盖默认值，未指定的字段保留默认值。

**路径解析**：`CACHE_DIR` 和 `OUTPUT_DIR` 通过 `PROJECT_ROOT / 相对路径` 解析到项目根目录下：

```python
PROJECT_ROOT: Path = _find_config_path().parent
CACHE_DIR: Path = PROJECT_ROOT / _get("data", "cache", "dir")    # → 项目根目录/.cache/
OUTPUT_DIR: Path = PROJECT_ROOT / _get("report", "output_dir")   # → 项目根目录/output/
```

### 配置项一览表

| 配置键 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `data.start_date` | str | `"2021-01-01"` | 数据起始日期 |
| `data.end_date` | str | `"2024-12-31"` | 数据截止日期 |
| `data.request_interval` | float | `1.0` | API 请求间隔（秒） |
| `data.max_retries` | int | `3` | 单只股票最大重试次数 |
| `data.base_delay` | float | `2.0` | 退避初始间隔秒数 |
| `data.adjust` | str | `"qfq"` | 复权方式 |
| `data.cache.dir` | str | `".cache"` | 缓存目录（相对于项目根目录） |
| `data.stock_pool.symbols` | list | `[]` | 股票代码列表 |
| `factors.default_window` | int | `20` | 默认滚动窗口 |
| `factors.classic` | dict | `{}` | 经典因子定义（名称→表达式） |
| `backtest.n_groups` | int | `5` | 回测分组数量 |
| `backtest.trading_days_per_year` | int | `252` | 年化交易日数 |
| `backtest.n_jobs` | int | `4` | 并行进程数 |
| `backtest.min_stocks_for_group` | int | `2` | 分组所需的最小截面股票数 |
| `backtest.min_stocks_for_ic` | int | `3` | IC 计算所需的最小截面股票数 |
| `backtest.ic_decay_days` | int | `10` | IC 衰减计算的最大持有天数 |
| `report.output_dir` | str | `"output"` | 报告输出目录（相对于项目根目录） |
| `report.dpi` | int | `150` | 输出图片 DPI |
| `report.top_n_display` | int | `5` | 批量报告展示因子数 |
| `report.figure.single` | list | `[16, 10]` | 单因子图尺寸 |
| `report.figure.batch` | list | `[18, 14]` | 批量报告图尺寸 |
| `report.filename.single` | str | `"factor_report.png"` | 单因子报告文件名 |
| `report.filename.batch` | str | `"batch_report.png"` | 批量报告文件名 |
| `report.style.colormap_groups` | str | `"RdYlGn"` | 分组净值色谱 |
| `report.style.colormap_corr` | str | `"coolwarm"` | 相关性矩阵色谱 |
| `report.style.colormap_nav` | str | `"tab10"` | 多空净值色谱 |
| `report.style.line_color` | str | `"steelblue"` | 单线图默认颜色 |
| `backend.type` | str | `"numpy"` | 计算后端 |
| `logging.level` | str | `"INFO"` | 日志级别 |
| `logging.format` | str | (格式字符串) | 日志格式 |
| `mining.gp.population_size` | int | `100` | GP 种群大小 |
| `mining.gp.n_generations` | int | `50` | 进化代数 |
| `mining.gp.crossover_prob` | float | `0.8` | 交叉概率 |
| `mining.gp.mutation_prob` | float | `0.15` | 变异概率 |
| `mining.gp.max_depth` | int | `4` | 表达式树最大深度 |
| `mining.gp.tournament_size` | int | `3` | 锦标赛选择大小 |
| `mining.gp.elites` | int | `5` | 精英保留数量 |
| `mining.gp.objectives` | list | `["ir", "sharpe"]` | NSGA-II 优化目标（`-` 前缀取反） |
| `mining.gp.seed` | int | `42` | 随机种子 |
| `mining.llm.api_url` | str | `"http://localhost:39001/v1/chat/completions"` | LLM API 地址 |
| `mining.llm.model_name` | str | `"glm-5.1"` | 模型名称 |
| `mining.llm.api_key` | str | `""` | API Key |
| `mining.llm.max_iterations` | int | `10` | 最大迭代轮数 |
| `mining.llm.factors_per_iteration` | int | `5` | 每轮生成因子数 |
| `mining.llm.temperature` | float | `0.7` | 采样温度 |
| `mining.llm.max_tokens` | int | `2048` | 单次回复最大 token 数 |

---

## 6. 安全设计

因子表达式由用户输入并最终通过 Python eval 执行，因此必须防止任意代码注入。xi-alpha 采用 AST 级别的安全控制。

### AST 白名单机制

编译器在 `_validate_ast()` 阶段遍历 AST 树，仅允许以下节点类型：

| 允许的 AST 节点 | 用途 |
|----------------|------|
| `ast.Expression` | 表达式根节点 |
| `ast.BinOp` | 二元运算 (+, -, *, /) |
| `ast.UnaryOp` | 一元运算 (-, +) |
| `ast.Call` | 函数调用（仅限白名单内算子） |
| `ast.Name` | 变量引用（仅限白名单内数据字段） |
| `ast.Constant` | 数字常量 |
| `ast.Load` | 值加载 |
| `ast.Add` / `ast.Sub` / `ast.Mult` / `ast.Div` | 二元运算符 |
| `ast.USub` / `ast.UAdd` | 一元运算符 |

### 禁止的 AST 节点

以下节点类型出现时直接拒绝（抛出 `ValueError`）：

- `ast.Import` / `ast.ImportFrom` — 禁止导入模块
- `ast.Assign` / `ast.AugAssign` — 禁止赋值
- `ast.Attribute` — 禁止属性访问（防止 `__import__` 等）
- `ast.Subscript` — 禁止下标访问（防止 `__builtins__` 等）
- `ast.Lambda` — 禁止匿名函数
- `ast.ListComp` / `ast.DictComp` / `ast.SetComp` — 禁止推导式
- `ast.FunctionDef` / `ast.ClassDef` — 禁止定义函数和类
- `ast.BoolOp` / `ast.Compare` / `ast.IfExp` — 禁止逻辑和比较操作
- `ast.Dict` / `ast.List` / `ast.Tuple` / `ast.Set` — 禁止复合类型构造
- `ast.Try` / `ast.Raise` / `ast.Assert` — 禁止异常和控制流操作

### 执行沙箱

编译后的表达式在受限环境中执行：

```python
sandbox_globals = {
    "__builtins__": {},              # 清空内置函数
    # 以下为白名单注入的算子函数
    "rolling_mean":   OPERATOR_REGISTRY["rolling_mean"],
    "rolling_std":    OPERATOR_REGISTRY["rolling_std"],
    "rank":           OPERATOR_REGISTRY["rank"],
    "zscore":         OPERATOR_REGISTRY["zscore"],
    # ... 其余算子
}

sandbox_locals = {
    "open":     stock_data.open,
    "close":    stock_data.close,
    "high":     stock_data.high,
    "low":      stock_data.low,
    "volume":   stock_data.volume,
    "amount":   stock_data.amount,
    "returns":  stock_data.returns,
}

result = eval(compiled_code, sandbox_globals, sandbox_locals)
```

`__builtins__: {}` 确保表达式无法访问任何 Python 内置函数（如 `open()`、`exec()`、`eval()`、`__import__()`），只能调用白名单中的算子。

---

## 7. 扩展指南

### 7.1 添加新算子

在 `factor/operators.py` 中使用 `@_register` 装饰器注册新算子：

```python
# factor/operators.py

@_register("cumsum")               # 注册名称，表达式中使用这个名字
def _cumsum(*args, stock_data=None, backend=None):
    (x,) = args
    return backend.cumsum(x)
```

注意事项：
- 算子函数通过 `*args` 接收操作数，通过关键字参数接收 `stock_data` 和 `backend`
- 返回值必须是同形状的 ndarray
- 如果算子需要新的后端方法，需要在 `BackendBase` 和 `NumPyBackend` 中同步添加

同时在 `compiler.py` 的白名单中添加算子名称（实际上编译器通过 `OPERATOR_REGISTRY` 动态校验，只要算子注册了就能通过校验）。

### 7.2 添加新后端

创建新文件继承 `BackendBase`：

```python
# backend/jax_backend.py

import numpy as np
from .base import BackendBase

class JaxBackend(BackendBase):
    """基于 JAX 的 GPU 加速后端"""

    def rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        import jax.numpy as jnp
        jdata = jnp.array(data)
        # JAX 实现滚动均值
        ...
        return np.asarray(result)

    # 实现其余 27 个抽象方法 ...
```

在 `auto.py` 中注册：

```python
# backend/auto.py
def get_backend() -> BackendBase:
    if config.BACKEND_TYPE == "jax":
        from .jax_backend import JaxBackend
        return JaxBackend()
    return NumPyBackend()
```

在配置文件中启用：

```toml
[backend]
type = "jax"
```

### 7.3 添加新数据源

在 `data/` 目录下创建新的加载器模块：

```python
# data/tushare_loader.py

from .loader import StockData

def load_from_tushare(
    codes: list[str],
    start_date: str,
    end_date: str,
    token: str,
) -> StockData:
    """从 Tushare 加载数据，返回标准 StockData 格式。"""
    import tushare as ts
    pro = ts.pro_api(token)
    ...
    return StockData(
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        volume=volume_arr,
        amount=amount_arr,
        returns=returns_arr,
        dates=dates_arr,
        symbols=np.array(codes, dtype=str),
    )
```

### 7.4 添加新因子到配置文件

在 TOML 配置文件的 `[factors.classic]` 节中追加：

```toml
[factors.classic]
my_custom_factor = "rolling_mean(close / shift(close, 1) - 1, 10) * rank(volume)"
```

添加后，使用 `--classic` 参数运行即可回测新因子：

```bash
xi-alpha --classic
```

无需修改任何 Python 代码。

### 7.5 自定义报告配色

在 `xi-alpha.toml` 的 `[report.style]` 节修改配色参数：

```toml
[report.style]
colormap_groups = "viridis"    # 分组净值色谱（任何 matplotlib colormap 名）
colormap_corr = "RdBu"        # 相关性矩阵色谱
colormap_nav = "Set2"         # 多空净值色谱
line_color = "#2196F3"        # 单线图颜色（支持 hex 颜色值）
```

这些配色参数在 `report/summary.py` 中通过 `config.COLORMAP_GROUPS` 等模块级变量读取，零硬编码。

### 7.6 自定义挖掘策略

继承 `BaseMiner` 实现自定义因子挖掘器。`BaseMiner` 已提供完整的编译→回测→指标评估管线，子类只需关注因子生成策略。

```python
# mining/my_miner.py

from .base import BaseMiner

class MyMiner(BaseMiner):
    """自定义挖掘策略示例。"""

    def mine(self) -> list[tuple[str, dict]]:
        # 1. 自定义因子生成逻辑
        candidates = self._generate_candidates()

        # 2. 使用 _evaluate_factor(expression) 评估每个因子
        for expr in candidates:
            metrics = self._evaluate_factor(expr)
            if metrics is not None:
                # 处理有效因子
                ...

        # 3. 使用 _evaluate_batch(expressions) 批量评估
        results = self._evaluate_batch(candidates)

        # 4. 返回 [(expression, metrics), ...] 按 IR 降序
        valid = [(expr, m) for expr, m in results if m is not None]
        valid.sort(key=lambda x: x[1]["ir"], reverse=True)
        return valid

    def _generate_candidates(self) -> list[str]:
        # 自定义因子表达式生成逻辑
        ...
```

关键接口说明：

| 方法 / 属性 | 说明 |
|-------------|------|
| `self._evaluate_factor(expression)` | 评估单个因子，返回指标字典或 `None` |
| `self._evaluate_batch(expressions)` | 批量评估，返回 `[(expr, metrics\|None), ...]` |
| `self._seen_expressions` | 去重集合，自动跳过已评估的表达式 |
| `self.stock_data` | 行情数据（`StockData` 实例） |
| `self.backend` | 计算后端（`BackendBase` 实例） |
| `self._forward_returns` | 预计算的前向收益率 ndarray |

注册自定义挖掘器到 CLI：

```python
# main.py 中添加 --mine-my 选项
from xialpha.mining.my_miner import MyMiner

elif args.mine_my:
    miner = MyMiner(stock_data, backend)
    results = miner.mine()
```
