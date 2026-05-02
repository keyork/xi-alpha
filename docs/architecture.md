# xi-alpha 系统架构文档

## 1. 概述

xi-alpha 是一个面向 A 股市场的向量化因子研究引擎。它提供从数据获取、因子计算、回测评估到可视化报告的完整流水线，目标是让因子研究员用最少的代码完成从因子表达式到回测结果的全流程验证。

### 设计原则

| 原则 | 说明 |
|------|------|
| **模块解耦** | 数据层、计算后端、因子引擎、回测引擎、报告生成各自独立，层间仅通过明确的接口交互 |
| **配置驱动** | 因子定义、回测参数、数据源选择均通过 TOML 配置文件控制，无需修改代码即可切换实验方案 |
| **NaN-safe** | 所有数值计算路径默认处理缺失值，backfill、滚动窗口、排名等操作对 NaN 有明确行为 |
| **安全沙箱** | 因子表达式通过 AST 编译执行，白名单机制限定可用算子和函数，杜绝任意代码执行风险 |

---

## 2. 系统架构图

```
CLI (main.py)
    │
    ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   Data   │ →  │ Backend  │ →  │  Factor  │ →  │ Backtest │ →  │  Report  │
│  Layer   │    │  Layer   │    │  Layer   │    │  Layer   │    │  Layer   |
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
  akshare         BackendBase      compiler         engine          summary
  loader.py       numpy_backend    operators.py     metrics.py      matplotlib
  cache.py        auto.py          library.py       batch.py

数据流向: 原始行情 → 标准化 ndarray → 因子值序列 → 分组回测结果 → 图表/表格
```

### 各层职责

| 层 | 输入 | 输出 | 职责 |
|----|------|------|------|
| Data Layer | 股票代码列表 + 日期范围 | `StockData` (含 OHLCV ndarray) | 从 akshare 拉取行情数据，缓存到本地 |
| Backend Layer | ndarray 运算请求 | ndarray 计算结果 | 抽象数值计算接口，当前提供 NumPy 实现 |
| Factor Layer | 因子表达式字符串 | `(T, N)` 因子值矩阵 | 编译表达式、调度算子、输出因子值 |
| Backtest Layer | 因子值矩阵 + 收益率矩阵 | `BacktestResult` (13 项指标) | 分组回测、多空组合、统计检验 |
| Report Layer | `BacktestResult` | PNG 图片 + 控制台表格 | 可视化报告生成 |

### 层间数据流

```
Data → Backend
    StockData.prices (DataFrame) → 通过 Backend 方法计算收益率等衍生指标

Backend → Factor
    Backend 实例注入 Factor Compiler，算子调用 Backend 方法完成实际计算

Factor → Backtest
    factor_values: ndarray(T, N) + returns: ndarray(T, N) → 输入回测引擎

Backtest → Report
    BacktestResult dict → 输入报告生成器
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

主入口函数。执行流程:

1. 检查缓存，命中则直接返回
2. 逐一拉取每只股票的行情数据
3. 双 API 降级策略: 优先调用 `ak.stock_zh_a_daily()` (东方财富)，失败时降级到 `ak.stock_zh_a_hist()` (新浪)
4. 指数退避重试: 初始间隔由配置决定，每次失败翻倍，最多重试次数可配置
5. 交易日对齐: 以所有股票交集的交易日为准，缺失交易日填 NaN
6. 写入缓存后返回

#### `cache.py`

基于 pickle 的文件缓存实现。

- 缓存路径: `{CACHE_DIR}/{start}_{end}_{md5_hash}.pkl`，默认位于 `src/xialpha/.cache/`
- 缓存 key: 对股票池排序后做 MD5 哈希取前 12 位，拼接日期范围
- 命中缓存时直接 pickle.load 返回，跳过全部网络请求

---

### 3.2 后端抽象层 (`backend/`)

后端层将数值计算抽象为统一接口，使得因子表达式与底层计算引擎解耦。

#### `base.py` — BackendBase ABC

定义 28 个抽象方法，覆盖因子计算所需的所有数值操作:

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

- 滚动操作: 通过 pandas `Series.rolling()` 桥接实现，利用 pandas 对 NaN 边界的成熟处理
- 截面排名: 使用 `scipy.stats.rankdata()` 计算百分位排名
- NaN 处理: 所有方法保证输出形状与输入一致，无效位置填 NaN

#### `auto.py` — 工厂函数

根据配置或环境自动选择后端:

```python
def create_backend(backend_name: str = "numpy") -> BackendBase:
    if backend_name == "numpy":
        return NumpyBackend()
    elif backend_name == "jax":
        # 预留: JAX GPU 加速后端
        raise NotImplementedError("JAX backend not yet implemented")
    elif backend_name == "torch":
        # 预留: PyTorch 后端
        raise NotImplementedError("Torch backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
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

**15 个已注册算子**:

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

因子表达式的编译执行引擎。核心流程:

```
表达式字符串
    │
    ▼  ast.parse()
AST 树
    │
    ▼  validate()   ← 安全白名单校验
校验后的 AST
    │
    ▼  compile()    ← 生成可调用对象
编译函数
    │
    ▼  eval()       ← 传入数据执行
ndarray 结果
```

**编译缓存**: 相同表达式只编译一次，后续直接从缓存取编译结果。

**关键函数**:

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

#### `library.py` — 经典因子库

从 TOML 配置文件的 `[factors.classic]` 节加载预定义因子集合，提供开箱即用的经典因子:

```python
# config.py 加载后提供:
CLASSIC_FACTORS: dict[str, str] = {
    "short_term_reversal": "-1 * rolling_sum(returns, 5)",
    "volatility": "-1 * rolling_std(returns, 20)",
    "ma_deviation": "close / rolling_mean(close, 20) - 1",
    ...
}

# library.py 消费:
def get_all_factors() -> list[tuple[str, str]]:
    return list(config.CLASSIC_FACTORS.items())
```

配置文件中的因子定义:

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

**分组规则**:

- 因子值从大到小排序
- `group 0` 对应因子值最高的一组（多头组）
- 最后一组对应因子值最低的一组（空头组）
- 多空组合 = group 0 收益率 - 最后一组收益率

```python
def run_backtest(
    factor_values: np.ndarray,   # (T, N) 因子值
    forward_returns: np.ndarray,  # (T, N) 前向收益率
    dates: np.ndarray,            # (T,) 日期序列
    n_groups: int = config.N_GROUPS,
) -> BacktestResult:
    ...
```

**百分位排名**: 因子值先做截面百分位排名，再按排名分组，避免异常值影响分组边界。

#### `metrics.py` — 指标计算

提供 13 项回测评估指标:

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

#### `batch.py` — 批量回测

支持同时对多个因子进行回测，并行执行:

```python
def batch_evaluate(
    factor_list: list[tuple[str, str]],   # [(因子名, 表达式), ...]
    stock_data: StockData,
    backend: BackendBase,
    forward_returns: np.ndarray,          # (T, N) 前向收益率
    dates: np.ndarray,                    # (T,) 日期
    n_groups: int = config.N_GROUPS,
    n_jobs: int = config.N_JOBS,
) -> tuple[list[tuple[str, BacktestResult, dict]], np.ndarray]:
    """并行批量回测，返回 (按IR排序的结果列表, 因子间相关系数矩阵)"""
    ...
```

**并行策略**: 使用 `ProcessPoolExecutor` 多进程并行，每个因子的回测在独立进程中执行。

**额外输出**:
- 因子间相关性矩阵: 计算所有因子两两之间的 IC 相关性
- IR 排序: 按 IR 值对所有因子降序排列，快速筛选有效因子

---

### 3.5 报告层 (`report/`)

#### `summary.py` — 报告生成器

将回测结果转化为可视化报告。

**单因子报告**: 2x2 子图布局

```
┌─────────────────────┬─────────────────────┐
│                     │                     │
│   分组累积净值曲线    │    多空净值 NAV 曲线   │
│   (分组色带 +        │    (steelblue 线)    │
│    多空线)           │                     │
│                     │                     │
├─────────────────────┼─────────────────────┤
│                     │                     │
│   日 IC 时序图       │   IC Decay 图        │
│   (柱状图/线图 +     │   (持有天数 vs IC)   │
│    均值虚线)         │                     │
│                     │                     │
└─────────────────────┴─────────────────────┘
```

**中文字体自适应**: 自动检测系统中可用的中文字体（SimHei、PingFang SC、Noto Sans CJK、WenQuanYi 等），无需手动配置。

**批量报告**: 对多个因子生成汇总表格，包含因子间 IC 相关性热力图和所有因子的 NAV 曲线叠加图。

---

## 4. 数据流

从命令行输入到最终报告的完整数据流:

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
│  输入: factor_values ndarray(T, N), returns ndarray(T, N)            │
│  输出: BacktestResult (dict 含 13 项指标 + 分组收益序列)                │
│  处理: 截面排名 → 分组 → 计算各组收益 → 多空组合 → 统计指标               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: 报告生成                                                    │
│  输入: BacktestResult                                                │
│  输出: PNG 图片文件 + 控制台表格                                        │
│  处理: matplotlib 绑图 → 保存/显示                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**核心数据类型流转**:

```
list[str] codes
    → StockData (dataclass)
    → dict[str, ndarray]  (数据字段名 → ndarray)
    → ndarray(T, N)       (因子值)
    → BacktestResult      (指标字典)
    → dict                (报告数据)
```

---

## 5. 配置系统

### TOML 配置文件结构

```toml
# xi-alpha 配置文件示例

[data]
start_date = "2021-01-01"
end_date = "2024-12-31"
request_interval = 1.0
max_retries = 3
base_delay = 2.0
adjust = "qfq"

[data.cache]
dir = ".cache"

[data.stock_pool]
symbols = ["600519", "000858", "000568", "601398"]

[backend]
type = "numpy"

[factors]
default_window = 20

[factors.classic]
volatility = "-1 * rolling_std(returns, 20)"
short_term_reversal = "-1 * rolling_sum(returns, 5)"
ma_deviation = "close / rolling_mean(close, 20) - 1"

[backtest]
n_groups = 5
trading_days_per_year = 252
n_jobs = 4

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
       d. src/xialpha/../../xi-alpha.toml（项目根目录）
    2. 读取 TOML 文件，与内置默认值深度合并（用户配置覆盖默认值）
    3. 返回完整配置字典
    """
    ...
```

**搜索路径**（按优先级排列）:

1. `path` 参数显式指定
2. `XIALPHA_CONFIG` 环境变量
3. `./xi-alpha.toml`（当前工作目录）
4. 项目根目录下的 `xi-alpha.toml`

**深度合并**: 用户配置中的嵌套字段会逐层覆盖默认值，未指定的字段保留默认值。

### 配置项一览表

| 配置键 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `data.start_date` | str | `"2021-01-01"` | 数据起始日期 |
| `data.end_date` | str | `"2024-12-31"` | 数据截止日期 |
| `data.request_interval` | float | `1.0` | API 请求间隔（秒） |
| `data.max_retries` | int | `3` | 单只股票最大重试次数 |
| `data.base_delay` | float | `2.0` | 退避初始间隔秒数 |
| `data.adjust` | str | `"qfq"` | 复权方式 |
| `data.cache.dir` | str | `".cache"` | 缓存目录 |
| `data.stock_pool.symbols` | list | `[]` | 股票代码列表 |
| `factors.default_window` | int | `20` | 默认滚动窗口 |
| `factors.classic` | dict | `{}` | 经典因子定义 (名称→表达式) |
| `backtest.n_groups` | int | `5` | 回测分组数量 |
| `backtest.trading_days_per_year` | int | `252` | 年化交易日数 |
| `backtest.n_jobs` | int | `4` | 并行进程数 |
| `report.output_dir` | str | `"output"` | 报告输出目录 |
| `report.dpi` | int | `150` | 输出图片 DPI |
| `report.top_n_display` | int | `5` | 批量报告展示因子数 |
| `report.figure.single` | list | `[16, 10]` | 单因子图尺寸 |
| `report.figure.batch` | list | `[18, 14]` | 批量报告图尺寸 |
| `report.filename.single` | str | `"factor_report.png"` | 单因子报告文件名 |
| `report.filename.batch` | str | `"batch_report.png"` | 批量报告文件名 |
| `backend.type` | str | `"numpy"` | 计算后端 |
| `logging.level` | str | `"INFO"` | 日志级别 |
| `logging.format` | str | (格式字符串) | 日志格式 |

---

## 6. 安全设计

因子表达式由用户输入并最终通过 Python eval 执行，因此必须防止任意代码注入。xi-alpha 采用 AST 级别的安全控制。

### AST 白名单机制

编译器在 `validate()` 阶段遍历 AST 树，仅允许以下节点类型:

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

以下节点类型出现时直接拒绝:

- `ast.Import` / `ast.ImportFrom` — 禁止导入模块
- `ast.Assign` / `ast.AugAssign` — 禁止赋值
- `ast.Attribute` — 禁止属性访问（防止 `__import__` 等）
- `ast.Subscript` — 禁止下标访问（防止 `__builtins__` 等）
- `ast.Lambda` — 禁止匿名函数
- `ast.ListComp` / `ast.DictComp` / `ast.SetComp` — 禁止推导式
- `ast.FunctionDef` / `ast.ClassDef` — 禁止定义函数和类

### 执行沙箱

编译后的表达式在受限环境中执行:

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

在 `factor/operators.py` 中使用 `@_register` 装饰器注册新算子:

```python
# factor/operators.py

@_register("cumsum")               # 注册名称，表达式中使用这个名字
def _cumsum(*args, stock_data=None, backend=None):
    (x,) = args
    return backend.cumsum(x)
```

注意事项:
- 算子函数通过 `*args` 接收操作数，通过关键字参数接收 `stock_data` 和 `backend`
- 返回值必须是同形状的 ndarray
- 如果算子需要新的后端方法，需要在 `BackendBase` 和 `NumPyBackend` 中同步添加

注意事项:
- 算子函数签名必须接受 ndarray 作为第一个参数
- 返回值必须是同形状的 ndarray
- 如果算子需要新的后端方法，需要在 `BackendBase` 和 `NumpyBackend` 中同步添加

同时在 `compiler.py` 的白名单中添加算子名称:

```python
ALLOWED_NAMES.add("ts_product")
```

### 7.2 添加新后端

创建新文件继承 `BackendBase`:

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

在 `auto.py` 中注册:

```python
# backend/auto.py
def get_backend() -> BackendBase:
    if config.BACKEND_TYPE == "jax":
        from .jax_backend import JaxBackend
        return JaxBackend()
    ...
```

在配置文件中启用:

```toml
[backend]
type = "jax"
```

### 7.3 添加新数据源

在 `data/` 目录下创建新的加载器模块:

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

在 TOML 配置文件的 `[factors.classic]` 节中追加:

```toml
[factors.classic]
my_custom_factor = "rolling_mean(close / shift(close, 1) - 1, 10) * rank(volume)"
```

添加后，使用 `--classic` 参数运行即可回测新因子:

```bash
xi-alpha --classic
```

无需修改任何 Python 代码。
