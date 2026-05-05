# xi-alpha

向量化 A 股因子研究引擎。通过 AST 编译器将因子表达式转为安全的向量化计算，配合分层分组回测框架，快速完成单因子检验和批量因子筛选。

## 特性

- **AST 因子表达式编译器** — 将字符串表达式解析为抽象语法树，在沙箱环境中安全执行，支持自定义算子扩展
- **分层分组回测** — 五分组多空组合，输出 IC、IR、Sharpe、最大回撤等核心指标
- **IC Decay 分析** — 计算因子在不同持有天数下的 IC 衰减曲线，辅助判断因子信号的有效持续时间
- **多空净值比值法** — 多空组合净值采用 `group_nav[0] / group_nav[N-1]` 比值法构建，分子分母量纲一致
- **批量因子并行评估** — 多因子并行计算，附带因子间相关性矩阵分析
- **Rich 终端美化** — 基于 Rich 库的面板头部、指标表格、进度条和计时器，替代原始 logging 输出，运行过程一目了然
- **TOML 配置驱动** — 数据源、回测参数、报告配色均通过 `xi-alpha.toml` 配置，零硬编码
- **GP+NSGA-II 因子挖掘** — 遗传编程自动生成因子表达式树，NSGA-II 多目标选择优化 IR 和 Sharpe
- **LLM Agent 因子挖掘** — 基于大语言模型的迭代式因子发现，根据回测反馈自动改进
- **后端抽象层** — 当前基于 NumPy/Pandas 实现，预留 JAX/PyTorch 后端接口

## 快速开始

要求 Python 3.10+，使用 [uv](https://docs.astral.sh/uv/) 管理依赖。

```bash
# 克隆并安装
git clone https://github.com/keyork/xi-alpha.git
cd xi-alpha
uv sync
```

运行经典因子（内置一组常用因子，开箱即用）：

```bash
uv run xi-alpha --classic
```

或者直接用 Python：

```bash
uv run python main.py --classic
```

运行单个因子表达式：

```bash
uv run xi-alpha --factor "rank(close) - rank(volume)"
```

因子挖掘：

```bash
# GP+NSGA-II 因子挖掘（自动进化搜索因子）
uv run xi-alpha --mine-gp

# LLM Agent 因子挖掘（需要配置 LLM API）
uv run xi-alpha --mine-llm
```

## 项目结构

```
xi-alpha/
├── main.py               # 根目录快捷入口（python main.py）
├── src/xialpha/          # 源码根目录
│   ├── backend/          # 计算后端抽象层（NumPy 实现 + JAX/Torch 预留）
│   ├── backtest/         # 分层分组回测引擎
│   ├── data/             # 数据获取与缓存（基于 akshare）
│   ├── display.py        # Rich 终端美化显示（面板、表格、进度条、计时器）
│   ├── factor/           # AST 因子表达式编译器与算子库
│   ├── mining/           # 因子挖掘（GP+NSGA-II 进化 / LLM Agent）
│   ├── report/           # 回测报告生成与可视化
│   ├── config.py         # 配置加载
│   └── main.py           # CLI 入口
├── .cache/               # 数据缓存目录（项目根目录）
├── output/               # 报告输出目录（项目根目录）
├── tests/                # 测试用例
├── xi-alpha.toml         # 默认配置文件
├── pyproject.toml        # 项目元数据与构建配置
└── README.md
```

## 配置说明

`xi-alpha.toml` 放在项目根目录或当前工作目录下。Python 3.11+ 使用内置 `tomllib` 解析，3.10 需安装 `tomli`。

> **幸存者偏差警告**：默认股票池由人工挑选的约 50 只大盘蓝筹组成，这些股票已经过市场长期验证，不包含已退市、ST、或长期表现不佳的标的。在此池子上得到的因子 IC / IR 可能显著高于全市场水平。生产环境应替换为指数成分股（如中证 500/1000）。

### 完整配置示例

```toml
# ── 数据源 ──
[data]
start_date = "2021-01-01"
end_date = "2024-12-31"
request_interval = 1.0    # 每次请求间隔（秒）
max_retries = 3           # 单只股票最大重试次数
base_delay = 2.0          # 首次退避等待（秒），后续指数翻倍
adjust = "qfq"            # 复权方式: qfq(前复权) / hfq(后复权) / ""(不复权)

[data.cache]
dir = ".cache"            # 缓存目录，位于项目根目录

[data.stock_pool]
symbols = ["600519", "000858", "000568", "601398"]  # 股票代码列表

# ── 因子 ──
[factors]
default_window = 20       # 默认滚动窗口

[factors.classic]
short_term_reversal = "-1 * rolling_sum(returns, 5)"
medium_term_momentum = "rolling_sum(returns, 20)"
volatility = "-1 * rolling_std(returns, 20)"
turnover = "-1 * rolling_mean(volume, 10)"
ma_deviation = "close / rolling_mean(close, 20) - 1"
price_volume_divergence = "-1 * rolling_corr(close, volume, 10)"

# ── 回测 ──
[backtest]
n_groups = 5              # 分组数
trading_days_per_year = 252
n_jobs = 4                # 并行进程数
min_stocks_for_group = 2  # 分组所需的最小截面股票数，低于此值跳过该期
min_stocks_for_ic = 3     # IC 计算所需的最小截面股票数
ic_decay_days = 10        # IC 衰减计算的最大持有天数

# ── 报告 ──
[report]
output_dir = "output"     # 输出目录，位于项目根目录
dpi = 150
top_n_display = 5         # 批量报告中展示前 N 个因子

[report.figure]
single = [16, 10]         # 单因子报告图尺寸 (宽, 高)
batch = [18, 14]          # 批量报告图尺寸

[report.filename]
single = "factor_report.png"
batch = "batch_report.png"

[report.style]
colormap_groups = "RdYlGn"    # 分组净值色谱
colormap_corr = "coolwarm"    # 相关性矩阵色谱
colormap_nav = "tab10"        # 多空净值色谱
line_color = "steelblue"      # 单线图默认颜色

# ── 因子挖掘 ──
[mining.gp]
population_size = 100     # GP 种群大小
n_generations = 50        # 进化代数
crossover_prob = 0.8      # 交叉概率
mutation_prob = 0.15      # 变异概率
max_depth = 4             # 表达式树最大深度
tournament_size = 3       # 锦标赛选择大小
elites = 5                # 精英保留数量
objectives = ["ir", "sharpe"]  # NSGA-II 优化目标
seed = 42                 # 随机种子

[mining.llm]
api_url = "http://localhost:39001/v1/chat/completions"  # OpenAI 兼容 API 地址
model_name = "glm-5.1"    # 模型名称
api_key = ""              # API Key（可为空）
max_iterations = 10       # 最大迭代轮数
factors_per_iteration = 5 # 每轮生成因子数
temperature = 0.7         # 采样温度
max_tokens = 2048         # 单次回复最大 token 数

# ── 后端 ──
[backend]
type = "numpy"            # numpy / jax / torch（预留）

# ── 日志 ──
[logging]
level = "INFO"
format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

## CLI 用法

```
xi-alpha [选项]
```

| 参数 | 说明 |
|------|------|
| `--factor EXPR` | 运行单个因子表达式 |
| `--factors-file PATH` | 从文件批量加载因子（每行一个表达式） |
| `--classic` | 运行内置经典因子组合 |
| `--start DATE` | 回测起始日期，格式 `YYYY-MM-DD` |
| `--end DATE` | 回测结束日期，格式 `YYYY-MM-DD` |
| `--groups N` | 分组数量，默认 5 |
| `--output DIR` | 报告输出目录 |
| `--no-cache` | 禁用数据缓存，强制重新获取 |
| `--mine-gp` | GP+NSGA-II 因子挖掘 |
| `--mine-llm` | LLM Agent 因子挖掘 |

示例：

```bash
# 指定日期范围，五分组回测
uv run xi-alpha --factor "rolling_mean(close, 20) / rolling_mean(close, 60)" --start 2022-01-01 --end 2024-06-30

# 批量评估因子文件，输出到指定目录
uv run xi-alpha --factors-file my_factors.txt --output ./my_reports

# 跳过缓存，全量拉取数据
uv run xi-alpha --classic --no-cache
```

## 因子表达式语法

因子表达式由数据字段和算子组合而成，支持嵌套调用。

### 数据字段

| 字段 | 含义 |
|------|------|
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额 |
| `returns` | 日收益率 |

### 算子

**截面算子**（在某个时间截面上对所有股票操作）：

| 算子 | 说明 | 示例 |
|------|------|------|
| `rank(x)` | 截面排名（归一化到 0-1） | `rank(close)` |
| `zscore(x)` | 截面标准化 | `zscore(volume)` |
| `demean(x)` | 截面去均值 | `demean(returns)` |

**时间序列算子**（沿时间轴滚动计算）：

| 算子 | 说明 | 示例 |
|------|------|------|
| `rolling_mean(x, n)` | n 日滚动均值 | `rolling_mean(close, 20)` |
| `rolling_std(x, n)` | n 日滚动标准差 | `rolling_std(returns, 20)` |
| `rolling_sum(x, n)` | n 日滚动求和 | `rolling_sum(returns, 5)` |
| `rolling_max(x, n)` | n 日滚动最大值 | `rolling_max(high, 10)` |
| `rolling_min(x, n)` | n 日滚动最小值 | `rolling_min(low, 10)` |
| `rolling_corr(x, y, n)` | n 日滚动相关系数 | `rolling_corr(close, volume, 10)` |
| `shift(x, n)` | 平移 n 期（正数向后，负数向前） | `shift(close, -1)` |
| `pct_change(x, n)` | n 期变化率 | `pct_change(close, 5)` |
| `diff(x, n)` | n 期差值 | `diff(close, 1)` |

**逐元素算子**：

| 算子 | 说明 | 示例 |
|------|------|------|
| `log(x)` | 自然对数 | `log(volume)` |
| `abs(x)` | 绝对值 | `abs(returns)` |
| `sign(x)` | 符号函数 | `sign(returns)` |

**算术运算**：`+`, `-`, `*`, `/`, 以及括号分组。

### 表达式示例

```python
# 短期反转：5 日收益率的反向
-1 * rolling_sum(returns, 5)

# 波动率因子：20 日波动率的截面标准化
zscore(rolling_std(returns, 20))

# 量价背离：价格与成交量的滚动相关性取反
-1 * rolling_corr(close, volume, 10)

# 均线偏离：价格偏离 20 日均线的程度
close / rolling_mean(close, 20) - 1

# 量价排名差
rank(close) - rank(volume)
```

## 因子挖掘

xi-alpha 提供两种自动因子挖掘方式：基于遗传编程（GP+NSGA-II）的进化搜索，以及基于大语言模型（LLM Agent）的迭代发现。

### GP+NSGA-II 因子挖掘

#### 原理

GP+NSGA-II 将因子搜索建模为多目标优化问题，流程如下：

1. **初始化** — 随机生成 `population_size` 棵表达式树，组成初始种群
2. **评估** — 对每个个体编译并回测，计算目标函数值（如 IR、Sharpe）
3. **非支配排序** — 使用 NSGA-II 算法按 Pareto 前沿分层，兼顾多个优化目标
4. **选择** — 锦标赛选择（`tournament_size` 参与者中选最优）
5. **交叉** — 以 `crossover_prob` 概率随机交换两棵树的子树
6. **变异** — 以 `mutation_prob` 概率随机替换一个子树
7. **精英保留** — 保留前 `elites` 个最优个体直接进入下一代
8. 循环 2-7 直到 `n_generations` 代结束

#### 关键参数

| 参数 | 说明 |
|------|------|
| `population_size` | 种群大小，越大搜索越广但越慢 |
| `n_generations` | 进化代数 |
| `max_depth` | 表达式树最大深度，控制因子复杂度 |
| `objectives` | 优化目标列表，支持 `ir`、`sharpe`、`-max_drawdown`、`-daily_turnover`（`-` 前缀表示最小化） |
| `elites` | 精英保留数量 |
| `tournament_size` | 锦标赛选择大小 |
| `crossover_prob` | 交叉概率 |
| `mutation_prob` | 变异概率 |
| `seed` | 随机种子 |

#### 表达式树操作

**终端节点**：
- 数据字段（85% 概率）：`open`、`high`、`low`、`close`、`volume`、`amount`、`returns`
- 常量（15% 概率）：随机浮点数

**算子节点**：
- 一元算子：`rank`、`zscore`、`demean`、`log`、`abs`、`sign`
- 窗口算子：`rolling_mean`、`rolling_std`、`rolling_sum`、`rolling_max`、`rolling_min`
- 算术运算：`+`、`-`、`*`、`/`
- 双变量算子：`rolling_corr`

**交叉**：随机选择两棵树中的一个节点，交换以该节点为根的子树。

**变异**：随机选择树中的一个节点，用新生成的随机子树替换。

#### 使用示例

```bash
# 使用默认配置运行 GP 因子挖掘
uv run xi-alpha --mine-gp

# 指定日期范围
uv run xi-alpha --mine-gp --start 2022-01-01 --end 2024-06-30
```

### LLM Agent 因子挖掘

#### 原理

LLM Agent 利用大语言模型的代码生成能力，通过迭代反馈循环发现有效因子：

1. **初始生成** — LLM 根据可用算子描述自由生成因子表达式
2. **编译回测** — 对每个表达式编译并执行回测，收集指标
3. **反馈改进** — 将 top 因子指标和失败表达式反馈给 LLM，引导改进方向
4. **迭代循环** — 重复 2-3 直到 `max_iterations` 轮结束

#### Agent 设计

- **第一轮**：LLM 根据算子列表和语法规则自由生成因子
- **后续轮**：向 LLM 展示当前 top 因子的指标（IR、IC、Sharpe）和编译失败的表达式，引导改进方向
- **自动去重**：已评估的表达式会被记录，不会重复提交给 LLM
- **提前终止**：连续 3 轮无 IR 改善则自动终止

#### API 配置

LLM Agent 使用兼容 OpenAI 规范的 API（`/v1/chat/completions`），需要配置以下参数：

| 参数 | 说明 |
|------|------|
| `api_url` | API 地址，如 `http://localhost:39001/v1/chat/completions` |
| `model_name` | 模型名称 |
| `api_key` | API Key，可为空 |
| `max_iterations` | 最大迭代轮数 |
| `factors_per_iteration` | 每轮生成因子数 |
| `temperature` | 采样温度 |
| `max_tokens` | 单次回复最大 token 数 |

#### 使用示例

```bash
# 运行 LLM Agent 因子挖掘（需要先在 xi-alpha.toml 中配置 API）
uv run xi-alpha --mine-llm

# 指定日期范围
uv run xi-alpha --mine-llm --start 2022-01-01 --end 2024-06-30
```

### 输出

两种挖掘方式共享相同的输出格式：

**终端显示**：运行结束后以 Rich 表格展示 top 10 因子，包含以下字段：

| 字段 | 说明 |
|------|------|
| IR | 信息比率 |
| IC | 信息系数均值 |
| Sharpe | 夏普比率 |
| 表达式 | 因子表达式字符串 |

**JSON 文件**：完整结果保存为 JSON 文件，路径分别为：
- GP 挖掘：`output/mining_gp.json`
- LLM 挖掘：`output/mining_llm.json`

## 开发

```bash
# 安装开发依赖
uv sync --dev

# 运行全部测试
uv run pytest tests/

# 运行单个测试文件
uv run pytest tests/test_factor.py -v
```

核心依赖：numpy, pandas, akshare, scipy, matplotlib, rich, httpx。

## 许可证

MIT License
