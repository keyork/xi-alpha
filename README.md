# xi-alpha

向量化 A 股因子研究引擎。通过 AST 编译器将因子表达式转为安全的向量化计算，配合分层分组回测框架，快速完成单因子检验和批量因子筛选。

## 特性

- **AST 因子表达式编译器** — 将字符串表达式解析为抽象语法树，在沙箱环境中安全执行，支持自定义算子扩展
- **分层分组回测** — 五分组多空组合，输出 IC、IR、Sharpe、最大回撤等 13 项核心指标
- **批量因子并行评估** — 多因子并行计算，附带因子间相关性矩阵分析
- **TOML 配置驱动** — 数据源、回测参数、报告输出均通过 `xi-alpha.toml` 配置，零硬编码
- **后端抽象层** — 当前基于 NumPy/Pandas 实现，预留 JAX/PyTorch 后端接口

## 快速开始

要求 Python 3.10+。

```bash
# 克隆并安装
git clone https://github.com/your-org/xi-alpha.git
cd xi-alpha
pip install -e .
```

运行经典因子（内置一组常用因子，开箱即用）：

```bash
xi-alpha --classic
```

运行单个因子表达式：

```bash
xi-alpha --factor "rank(close) - rank(volume)"
```

## 项目结构

```
xi-alpha/
├── src/xialpha/          # 源码根目录
│   ├── backend/          # 计算后端抽象层（NumPy 实现 + JAX/Torch 预留）
│   ├── backtest/         # 分层分组回测引擎
│   ├── data/             # 数据获取与缓存（基于 akshare）
│   ├── factor/           # AST 因子表达式编译器与算子库
│   ├── report/           # 回测报告生成与可视化
│   └── main.py           # CLI 入口
├── tests/                # 测试用例（35 个）
├── xi-alpha.toml         # 默认配置文件
├── pyproject.toml        # 项目元数据与构建配置
└── README.md
```

## 配置说明

`xi-alpha.toml` 放在项目根目录或当前工作目录下。Python 3.11+ 使用内置 `tomllib` 解析，3.10 需安装 `tomli`。

主要配置节：

```toml
[data]
start_date = "2021-01-01"
end_date = "2024-12-31"
request_interval = 1.0
max_retries = 3
adjust = "qfq"

[data.stock_pool]
symbols = ["600519", "000858", "000568", "601398"]

[factors.classic]
volatility = "-1 * rolling_std(returns, 20)"
short_term_reversal = "-1 * rolling_sum(returns, 5)"

[backtest]
n_groups = 5
trading_days_per_year = 252
n_jobs = 4

[report]
output_dir = "output"
dpi = 150

[backend]
type = "numpy"
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

示例：

```bash
# 指定日期范围，五分组回测
xi-alpha --factor "rolling_mean(close, 20) / rolling_mean(close, 60)" --start 2022-01-01 --end 2024-06-30

# 批量评估因子文件，输出到指定目录
xi-alpha --factors-file my_factors.txt --output ./my_reports

# 跳过缓存，全量拉取数据
xi-alpha --classic --no-cache
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

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/

# 运行单个测试文件
pytest tests/test_factor.py -v
```

核心依赖：numpy, pandas, akshare, scipy, matplotlib。

## 许可证

MIT License
