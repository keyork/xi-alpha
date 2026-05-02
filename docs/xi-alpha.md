# XI-Alpha 技术设计文档

## 一、项目定位

专为因子研究设计的向量化回测引擎。名称取自《三体》中质子展开到第十一维的概念——因子研究的本质是把低维价格数据展开到高维空间搜索 alpha 信号。定位是因子挖掘流水线的验证层——接收因子表达式，秒级输出评估报告。MVP 使用 NumPy 实现，后续扩展 JAX 和 Torch 后端。

---

## 二、目录结构

```
xi-alpha/
├── config.py                # 全局配置（数据路径、默认参数）
├── main.py                  # 入口脚本
│
├── data/
│   ├── loader.py            # 数据加载与标准化
│   └── cache.py             # 本地数据缓存（避免重复拉取）
│
├── backend/
│   ├── base.py              # 后端抽象基类
│   ├── numpy_backend.py     # NumPy 实现（MVP）
│   └── auto.py              # 后端自动选择（MVP 阶段直接返回 NumPy）
│
├── factor/
│   ├── operators.py         # 算子注册表 + 高层算子函数
│   ├── compiler.py          # 字符串表达式 → 可调用函数
│   └── library.py           # 内置经典因子定义
│
├── backtest/
│   ├── engine.py            # 回测核心（分组、多空收益）
│   ├── metrics.py           # 评估指标计算
│   └── batch.py             # 批量因子并行评估
│
├── report/
│   └── summary.py           # 文字报告 + matplotlib 图表
│
└── tests/
    ├── test_operators.py    # 算子单元测试
    ├── test_engine.py       # 回测引擎测试
    └── test_classic.py      # 经典因子结果校验
```

---

## 三、数据层（data/）

### 3.1 loader.py

**职责：** 从数据源拉取行情数据，转换为标准化的内部数据结构。

**数据源：** MVP 使用 AKShare（免费、无需注册）。拉取 A 股日频行情数据。

**核心输出数据结构 `StockData`：** 一个数据类或字典，包含以下字段，每个字段都是二维 ndarray，shape 为 `(T, N)`，其中 T 是交易日数，N 是股票数。

| 字段名     | 含义     | 说明              |
| ---------- | -------- | ----------------- |
| `open`     | 开盘价   |                   |
| `high`     | 最高价   |                   |
| `low`      | 最低价   |                   |
| `close`    | 收盘价   | 最常用            |
| `volume`   | 成交量   |                   |
| `amount`   | 成交额   |                   |
| `returns`  | 日收益率 | 由 close 计算得出 |
| `dates`    | 日期数组 | 一维，长度 T      |
| `symbols`  | 股票代码 | 一维，长度 N      |

**关键处理逻辑：**

1. 逐只股票拉取日线数据
2. 对齐所有股票到相同的交易日序列（以沪深交易日历为准）
3. 停牌日数据填充为 NaN（不是前值填充，因为前值填充会引入虚假信号）
4. 计算日收益率：`returns[t] = close[t] / close[t-1] - 1`
5. 所有字段从 DataFrame 转为 ndarray 存储

**接口设计：**

- `load_stock_data(start_date, end_date, stock_pool)` → `StockData`
- `stock_pool` 参数支持传入股票代码列表，MVP 阶段先写死沪深300成分股或随机选50-100只

### 3.2 cache.py

**职责：** 首次拉取的数据存为本地 pickle 或 parquet 文件，后续直接读本地。

**逻辑：** 以 `{start_date}_{end_date}_{pool_hash}.pkl` 为文件名，存在则直接 load，不存在则调 loader 拉取后保存。

---

## 四、后端抽象层（backend/）

### 4.1 base.py

**职责：** 定义所有数值计算操作的抽象接口。上层代码只调这些接口，不直接调用 np/jnp/torch。

**需要定义的抽象方法（按类别）：**

**时序算子（沿 axis=0 即时间轴计算）：**

| 方法              | 输入              | 输出    | 说明                    |
| ----------------- | ----------------- | ------- | ----------------------- |
| `rolling_mean`    | x (T×N), window   | T×N     | 滚动均值                |
| `rolling_std`     | x (T×N), window   | T×N     | 滚动标准差              |
| `rolling_max`     | x (T×N), window   | T×N     | 滚动最大值              |
| `rolling_min`     | x (T×N), window   | T×N     | 滚动最小值              |
| `rolling_sum`     | x (T×N), window   | T×N     | 滚动求和                |
| `rolling_corr`    | x, y (T×N), window| T×N     | 滚动相关系数            |
| `shift`           | x (T×N), periods  | T×N     | 时序平移，空位填 NaN    |
| `pct_change`      | x (T×N), periods  | T×N     | 百分比变化              |
| `diff`            | x (T×N), periods  | T×N     | 差分                    |
| `cumsum`          | x (T×N)           | T×N     | 累积求和                |
| `cumprod`         | x (T×N)           | T×N     | 累积求积                |

**截面算子（沿 axis=1 即股票轴计算）：**

| 方法            | 输入    | 输出    | 说明                             |
| --------------- | ------- | ------- | -------------------------------- |
| `rank`          | x (T×N) | T×N     | 截面排名，输出百分位 0~1         |
| `zscore`        | x (T×N) | T×N     | 截面 z-score 标准化              |
| `demean`        | x (T×N) | T×N     | 截面去均值                       |
| `clip`          | x, lo, hi | T×N  | 截面截断                         |

**基础运算：**

| 方法    | 说明                       |
| ------- | -------------------------- |
| `add`   | 逐元素加                   |
| `sub`   | 逐元素减                   |
| `mul`   | 逐元素乘                   |
| `div`   | 逐元素除（分母为0返回NaN） |
| `log`   | 逐元素取对数               |
| `abs`   | 逐元素取绝对值             |
| `sign`  | 逐元素取符号               |
| `where` | 条件选择                   |
| `nanmean` | 忽略 NaN 的均值          |

**相关性计算：**

| 方法         | 说明                                  |
| ------------ | ------------------------------------- |
| `cross_corr` | 截面相关系数，逐行计算 x 和 y 的相关  |

**NaN 处理：**

所有算子必须正确处理 NaN。NaN 代表停牌或数据不足，不能参与计算，也不能污染其他有效值。这是整个框架正确性的基石。

### 4.2 numpy_backend.py

**职责：** 用 NumPy 实现 base.py 中定义的所有抽象方法。

**实现要点：**

- 滚动窗口算子：用 `np.lib.stride_tricks.sliding_window_view` 或手动用循环+切片实现。stride_tricks 方式更快但对 NaN 处理不方便，建议先用 pandas 的 rolling 做一层桥接（ndarray → Series → rolling → ndarray），性能不是瓶颈。
- `rank`：用 `scipy.stats.rankdata` 或 `np.argsort(np.argsort(x))` 实现。注意处理 NaN：NaN 不参与排名，排名结果中对应位置也应为 NaN。
- 所有方法的输入输出都是 ndarray，不是 DataFrame。

### 4.3 auto.py

**职责：** 根据环境和任务自动选择后端。

**MVP 逻辑：** 直接返回 NumPyBackend 实例。预留接口，后续加 JAX/Torch 判断。

---

## 五、因子层（factor/）

### 5.1 operators.py

**职责：** 定义高层因子算子函数，作为 AST 表达式中可以调用的"词汇表"。每个算子内部调用 backend 的方法。

**算子注册机制：** 维护一个字典 `OPERATOR_REGISTRY`，key 是算子名字符串，value 是对应的函数引用。AST 编译器通过这个字典把表达式中的算子名解析为具体函数。

**需要注册的算子列表：**

| 算子名            | 语义                                        | 调用的 backend 方法 |
| ----------------- | ------------------------------------------- | ------------------- |
| `rolling_mean`    | 滚动均值                                    | `backend.rolling_mean` |
| `rolling_std`     | 滚动标准差                                  | `backend.rolling_std` |
| `rolling_max`     | 滚动最大值                                  | `backend.rolling_max` |
| `rolling_min`     | 滚动最小值                                  | `backend.rolling_min` |
| `rolling_sum`     | 滚动求和                                    | `backend.rolling_sum` |
| `rolling_corr`    | 滚动相关系数                                | `backend.rolling_corr` |
| `shift`           | 时序平移                                    | `backend.shift` |
| `pct_change`      | 百分比变化                                  | `backend.pct_change` |
| `diff`            | 差分                                        | `backend.diff` |
| `rank`            | 截面排名（输出百分位）                       | `backend.rank` |
| `zscore`          | 截面 z-score                                | `backend.zscore` |
| `demean`          | 截面去均值                                  | `backend.demean` |
| `log`             | 取对数                                      | `backend.log` |
| `abs`             | 取绝对值                                    | `backend.abs` |
| `sign`            | 取符号                                      | `backend.sign` |

**数据字段引用：** 除了算子之外，表达式中还可以直接引用数据字段，如 `close`、`open`、`volume`、`returns`。编译器需要把这些名字解析为 StockData 中对应的 ndarray。

### 5.2 compiler.py

**职责：** 把字符串形式的因子表达式编译成可调用的 Python 函数。

**输入：** 字符串表达式，如 `"rank(rolling_mean(close, 5) / rolling_mean(close, 20))"`

**输出：** 一个函数 `fn(stock_data, backend) → ndarray (T×N)`

**编译流程：**

1. **解析：** 用 Python 内置的 `ast` 模块将字符串解析为 AST
2. **校验：** 遍历 AST 节点，检查所有函数调用名是否在 OPERATOR_REGISTRY 中，所有变量名是否在数据字段列表中。不在的就报错，防止注入任意代码
3. **安全检查：** 禁止 import、exec、eval、文件操作等危险节点
4. **编译：** 将校验通过的 AST 用 `compile()` 编译为 code object
5. **封装：** 包装成函数，调用时传入 stock_data 和 backend，在受限的命名空间中执行

**注意事项：**

- 表达式中的算术运算符（+、-、*、/）直接映射为逐元素运算
- 常量（如 `rolling_mean(close, 20)` 中的 `20`）直接作为参数传递
- 编译结果应缓存，同一个表达式不需要重复编译

### 5.3 library.py

**职责：** 定义一组内置的经典因子，用于校验回测引擎的正确性。

**需要内置的经典因子：**

| 因子名          | 表达式                                                   | 预期行为（A股）          |
| --------------- | -------------------------------------------------------- | ------------------------ |
| 短期反转        | `-1 * rolling_sum(returns, 5)`                           | IC 应为正（A股反转效应） |
| 中期动量        | `rolling_sum(returns, 20)`                               | IC 可能为弱正或不显著    |
| 波动率          | `-1 * rolling_std(returns, 20)`                          | IC 应为正（低波动异象）  |
| 换手率          | `-1 * rolling_mean(volume, 10)`                          | IC 应为正（低换手溢价）  |
| 均线偏离        | `close / rolling_mean(close, 20) - 1`                    | 取决于市场周期           |
| 量价背离        | `-1 * rolling_corr(close, volume, 10)`                   | IC 通常为弱正            |

这些因子的预期行为是公认的，如果回测结果和预期方向一致，说明引擎逻辑大概率没问题。如果方向完全反了，大概率是代码有 bug（常见错误：收益率没有正确 shift、排名方向搞反了）。

---

## 六、回测引擎（backtest/）

### 6.1 engine.py

**职责：** 回测的核心逻辑——接收因子值矩阵，输出分组收益和多空组合收益。

**核心函数 `run_backtest(factor_values, forward_returns, n_groups=5)`：**

**输入：**

- `factor_values`：ndarray (T×N)，每只股票每天的因子值
- `forward_returns`：ndarray (T×N)，每只股票的下一期收益率（调用方负责 shift）
- `n_groups`：分组数，默认5

**处理流程：**

1. **NaN 掩码生成：** 标记 factor_values 或 forward_returns 中任一为 NaN 的位置。这些位置不参与后续计算。

2. **截面排名：** 对每一行（每个交易日），将有效股票按因子值排名，输出百分位（0~1）。NaN 位置的排名结果也为 NaN。

3. **分组标签：** 百分位乘以 n_groups 向下取整得到组号（0 到 n_groups-1）。注意边界处理：百分位恰好等于 1.0 的应归入最后一组。

4. **分组收益计算：** 对每个组号，用布尔掩码选出该组的股票，算这些股票 forward_returns 的等权平均值。输出一个字典，key 是组号，value 是长度为 T 的收益率序列。

5. **多空组合：** 第0组（因子值最高）减去最后一组（因子值最低）的收益率序列。这就是因子的多空收益。

6. **累计净值：** 对每组的收益率序列做 cumprod(1 + r) 得到净值曲线。

**输出 `BacktestResult`：** 一个数据类或字典，包含：

| 字段               | 类型              | 说明                          |
| ------------------ | ----------------- | ----------------------------- |
| `group_returns`    | dict[int, 1D arr] | 每组的日收益率序列            |
| `group_nav`        | dict[int, 1D arr] | 每组的累计净值曲线            |
| `long_short`       | 1D arr            | 多空组合日收益率序列          |
| `long_short_nav`   | 1D arr            | 多空组合累计净值曲线          |
| `ic_series`        | 1D arr            | 每日 IC 序列                  |
| `dates`            | 1D arr            | 日期数组                      |
| `n_groups`         | int               | 分组数                        |

**关键注意事项：**

- forward_returns 的对齐：因子值是 T 日算的，收益率应该是 T+1 日的。这个 shift 应该在调用 engine 之前完成（在 main.py 中做），而不是在 engine 内部做。职责分离，避免重复 shift。
- 每天参与计算的有效股票数可能不同（停牌导致 NaN），分组时只对有效股票分组。
- 分组要保证每组的股票数尽量均等。

### 6.2 metrics.py

**职责：** 基于 BacktestResult 计算所有评估指标。

**核心函数 `calc_metrics(result: BacktestResult) → dict`：**

需要计算的指标及其计算方法：

| 指标               | 计算方法                                                                             | 说明                       |
| ------------------ | ------------------------------------------------------------------------------------ | -------------------------- |
| IC 均值            | `mean(ic_series)`                                                                    | 因子预测力的平均水平       |
| IC 标准差          | `std(ic_series)`                                                                     | 因子预测力的稳定性         |
| IR                 | `IC均值 / IC标准差`                                                                   | 风险调整后的预测力         |
| IC > 0 占比        | `sum(ic_series > 0) / len(ic_series)`                                                | 因子方向正确的天数占比     |
| 年化收益率         | `mean(long_short) * 252`                                                              | 多空组合的年化期望收益     |
| 年化波动率         | `std(long_short) * sqrt(252)`                                                         | 多空组合的年化风险         |
| 夏普比率           | `年化收益率 / 年化波动率`                                                              | 每单位风险的收益           |
| 最大回撤           | 净值序列从峰值到谷底的最大跌幅：`max(1 - nav / cummax(nav))`                           | 最坏情况下的亏损           |
| 最大回撤持续天数   | 从峰值到回撤结束（净值回到峰值）的最长天数                                              | 恢复能力                   |
| 日均换手率         | 相邻两日分组标签的变化：每天有多少股票从一个组换到另一个组，占总股票数的比例，取均值       | 交易成本的代理指标         |
| 多空收益 t 检验    | 对 long_short 序列做单样本 t 检验（H0: 均值=0）                                        | 统计显著性                 |
| IC 衰减序列        | 分别计算 factor_values 与 shift(returns, k) 的 IC，k=1,2,...,20                         | 因子预测力随持仓周期的衰减 |
| 分组单调性检验     | 检查各组年化收益是否单调递减（第1组最高，最后一组最低）                                   | 因子是否具有线性排序能力   |

### 6.3 batch.py

**职责：** 批量评估多个因子，支持并行。

**核心函数 `batch_evaluate(factor_list, stock_data, backend, n_groups=5, n_jobs=4)`：**

**输入：**

- `factor_list`：因子列表，每个元素是一个字符串表达式或已编译的函数
- `n_jobs`：并行进程数

**处理流程：**

1. 编译所有因子表达式（如果是字符串的话）
2. 计算 forward_returns（只算一次，所有因子共用）
3. 用 `multiprocessing.Pool` 或 `concurrent.futures.ProcessPoolExecutor` 并行执行每个因子的回测
4. 收集所有结果，按 IR 降序排列
5. 额外计算因子间相关性矩阵：对每对因子，计算它们因子值的截面相关系数均值，标记高度相关的因子对（可能是重复因子）

**输出：** 一个列表，每个元素是 `(因子表达式, 指标字典)`，按 IR 降序排列。额外输出因子间相关性矩阵。

---

## 七、报告层（report/）

### 7.1 summary.py

**职责：** 将回测结果可视化，生成图表和文字摘要。

**单因子报告，需要输出的内容：**

**文字摘要：** 一行打印所有核心指标（IC均值、IR、夏普、最大回撤、换手率、t检验p值）。

**图表（用 matplotlib，组织成一个 2×2 或 3×2 的子图布局）：**

| 子图位置 | 内容               | 说明                                                    |
| -------- | ------------------ | ------------------------------------------------------- |
| 左上     | 分组累计净值曲线   | N 条线（每组一条），如果因子有效应呈扇形展开             |
| 右上     | 多空组合累计净值   | 一条线，应该稳定向上                                     |
| 左下     | IC 时间序列        | 柱状图或折线图，附 IC 均值的水平参考线                    |
| 右下     | IC 衰减曲线        | x 轴是持仓天数 k，y 轴是对应的 IC 值                     |

**批量因子报告：**

- 所有因子的指标汇总表（按 IR 排序）
- 因子间相关性热力图
- Top N 因子的分组净值曲线

---

## 八、入口（main.py）

**职责：** 串联整个流程，提供命令行接口。

**主流程：**

1. 初始化后端（`auto.get_backend()` → NumPyBackend）
2. 加载数据（`loader.load_stock_data()`），先检查缓存
3. 计算 forward_returns：`shift(returns, -1)`，即每天的收益率向前移一天，使得 T 日的 forward_return 是 T+1 日的实际收益。最后一天的 forward_return 为 NaN
4. 模式分支：
   - 单因子模式：编译因子 → 计算因子值 → 回测 → 出报告
   - 批量模式：读取因子列表 → 批量评估 → 出汇总报告
5. 保存结果

**命令行参数：**

| 参数            | 说明                          | 默认值           |
| --------------- | ----------------------------- | ---------------- |
| `--factor`      | 因子表达式字符串              | 无               |
| `--factors-file`| 因子列表文件路径（每行一个）  | 无               |
| `--start`       | 回测开始日期                  | `2021-01-01`     |
| `--end`         | 回测结束日期                  | `2024-12-31`     |
| `--groups`      | 分组数                        | `5`              |
| `--output`      | 报告输出目录                  | `./output`       |

---

## 九、开发顺序（推荐）

按以下顺序开发，每完成一步都能跑通验证，避免最后集成时出一堆问题。

### Day 1：数据层 + 后端基础

1. 实现 `data/loader.py`：先硬编码 20-30 只股票，用 AKShare 拉 3 年日线数据，转成 ndarray
2. 实现 `data/cache.py`：pickle 存储，第二次运行秒级加载
3. 实现 `backend/numpy_backend.py`：先只实现 `rolling_mean`、`rolling_std`、`rank`、`shift`、`pct_change` 这 5 个核心算子
4. **验证：** 手动算几只股票的 20 日滚动均值，和你的 rolling_mean 输出对比，确认一致

### Day 2：回测引擎 + 指标

1. 实现 `backtest/engine.py`：run_backtest 函数
2. 实现 `backtest/metrics.py`：先实现 IC 均值、IR、夏普、最大回撤这 4 个核心指标
3. 用 `library.py` 中的短期反转因子（`-1 * rolling_sum(returns, 5)`）手动构造因子值矩阵，喂入 engine 跑一遍
4. **验证：** 检查 IC 均值是否为正（A 股短期反转效应），分组净值是否呈扇形。如果 IC 均值是负的或者分组完全无序，排查 shift 方向和 rank 方向

### Day 3：因子编译器 + 报告

1. 实现 `factor/compiler.py`：字符串表达式 → 函数
2. 实现 `factor/library.py`：把经典因子注册进去
3. 实现 `report/summary.py`：matplotlib 子图布局
4. 补全 `backend/numpy_backend.py` 中剩余的算子
5. **验证：** 用字符串表达式 `"rank(rolling_mean(close, 5) / rolling_mean(close, 20))"` 跑通完整链路，从字符串到最终报告图表

### Day 4：批量评估 + 完善

1. 实现 `backtest/batch.py`：并行批量评估
2. 补全 `metrics.py` 中的换手率、IC 衰减、t 检验等指标
3. 完善报告：因子汇总表、相关性热力图
4. 写几个测试用例

### Day 5：接入 LLM Agent 挖因子

1. 用你已有的 LLM Agent 生成一批候选因子表达式
2. 喂入 batch_evaluate 批量跑
3. 整理结果：生成了多少 → 编译通过多少 → IC 显著的有多少 → 最终可能有效的有几个
4. 把完整的漏斗数据和 top 因子的报告截图保存，面试时展示

---

## 十、关键 Checklist（避免踩坑）

- [ ] forward_returns 的 shift 方向：因子值是 T 日的，收益率必须是 T+1 日的。shift(-1) 是向前移一格，即第 t 行放的是 t+1 日的收益率
- [ ] rank 方向：确认高因子值对应高排名。如果因子逻辑是"越小越好"（如波动率），在因子定义时取负号，而不是在 engine 里改排名方向
- [ ] NaN 传播：任何一个算子的输入包含 NaN 时，输出对应位置也必须是 NaN。特别是 rolling 窗口不满时，前 window-1 行必须是 NaN
- [ ] 分组边界：百分位 1.0 的股票应归入最后一组而不是溢出
- [ ] 收益率计算：确认用的是简单收益率 (p1/p0 - 1) 而不是对数收益率，分组收益用等权平均
- [ ] 日期对齐：所有股票必须在相同的日期序列上，缺失日用 NaN 填充
