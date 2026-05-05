"""LLM Agent 因子挖掘器的提示词模板。"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = """\
你是一个专业的量化因子研究员。你的任务是生成有效的 A 股 alpha 因子表达式。

可用数据字段:
- open, high, low, close: 价格（开盘/最高/最低/收盘）
- volume: 成交量
- amount: 成交额
- returns: 日收益率

可用算子:
- 截面: rank(x), zscore(x), demean(x)
- 滚动: rolling_mean(x, n), rolling_std(x, n), rolling_sum(x, n), rolling_max(x, n), rolling_min(x, n)
- 相关: rolling_corr(x, y, n)
- 时序: shift(x, n), pct_change(x, n), diff(x, n)
- 逐元素: log(x), abs(x), sign(x)
- 算术: +, -, *, /

规则:
1. 只使用上述算子和字段，不要发明新算子
2. 滚动窗口 n 建议用 5, 10, 20, 60
3. 表达式必须是合法的 Python 表达式
4. 返回 JSON 格式: {"factors": ["表达式1", "表达式2", ...]}
5. 每次生成互不相同的、有创意的因子"""


def build_initial_prompt(n_factors: int) -> list[dict]:
    user_prompt = (
        f"请生成 {n_factors} 个你认为最可能有预测能力的 A 股因子表达式。返回 JSON 格式。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_iteration_prompt(
    n_factors: int,
    best_factors: list[tuple[str, dict]],
    recent_failures: list[str],
    iteration: int,
    max_iterations: int,
) -> list[dict]:
    context_parts: list[str] = []

    if best_factors:
        lines = ["以下是目前表现最好的因子及其指标："]
        for expr, metrics in best_factors:
            ir = metrics.get("ir", 0.0)
            ic = metrics.get("ic_mean", 0.0)
            sharpe = metrics.get("sharpe", 0.0)
            lines.append(f"  - IR={ir:.4f}  IC={ic:.4f}  Sharpe={sharpe:.4f}  表达式: {expr}")
        context_parts.append("\n".join(lines))

    if recent_failures:
        lines = ["以下表达式评估失败，请避免类似错误："]
        for expr in recent_failures:
            lines.append(f"  - {expr}")
        context_parts.append("\n".join(lines))

    context_parts.append(
        f"当前是第 {iteration}/{max_iterations} 轮迭代。"
        f"请基于以上反馈生成 {n_factors} 个新的因子表达式。"
        "你可以改进已有因子的变体，也可以探索全新的方向。返回 JSON 格式。"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(context_parts)},
    ]


_EXPRESSION_PATTERN = re.compile(
    r'["\']('
    r'(?:rank|zscore|demean|rolling_mean|rolling_std|rolling_sum|rolling_max|rolling_min'
    r'|rolling_corr|shift|pct_change|diff|log|abs|sign'
    r'|open|high|low|close|volume|amount|returns'
    r'|[\d\.\+\-\*/\s\(\),])+'
    r')["\']'
)


def parse_llm_response(content: str) -> list[str]:
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "factors" in data:
            factors = [f for f in data["factors"] if isinstance(f, str) and f.strip()]
            if factors:
                return factors
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    matches = _EXPRESSION_PATTERN.findall(content)
    return [m.strip() for m in matches if m.strip()]
