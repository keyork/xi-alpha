"""因子表达式树的表示、随机生成、交叉、变异、序列化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── 算子分类 ────────────────────────────────────────────────────────

DATA_FIELDS: list[str] = [
    "open", "high", "low", "close", "volume", "amount", "returns",
]

UNARY_OPS: list[str] = [
    "rank", "zscore", "demean", "log", "abs", "sign",
]

WINDOW_OPS: list[str] = [
    "rolling_mean", "rolling_std", "rolling_sum", "rolling_max",
    "rolling_min", "shift", "pct_change", "diff",
]

ARITHMETIC_OPS: list[str] = ["+", "-", "*", "/"]

CONSTANTS_STR: list[str] = [
    "-1", "-0.5", "0.5", "1", "2", "3", "5", "10", "20", "60",
]

WINDOW_VALUES: list[int] = [5, 10, 20, 60]

# ── 节点 ────────────────────────────────────────────────────────────


@dataclass
class FactorNode:
    """因子表达式树的节点。"""

    node_type: str          # 'field', 'constant', 'unary', 'window', 'corr', 'arithmetic'
    value: str              # field名/算子名/运算符/常量字符串
    children: list[FactorNode]
    window: int | None = None  # 仅 window 和 corr 类型使用


# ── 序列化 ──────────────────────────────────────────────────────────


def to_expression(node: FactorNode) -> str:
    """将树序列化为可被 AST 编译器解析的字符串。"""
    if node.node_type == "field":
        return node.value
    if node.node_type == "constant":
        return node.value
    if node.node_type == "unary":
        return f"{node.value}({to_expression(node.children[0])})"
    if node.node_type == "window":
        return f"{node.value}({to_expression(node.children[0])}, {node.window})"
    if node.node_type == "corr":
        left = to_expression(node.children[0])
        right = to_expression(node.children[1])
        return f"{node.value}({left}, {right}, {node.window})"
    if node.node_type == "arithmetic":
        left = to_expression(node.children[0])
        right = to_expression(node.children[1])
        return f"({left}) {node.value} ({right})"
    raise ValueError(f"Unknown node type: {node.node_type}")


# ── 树工具 ──────────────────────────────────────────────────────────


def tree_depth(node: FactorNode) -> int:
    """返回树的深度（单节点深度为 1）。"""
    if not node.children:
        return 1
    return 1 + max(tree_depth(c) for c in node.children)


def copy_tree(node: FactorNode) -> FactorNode:
    """深拷贝一棵表达式树。"""
    return FactorNode(
        node_type=node.node_type,
        value=node.value,
        children=[copy_tree(c) for c in node.children],
        window=node.window,
    )


def get_subtree_indices(node: FactorNode) -> list[tuple[int, ...]]:
    """返回所有子树的路径索引（含根节点 ``()``）。"""
    result: list[tuple[int, ...]] = [()]
    for i, child in enumerate(node.children):
        for path in get_subtree_indices(child):
            result.append((i,) + path)
    return result


def _get_subtree(node: FactorNode, path: tuple[int, ...]) -> FactorNode:
    """按路径索引获取子树。"""
    current = node
    for idx in path:
        current = current.children[idx]
    return current


def _set_subtree(
    node: FactorNode, path: tuple[int, ...], new_subtree: FactorNode
) -> FactorNode:
    """就地替换路径处的子树，返回根节点。"""
    if not path:
        return new_subtree
    current = node
    for idx in path[:-1]:
        current = current.children[idx]
    current.children[path[-1]] = new_subtree
    return node


# ── 随机生成 ────────────────────────────────────────────────────────


def _generate_terminal(rng: np.random.Generator) -> FactorNode:
    if rng.random() < 0.85:
        return FactorNode("field", str(rng.choice(DATA_FIELDS)), [])
    return FactorNode("constant", str(rng.choice(CONSTANTS_STR)), [])


def _generate_operator(
    max_depth: int, rng: np.random.Generator, method: str
) -> FactorNode:
    category = str(rng.choice(
        ["unary"] * 3 + ["window"] * 3 + ["arithmetic"] * 2 + ["corr"]
    ))

    if category == "unary":
        return FactorNode(
            "unary",
            str(rng.choice(UNARY_OPS)),
            [generate_tree(max_depth - 1, rng, method)],
        )
    if category == "window":
        return FactorNode(
            "window",
            str(rng.choice(WINDOW_OPS)),
            [generate_tree(max_depth - 1, rng, method)],
            window=int(rng.choice(WINDOW_VALUES)),
        )
    if category == "corr":
        return FactorNode(
            "corr",
            "rolling_corr",
            [
                generate_tree(max_depth - 1, rng, method),
                generate_tree(max_depth - 1, rng, method),
            ],
            window=int(rng.choice(WINDOW_VALUES)),
        )
    return FactorNode(
        "arithmetic",
        str(rng.choice(ARITHMETIC_OPS)),
        [
            generate_tree(max_depth - 1, rng, method),
            generate_tree(max_depth - 1, rng, method),
        ],
    )


def generate_tree(
    max_depth: int, rng: np.random.Generator, method: str = "grow"
) -> FactorNode:
    """随机生成一棵因子表达式树。

    Parameters
    ----------
    max_depth : int
        树的最大深度。
    rng : numpy.random.Generator
        随机数生成器。
    method : str
        ``'grow'`` — 随机选择终端或算子（~0.4 概率选算子）；
        ``'full'`` — 深度未达上限前只选算子。
    """
    if max_depth <= 0:
        return _generate_terminal(rng)

    if method == "grow":
        if rng.random() < 0.4:
            return _generate_operator(max_depth, rng, method)
        return _generate_terminal(rng)

    if method == "full":
        return _generate_operator(max_depth, rng, method)

    raise ValueError(f"Unknown method: {method}")


# ── 交叉 ────────────────────────────────────────────────────────────


def crossover(
    parent1: FactorNode,
    parent2: FactorNode,
    max_depth: int,
    rng: np.random.Generator,
) -> tuple[FactorNode, FactorNode]:
    """随机交换两棵树的子树，返回子代（深拷贝，不修改原始树）。"""
    child1 = copy_tree(parent1)
    child2 = copy_tree(parent2)

    indices1 = get_subtree_indices(child1)
    indices2 = get_subtree_indices(child2)

    path1 = indices1[int(rng.integers(len(indices1)))]
    path2 = indices2[int(rng.integers(len(indices2)))]

    subtree1 = _get_subtree(child1, path1)
    subtree2 = _get_subtree(child2, path2)

    child1 = _set_subtree(child1, path1, copy_tree(subtree2))
    child2 = _set_subtree(child2, path2, copy_tree(subtree1))

    if tree_depth(child1) > max_depth or tree_depth(child2) > max_depth:
        return copy_tree(parent1), copy_tree(parent2)

    return child1, child2


# ── 变异 ────────────────────────────────────────────────────────────


def mutate(
    tree: FactorNode, max_depth: int, rng: np.random.Generator
) -> FactorNode:
    """随机替换一个子树为新随机生成的子树（深拷贝）。"""
    result = copy_tree(tree)
    indices = get_subtree_indices(result)
    path = indices[int(rng.integers(len(indices)))]

    available_depth = max(1, max_depth - len(path))
    new_subtree = generate_tree(available_depth, rng, "grow")

    result = _set_subtree(result, path, new_subtree)

    if tree_depth(result) > max_depth:
        return copy_tree(tree)

    return result
