"""将因子表达式字符串编译为可调用函数 (stock_data, backend) -> ndarray。"""

from __future__ import annotations

import ast
from typing import Callable

import numpy as np

from .operators import OPERATOR_REGISTRY, DATA_FIELDS

# ── AST 节点白名单 / 黑名单 ───────────────────────────────────────

_FORBIDDEN_NODE_TYPES = frozenset({
    ast.Import,
    ast.ImportFrom,
    ast.Attribute,
    ast.Subscript,
    ast.Assign,
    ast.Global,
    ast.Nonlocal,
    ast.Lambda,
    ast.ListComp,
    ast.DictComp,
    ast.Try,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Yield,
    ast.Await,
    ast.Delete,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Assert,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Starred,
    ast.GeneratorExp,
})

_ALLOWED_NODE_TYPES = frozenset({
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    # 二元运算符类型
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    # 一元运算符类型
    ast.USub,
    ast.UAdd,
})

# ── 校验 ───────────────────────────────────────────────────────────


def _validate_ast(tree: ast.Expression) -> None:
    call_func_ids: set[int] = set()

    # 第一遍：类型白名单 + 禁止检查 + Call 校验
    for node in ast.walk(tree):
        node_type = type(node)

        if node_type in _FORBIDDEN_NODE_TYPES:
            raise ValueError(f"Forbidden construct: {node_type.__name__}")

        if node_type not in _ALLOWED_NODE_TYPES:
            raise ValueError(f"Disallowed AST node: {node_type.__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named function calls are allowed")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed in operator calls")
            call_func_ids.add(id(node.func))
            op_name = node.func.id
            if op_name not in OPERATOR_REGISTRY:
                valid = sorted(OPERATOR_REGISTRY.keys())
                raise ValueError(
                    f"Unknown operator: '{op_name}'. Valid operators: {valid}"
                )

    # 第二遍：非 Call.func 的 Name 节点必须是数据字段
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in call_func_ids:
            if node.id not in DATA_FIELDS:
                raise ValueError(
                    f"Unknown variable: '{node.id}'. Valid fields: {DATA_FIELDS}"
                )


# ── 公共 API ───────────────────────────────────────────────────────

_compile_cache: dict[str, Callable] = {}


def compile_factor(expr: str) -> Callable:
    """将表达式编译为可调用函数 ``(stock_data, backend) -> np.ndarray``。"""
    if expr in _compile_cache:
        return _compile_cache[expr]

    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree)
    code = compile(tree, "<factor>", "eval")

    def _compiled(stock_data, backend) -> np.ndarray:
        ns: dict = {}
        # 通过闭包捕获 backend 的算子包装
        for name, op_func in OPERATOR_REGISTRY.items():
            ns[name] = lambda *a, _fn=op_func, _be=backend: _fn(*a, backend=_be)
        # 数据字段查找
        for field in DATA_FIELDS:
            ns[field] = getattr(stock_data, field)
        return eval(code, {"__builtins__": {}}, ns)  # noqa: S307

    _compile_cache[expr] = _compiled
    return _compiled
