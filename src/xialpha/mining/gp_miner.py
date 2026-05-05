"""GP + NSGA-II 因子挖掘器。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .. import config
from .base import BaseMiner
from .nsga import nsga2_select
from .tree import (
    FactorNode,
    copy_tree,
    crossover,
    generate_tree,
    mutate,
    to_expression,
    tree_depth,
)

logger = logging.getLogger(__name__)


def _extract_objective(metrics: dict | None, obj_name: str) -> float:
    """从指标字典中提取目标值。

    以 ``-`` 开头的目标名表示取反（用于需要最小化的指标）。
    """
    if metrics is None:
        return 0.0
    if obj_name.startswith("-"):
        return -metrics.get(obj_name[1:], 0.0)
    return metrics.get(obj_name, 0.0)


class GPMiner(BaseMiner):

    def __init__(self, stock_data: Any, backend: Any) -> None:
        super().__init__(stock_data, backend)
        self._rng = np.random.default_rng(config.GP_SEED)
        self._history: list[dict] = []

    def _compute_fitness(self, metrics: dict | None) -> tuple[float, ...]:
        return tuple(
            _extract_objective(metrics, obj) for obj in config.GP_OBJECTIVES
        )

    def mine(self) -> list[tuple[str, dict]]:
        pop_size = config.GP_POPULATION_SIZE
        n_gen = config.GP_N_GENERATIONS
        max_depth = config.GP_MAX_DEPTH
        cx_prob = config.GP_CROSSOVER_PROB
        mut_prob = config.GP_MUTATION_PROB
        n_elites = config.GP_ELITES
        rng = self._rng

        # ── 1. 初始化种群 ─────────────────────────────────────────────
        population: list[FactorNode] = [
            generate_tree(max_depth, rng, method="grow")
            for _ in range(pop_size)
        ]

        # ── 2. 评估初始种群 ───────────────────────────────────────────
        expr_map: dict[str, FactorNode] = {}
        metrics_map: dict[str, dict | None] = {}

        for tree in population:
            expr = to_expression(tree)
            expr_map[expr] = tree
            if expr not in metrics_map:
                metrics_map[expr] = self._evaluate_factor(expr)

        # ── 3. 进化主循环 ─────────────────────────────────────────────
        for gen in range(1, n_gen + 1):
            current_exprs = [to_expression(t) for t in population]

            fitnesses = [
                self._compute_fitness(metrics_map.get(expr))
                for expr in current_exprs
            ]

            valid = []
            for expr in set(current_exprs):
                m = metrics_map.get(expr)
                if m is not None:
                    valid.append((expr, m))
            best_ir = max((m["ir"] for _, m in valid), default=float("nan"))

            logger.info(
                "GP generation=%d/%d  population=%d  valid=%d  best_ir=%.4f",
                gen, n_gen, len(population), len(valid), best_ir,
            )

            self._history.append({
                "generation": gen,
                "population_size": len(population),
                "valid_count": len(valid),
                "best_ir": best_ir,
            })

            # ── NSGA-II 选择 ──────────────────────────────────────────
            selected_indices = nsga2_select(
                list(range(len(population))), fitnesses, pop_size - n_elites
            )
            selected = [population[i] for i in selected_indices]

            # ── 精英保留 ──────────────────────────────────────────────
            if valid:
                elite_valid = sorted(
                    valid, key=lambda x: x[1]["ir"], reverse=True  # type: ignore[index]
                )[:n_elites]
                elites = [copy_tree(expr_map[e]) for e, _ in elite_valid]
            else:
                elites = [
                    generate_tree(max_depth, rng, method="grow")
                    for _ in range(n_elites)
                ]

            # ── 交叉 + 变异生成子代 ──────────────────────────────────
            offspring: list[FactorNode] = []
            while len(offspring) < pop_size - n_elites:
                idx = rng.integers(0, len(selected))
                parent = selected[idx]
                roll = rng.random()

                if roll < cx_prob and len(selected) > 1:
                    idx2 = idx
                    while idx2 == idx and len(selected) > 1:
                        idx2 = rng.integers(0, len(selected))
                    child, _ = crossover(
                        copy_tree(parent), copy_tree(selected[idx2]),
                        max_depth, rng,
                    )
                elif roll < cx_prob + mut_prob:
                    child = mutate(copy_tree(parent), max_depth, rng)
                else:
                    child = copy_tree(parent)

                if tree_depth(child) > max_depth:
                    child = generate_tree(max_depth, rng, method="grow")

                offspring.append(child)

            # ── 组成新一代种群 ────────────────────────────────────────
            population = elites + offspring[: pop_size - n_elites]

            # ── 评估新个体 ────────────────────────────────────────────
            for tree in population:
                expr = to_expression(tree)
                expr_map[expr] = tree
                if expr not in metrics_map:
                    metrics_map[expr] = self._evaluate_factor(expr)

        # ── 4. 汇总结果 ───────────────────────────────────────────────
        all_valid = [
            (expr, m) for expr, m in metrics_map.items() if m is not None
        ]
        all_valid.sort(key=lambda x: x[1]["ir"], reverse=True)
        return all_valid
