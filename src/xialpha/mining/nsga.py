"""NSGA-II 多目标选择算法。"""

from __future__ import annotations

from typing import Any

import numpy as np


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """a 支配 b 当且仅当所有目标 a >= b 且至少一个 a > b（maximize）。"""
    all_geq = True
    any_gt = False
    for ai, bi in zip(a, b):
        if ai < bi:
            all_geq = False
            break
        if ai > bi:
            any_gt = True
    return all_geq and any_gt


def fast_non_dominated_sort(
    fitnesses: list[tuple[float, ...]],
) -> list[list[int]]:
    """非支配排序，返回前沿列表（每个前沿是索引列表）。"""
    n = len(fitnesses)
    domination_count = [0] * n
    dominated_sets: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(fitnesses[i], fitnesses[j]):
                dominated_sets[i].append(j)
                domination_count[j] += 1
            elif dominates(fitnesses[j], fitnesses[i]):
                dominated_sets[j].append(i)
                domination_count[i] += 1

    fronts: list[list[int]] = []
    current_front: list[int] = [
        i for i in range(n) if domination_count[i] == 0
    ]

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def crowding_distance(front_fitnesses: list[tuple[float, ...]]) -> list[float]:
    """计算单个前沿内个体的拥挤度距离，边界个体距离为 inf。"""
    n = len(front_fitnesses)
    if n <= 2:
        return [float("inf")] * n

    n_objectives = len(front_fitnesses[0])
    distances = [0.0] * n

    for m in range(n_objectives):
        sorted_idx = sorted(range(n), key=lambda i: front_fitnesses[i][m])
        distances[sorted_idx[0]] = float("inf")
        distances[sorted_idx[-1]] = float("inf")

        f_min = front_fitnesses[sorted_idx[0]][m]
        f_max = front_fitnesses[sorted_idx[-1]][m]
        if f_max == f_min:
            continue

        span = f_max - f_min
        for k in range(1, n - 1):
            prev_val = front_fitnesses[sorted_idx[k - 1]][m]
            next_val = front_fitnesses[sorted_idx[k + 1]][m]
            distances[sorted_idx[k]] += (next_val - prev_val) / span

    return distances


def nsga2_select(
    population: list,
    fitnesses: list[tuple[float, ...]],
    n_select: int,
) -> list:
    """NSGA-II 选择：非支配排序 → 按前沿顺序选择 → 拥挤度距离截断。"""
    fronts = fast_non_dominated_sort(fitnesses)
    selected: list = []

    for front in fronts:
        remaining = n_select - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(population[i] for i in front)
            continue
        front_fit = [fitnesses[i] for i in front]
        dists = crowding_distance(front_fit)
        order = sorted(range(len(front)), key=lambda i: -dists[i])
        selected.extend(population[front[i]] for i in order[:remaining])
        break

    return selected


def tournament_select(
    population: list,
    fitnesses: list[tuple[float, ...]],
    tournament_size: int,
    rng: np.random.Generator,
) -> Any:
    """锦标赛选择：随机选 tournament_size 个个体，返回 Pareto 排名最高的。"""
    fronts = fast_non_dominated_sort(fitnesses)
    rank: dict[int, int] = {}
    crowding: dict[int, float] = {}

    for r, front in enumerate(fronts):
        front_fit = [fitnesses[i] for i in front]
        dists = crowding_distance(front_fit)
        for k, idx in enumerate(front):
            rank[idx] = r
            crowding[idx] = dists[k]

    replace = tournament_size > len(population)
    candidates = rng.choice(len(population), size=tournament_size, replace=replace)
    best = int(candidates[0])
    for c in candidates[1:]:
        c = int(c)
        if rank[c] < rank[best] or (
            rank[c] == rank[best] and crowding[c] > crowding[best]
        ):
            best = c

    return population[best]
