"""LLM Agent 因子挖掘器。"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .. import config
from .base import BaseMiner
from .prompts import build_initial_prompt, build_iteration_prompt, parse_llm_response

logger = logging.getLogger(__name__)

_BEST_FACTOR_CAPACITY = 20
_FAILURE_CAPACITY = 10
_STAGNATION_LIMIT = 3


class LLMMiner(BaseMiner):

    def __init__(self, stock_data: Any, backend: Any) -> None:
        super().__init__(stock_data, backend)
        self._best_factors: list[tuple[str, dict]] = []
        self._recent_failures: list[str] = []
        self._all_results: list[tuple[str, dict | None]] = []

    def _call_llm(self, messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json"}
        if config.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"

        payload = {
            "model": config.LLM_MODEL_NAME,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }

        try:
            response = httpx.post(
                config.LLM_API_URL,
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.warning("LLM API call failed", exc_info=True)
            return ""

    def _update_state(
        self, results: list[tuple[str, dict | None]]
    ) -> float | None:
        """更新 best_factors 和 recent_failures，返回本轮最佳 IR 或 None。"""
        iteration_best_ir: float | None = None

        for expr, metrics in results:
            self._all_results.append((expr, metrics))
            if metrics is not None:
                self._best_factors.append((expr, metrics))
                ir = metrics["ir"]
                if iteration_best_ir is None or ir > iteration_best_ir:
                    iteration_best_ir = ir
            else:
                self._recent_failures.append(expr)

        self._best_factors.sort(key=lambda x: x[1]["ir"], reverse=True)
        self._best_factors = self._best_factors[:_BEST_FACTOR_CAPACITY]
        self._recent_failures = self._recent_failures[-_FAILURE_CAPACITY:]

        return iteration_best_ir

    def mine(self) -> list[tuple[str, dict]]:
        max_iter = config.LLM_MAX_ITERATIONS
        n_factors = config.LLM_FACTORS_PER_ITER
        stagnation = 0
        global_best_ir = float("-inf")

        for iteration in range(1, max_iter + 1):
            if iteration == 1:
                messages = build_initial_prompt(n_factors)
            else:
                messages = build_iteration_prompt(
                    n_factors,
                    self._best_factors[:5],
                    self._recent_failures[-5:],
                    iteration,
                    max_iter,
                )

            content = self._call_llm(messages)
            if not content:
                logger.warning(
                    "iteration=%d/%d  LLM returned empty response, skipping",
                    iteration, max_iter,
                )
                stagnation += 1
                if stagnation >= _STAGNATION_LIMIT:
                    logger.info("Stopping: %d consecutive iterations with no improvement", stagnation)
                    break
                continue

            expressions = parse_llm_response(content)
            if not expressions:
                logger.warning(
                    "iteration=%d/%d  no expressions parsed from LLM response",
                    iteration, max_iter,
                )
                stagnation += 1
                if stagnation >= _STAGNATION_LIMIT:
                    logger.info("Stopping: %d consecutive iterations with no improvement", stagnation)
                    break
                continue

            results = self._evaluate_batch(expressions)
            iteration_best_ir = self._update_state(results)

            n_success = sum(1 for _, m in results if m is not None)
            n_fail = len(results) - n_success
            best_expr = self._best_factors[0][0] if self._best_factors else "N/A"
            display_ir = self._best_factors[0][1]["ir"] if self._best_factors else float("nan")

            logger.info(
                "LLM iteration=%d/%d  evaluated=%d  success=%d  fail=%d  best=%s  best_ir=%.4f",
                iteration, max_iter, len(results), n_success, n_fail,
                best_expr, display_ir,
            )

            if iteration_best_ir is not None and iteration_best_ir > global_best_ir:
                global_best_ir = iteration_best_ir
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= _STAGNATION_LIMIT:
                logger.info(
                    "Stopping: %d consecutive iterations with no improvement", stagnation,
                )
                break

        return self._best_factors
