import numpy as np
import pytest

from dataclasses import dataclass

from xialpha.factor.compiler import compile_factor
from xialpha.factor.operators import OPERATOR_REGISTRY, DATA_FIELDS
from xialpha.factor.library import get_all_factors
from xialpha import config
from xialpha.backend.numpy_backend import NumPyBackend


@dataclass
class MockStockData:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    returns: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray


def _mock_data(T=200, N=30, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal((T, N)) * 0.5, axis=0)
    close = np.maximum(close, 1.0)
    returns = np.full_like(close, np.nan)
    returns[1:] = close[1:] / close[:-1] - 1.0

    return MockStockData(
        open=close * (1 + rng.standard_normal((T, N)) * 0.005),
        high=close * (1 + np.abs(rng.standard_normal((T, N))) * 0.01),
        low=close * (1 - np.abs(rng.standard_normal((T, N))) * 0.01),
        close=close,
        volume=np.abs(rng.standard_normal((T, N))) * 1e6,
        amount=np.abs(rng.standard_normal((T, N))) * 1e8,
        returns=returns,
        dates=np.arange(T),
        symbols=np.array([f"S{i}" for i in range(N)]),
    )


@pytest.fixture
def backend():
    return NumPyBackend()


@pytest.fixture
def stock_data():
    return _mock_data()


class TestCompiler:
    def test_compile_simple(self, stock_data, backend):
        fn = compile_factor("rank(close)")
        result = fn(stock_data, backend)
        assert result.shape == (200, 30)
        assert np.all(result[~np.isnan(result)] >= 0)
        assert np.all(result[~np.isnan(result)] <= 1)

    def test_compile_complex(self, stock_data, backend):
        fn = compile_factor("rank(rolling_mean(close, 5) / rolling_mean(close, 20))")
        result = fn(stock_data, backend)
        assert result.shape == (200, 30)

    def test_compile_caching(self):
        fn1 = compile_factor("rank(close)")
        fn2 = compile_factor("rank(close)")
        assert fn1 is fn2

    def test_reject_unknown_operator(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            compile_factor("hacker_fn(close)")

    def test_reject_unknown_variable(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            compile_factor("rank(nonexistent)")

    def test_reject_attribute_access(self):
        with pytest.raises(ValueError, match="Forbidden|Disallowed"):
            compile_factor("close.__class__")

    def test_reject_subscript(self):
        with pytest.raises(ValueError, match="Forbidden|Disallowed"):
            compile_factor("close[0]")

    def test_arithmetic_expression(self, stock_data, backend):
        fn = compile_factor("close / rolling_mean(close, 20) - 1")
        result = fn(stock_data, backend)
        assert result.shape == (200, 30)


class TestClassicFactors:
    def test_all_classic_compile(self, stock_data, backend):
        for name, expr in get_all_factors():
            fn = compile_factor(expr)
            result = fn(stock_data, backend)
            assert result.shape == (200, 30), f"Factor '{name}' shape mismatch"

    def test_short_term_reversal(self, stock_data, backend):
        fn = compile_factor(config.CLASSIC_FACTORS["short_term_reversal"])
        result = fn(stock_data, backend)
        assert result.shape == (200, 30)

    def test_volatility(self, stock_data, backend):
        fn = compile_factor(config.CLASSIC_FACTORS["volatility"])
        result = fn(stock_data, backend)
        assert result.shape == (200, 30)


class TestRegistry:
    def test_all_operators_registered(self):
        expected = {
            "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
            "rolling_sum", "rolling_corr", "shift", "pct_change", "diff",
            "rank", "zscore", "demean", "log", "abs", "sign",
        }
        assert expected.issubset(OPERATOR_REGISTRY.keys())

    def test_data_fields(self):
        expected = {"open", "high", "low", "close", "volume", "amount", "returns"}
        assert set(DATA_FIELDS) == expected
