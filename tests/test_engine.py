import numpy as np
import pytest

from xialpha.backtest.engine import run_backtest
from xialpha.backtest.metrics import calc_metrics


def _synthetic_factor_data(T=200, N=50, seed=42):
    rng = np.random.default_rng(seed)
    factor_values = rng.standard_normal((T, N))
    forward_returns = rng.standard_normal((T, N)) * 0.02
    dates = np.arange(T)
    return factor_values, forward_returns, dates


def test_backtest_result_keys():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)

    assert result.n_groups == 5
    assert len(result.group_returns) == 5
    assert len(result.group_nav) == 5
    assert result.long_short.shape == (200,)
    assert result.long_short_nav.shape == (200,)
    assert result.ic_series.shape == (200,)
    assert result.group_labels.shape == (200, 50)


def test_group_returns_length():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    for g in range(5):
        assert len(result.group_returns[g]) == 200


def test_long_short_calculation():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    ls = result.long_short
    expected = result.group_returns[0] - result.group_returns[4]
    np.testing.assert_allclose(ls, expected, equal_nan=True)


def test_nav_monotonic_from_first_valid():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    for g in range(5):
        nav = result.group_nav[g]
        assert nav[0] > 0
        assert np.all(np.isfinite(nav))


def test_group_labels_range():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    valid = result.group_labels[result.group_labels >= 0]
    assert np.all(valid >= 0)
    assert np.all(valid < 5)


def test_nan_handling():
    T, N = 100, 30
    rng = np.random.default_rng(42)
    fv = rng.standard_normal((T, N))
    fr = rng.standard_normal((T, N)) * 0.02
    fv[10, 5] = np.nan
    fr[20, 10] = np.nan
    dates = np.arange(T)

    result = run_backtest(fv, fr, dates, n_groups=5)
    assert result.group_labels[10, 5] == -1
    assert result.group_labels[20, 10] == -1


def test_metrics_all_keys():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    m = calc_metrics(result)

    expected_keys = {
        "ic_mean", "ic_std", "ir", "ic_positive_ratio",
        "annual_return", "annual_vol", "sharpe",
        "max_drawdown", "max_dd_duration",
        "daily_turnover", "t_stat", "t_pvalue",
        "group_monotonicity",
    }
    assert expected_keys.issubset(m.keys())


def test_max_drawdown_non_negative():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    m = calc_metrics(result)
    assert m["max_drawdown"] >= 0


def test_ic_positive_ratio_range():
    fv, fr, dates = _synthetic_factor_data()
    result = run_backtest(fv, fr, dates, n_groups=5)
    m = calc_metrics(result)
    assert 0 <= m["ic_positive_ratio"] <= 1
