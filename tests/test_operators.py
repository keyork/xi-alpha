import numpy as np
import pytest

from xialpha.backend.numpy_backend import NumPyBackend


@pytest.fixture
def be():
    return NumPyBackend()


def _make_panel(T=100, N=30, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N))


def test_rolling_mean_shape(be):
    x = _make_panel()
    out = be.rolling_mean(x, 20)
    assert out.shape == x.shape
    assert np.all(np.isnan(out[:19]))
    assert not np.all(np.isnan(out[20:]))


def test_rolling_std_positive(be):
    x = _make_panel()
    out = be.rolling_std(x, 20)
    valid = out[20:]
    assert np.all(valid[~np.isnan(valid)] >= 0)


def test_shift_forward(be):
    x = _make_panel()
    out = be.shift(x, 1)
    assert np.all(np.isnan(out[0]))
    np.testing.assert_allclose(out[1:], x[:-1])


def test_shift_backward(be):
    x = _make_panel()
    out = be.shift(x, -1)
    assert np.all(np.isnan(out[-1]))
    np.testing.assert_allclose(out[:-1], x[1:])


def test_rank_range(be):
    x = _make_panel()
    out = be.rank(x)
    valid = out[~np.isnan(out)]
    assert np.all(valid >= 0)
    assert np.all(valid <= 1)


def test_rank_nan_propagation(be):
    x = _make_panel()
    x[0, 0] = np.nan
    out = be.rank(x)
    assert np.isnan(out[0, 0])


def test_div_zero_returns_nan(be):
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    out = be.div(x, y)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[0, 2])
    assert np.isnan(out[1, 1])
    np.testing.assert_allclose(out[0, 1], 2.0)
    np.testing.assert_allclose(out[1, 0], 2.0)


def test_log_negative_is_nan(be):
    x = np.array([[1.0, -1.0, np.e], [0.5, 0.0, 2.0]])
    out = be.log(x)
    assert np.isnan(out[0, 1])
    assert np.isnan(out[1, 1])
    np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(out[0, 2], 1.0, atol=1e-10)


def test_sign_nan(be):
    x = np.array([[1.0, -2.0, 0.0, np.nan]])
    out = be.sign(x)
    np.testing.assert_array_equal(out[0, :3], [1.0, -1.0, 0.0])
    assert np.isnan(out[0, 3])


def test_zscore(be):
    x = _make_panel(T=50, N=20)
    out = be.zscore(x)
    for t in range(50):
        row = out[t]
        valid = row[~np.isnan(row)]
        if len(valid) > 1:
            np.testing.assert_allclose(np.mean(valid), 0.0, atol=1e-10)


def test_rolling_corr_range(be):
    x = _make_panel()
    y = _make_panel(seed=99)
    out = be.rolling_corr(x, y, 20)
    assert out.shape == x.shape
    valid = out[20:]
    valid = valid[~np.isnan(valid)]
    assert np.all(np.abs(valid) <= 1.01)


def test_cross_corr_range(be):
    x = _make_panel(T=50)
    y = _make_panel(T=50, seed=99)
    out = be.cross_corr(x, y)
    assert out.shape == (50,)
    valid = out[~np.isnan(out)]
    assert np.all(np.abs(valid) <= 1.01)


def test_cumsum(be):
    x = np.ones((5, 3))
    out = be.cumsum(x)
    for j in range(3):
        np.testing.assert_array_equal(out[:, j], [1, 2, 3, 4, 5])
