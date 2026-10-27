import numpy as np

from resonet_sync import sanitize as S


def test_nonfinite_and_fill():
    X = np.ones((3, 100)); X[1, 10] = np.nan; X[2, 5:8] = np.inf
    r = S.nonfinite_report(X)
    assert r["n_nonfinite"] == 4 and r["channels"] == [1, 2]
    assert np.isfinite(S.fill_nonfinite(X)).all()


def test_flat_channels():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 2000)); X[2] = 3.0
    X[3, 500:1200] = X[3, 500]              # 700-sample frozen stretch
    assert S.flat_channels(X) == [2]
    assert S.flat_channels(X, fs=500, max_const_run_s=1.0) == [2, 3]


def test_clipping():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000) * 8000
    xc = np.clip(x, -32768, 32767 // 4)
    r = S.clipping_report(np.stack([x, xc]), min_run=3)
    assert r["flagged"] == [1]
    xi = np.clip(x * 5, -32768, 32767)
    assert S.clipping_report(xi, rail=32767)["flagged"] == [0]


def test_line_noise_50_vs_60():
    fs = 500.0
    t = np.arange(int(60 * fs)) / fs
    rng = np.random.default_rng(0)
    for mains in (50.0, 60.0):
        x = rng.standard_normal(t.size) + 0.5 * np.sin(2 * np.pi * mains * t)
        r = S.line_noise(x, fs)
        assert r["dominant"] == mains, r
        assert r["prominence_db"][str(mains)] > 10


def test_verify_fs():
    assert S.verify_fs(1250, n_samples=125000, duration_s=100.0)["ok"]
    assert not S.verify_fs(1250, n_samples=125000, duration_s=62.5)["ok"]
    assert not S.verify_fs(20000, expected_fs=32552)["ok"]


def test_line_noise_fs_check_detects_wrong_fs():
    true_fs = 1000.0
    t = np.arange(int(60 * true_fs)) / true_fs
    x = np.random.default_rng(0).standard_normal(t.size) * 0.1 + np.sin(2 * np.pi * 50 * t)
    r = S.line_noise_fs_check(x, declared_fs=1020.0, mains=(50.0,))
    assert abs(r["50.0"]["implied_fs"] - 1000.0) < 10
