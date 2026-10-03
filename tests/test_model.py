import numpy as np
import pandas as pd

from src.data_sources import parse_night_futures
from src.model import (blend_with_futures, build_dataset, dram_pure, fit_model, fit_staged, gap_beta,
                       next_kr_session, walk_forward)


def _market(n=300, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n)
    ret = rng.normal(0, 1, n)  # 미국 일수익률(%)
    us = pd.DataFrame({"Close": 100 * np.cumprod(1 + ret / 100)}, index=days)
    # T일 갭 = 0.5 * (T 이전 마지막 미국일 수익률) + 노이즈
    gap = np.r_[0, 0.5 * ret[:-1]] + rng.normal(0, 0.1, n)
    close = 1000 * np.cumprod(1 + rng.normal(0, 0.005, n))
    open_ = np.r_[close[0], close[:-1]] * (1 + gap / 100)
    kospi = pd.DataFrame({"Open": open_, "Close": close}, index=days)
    return kospi, {"EWY": us}


def test_alignment_has_no_lookahead():
    kospi, prices = _market()
    data = build_dataset(kospi, prices, ["EWY"])
    # 학습표의 지표 날짜는 항상 T보다 이전이어야 하므로 계수는 약 0.5로 복원된다
    fit = fit_model(data, ["EWY"], alpha=0.01)
    assert abs(fit.coefficients()["EWY"] - 0.5) < 0.05


def test_walk_forward_beats_baseline():
    kospi, prices = _market()
    data = build_dataset(kospi, prices, ["EWY"])
    bt = walk_forward(data, ["EWY"], alpha=1.0, test_days=100)
    assert bt is not None and bt.mae < bt.mae_baseline and bt.hit_rate > 0.8


def test_walk_forward_too_small_returns_none():
    kospi, prices = _market(n=70)
    data = build_dataset(kospi, prices, ["EWY"])
    assert walk_forward(data, ["EWY"], test_days=100) is None


def test_next_session_skips_weekend():
    assert next_kr_session(pd.Timestamp("2026-10-02")) == pd.Timestamp("2026-10-05")  # 금 -> 월
    assert next_kr_session(pd.Timestamp("2026-10-05")) == pd.Timestamp("2026-10-06")


def test_blend():
    assert blend_with_futures(1.0, None, 0.6) == 1.0
    assert abs(blend_with_futures(1.0, 2.0, 0.5) - 1.5) < 1e-9


def test_parse_night_futures():
    assert parse_night_futures("<div>아무것도 없음</div>") is None
    got = parse_night_futures('<span class="night-change">-0.45%</span>')
    assert got is not None and got.change_pct == -0.45


def _ohlc(close):
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close})


def test_dram_pure_removes_peer_effect():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2025-01-01", periods=150)
    mu = rng.normal(0, 2, 150)
    own = rng.normal(0, 1, 150)  # 순수 수급
    dram = 0.8 * mu + own
    mk = lambda r: _ohlc(pd.Series(100 * np.cumprod(1 + r / 100), index=days))
    pure, info = dram_pure({"DRAM": mk(dram), "MU": mk(mu)}, ["MU"])
    assert abs(info["betas"]["MU"] - 0.8) < 0.15
    assert abs(np.corrcoef(pure.to_numpy(), own[1:])[0, 1]) > 0.9


def test_dram_pure_insufficient_data_is_empty():
    days = pd.bdate_range("2025-01-01", periods=20)
    df = _ohlc(pd.Series(np.linspace(100, 110, 20), index=days))
    pure, info = dram_pure({"DRAM": df, "MU": df}, ["MU"])
    assert pure.empty and info == {}


def test_staged_uses_short_history_extra():
    kospi, prices = _market(n=300, seed=3)
    extra = pd.Series(np.random.default_rng(4).normal(0, 1, 300),
                      index=prices["EWY"].index).tail(100)  # 최근 100일만 존재
    data = build_dataset(kospi, prices, ["EWY"], {"X": extra})
    assert data["X"].notna().sum() > 80 and data["X"].isna().sum() > 100
    fit = fit_staged(data, ["EWY"], ["X"], alpha=1.0)
    assert fit.extra is not None and fit.keys == ["EWY", "X"]
    assert fit_staged(data.head(150), ["EWY"], ["X"]).extra is None  # extra 표본 부족 -> 1단계만
    assert np.isfinite(fit.predict(data.tail(5))).all()


def test_gap_beta():
    kospi, _ = _market()
    stock = kospi.copy()
    stock["Open"] = stock["Close"].shift(1) * (1 + 2 * (kospi["Open"] / kospi["Close"].shift(1) - 1))
    assert abs(gap_beta(stock.dropna(), kospi) - 2.0) < 0.1
