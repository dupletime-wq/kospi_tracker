import numpy as np
import pandas as pd
import pytest

from src import features as F


def _ohlc(days, close, open_=None):
    open_ = close if open_ is None else open_
    return pd.DataFrame({"Open": open_, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=days)


@pytest.fixture
def market():
    rng = np.random.default_rng(0)
    kr_days = pd.bdate_range("2024-01-02", periods=120)
    us_days = kr_days  # 같은 평일 캘린더(휴장 차이는 merge_asof가 처리)
    kclose = pd.Series(1000 * np.cumprod(1 + rng.normal(0, 0.005, 120)), index=kr_days)
    kopen = kclose.shift(1).fillna(kclose.iloc[0]) * (1 + rng.normal(0, 0.003, 120))
    targets = {"KOSPI": _ohlc(kr_days, kclose, kopen), "SEC": _ohlc(kr_days, kclose * 50, kopen * 50)}
    ewy = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.01, 120)), index=us_days)
    krw = pd.Series(1300 * np.cumprod(1 + rng.normal(0, 0.003, 120)), index=us_days)
    ewy_open = ewy.shift(1).fillna(ewy.iloc[0]) * (1 + rng.normal(0, 0.003, 120))
    prices = {"EWY": _ohlc(us_days, ewy, ewy_open), "USDKRW": _ohlc(us_days, krw)}
    return targets, prices


def test_cross_features_use_strictly_earlier_asset_date(market):
    targets, prices = market
    fr = F.build_frame("KOSPI", targets, prices, ["EWY", "USDKRW"])
    ewy_ret = F.cc_ret(prices["EWY"])
    for i in (10, 50, 100):
        T = fr["date"].iloc[i]
        prev = ewy_ret[ewy_ret.index < T].iloc[-1]  # T 이전 가장 최근 거래일
        assert fr["EWY"].iloc[i] == pytest.approx(prev)
        assert fr["EWY"].iloc[i] != pytest.approx(ewy_ret.loc[T])  # 당일 값(미래)이 아니다


def test_own_features_are_lagged_one_session(market):
    targets, prices = market
    fr = F.build_frame("SEC", targets, prices, ["EWY"])
    sec_ret = F.cc_ret(targets["SEC"])
    for i in (25, 60):
        assert fr["own_prev_ret"].iloc[i] == pytest.approx(sec_ret.iloc[i - 1])
    kospi_ret = F.cc_ret(targets["KOSPI"])
    assert fr["kospi_prev_ret"].iloc[40] == pytest.approx(kospi_ret.iloc[39])


def test_target_definition(market):
    targets, prices = market
    fr = F.build_frame("KOSPI", targets, prices, ["EWY"])
    k = targets["KOSPI"]
    i = 30
    assert fr["y"].iloc[i] == pytest.approx((k["Open"].iloc[i] / k["Close"].iloc[i - 1] - 1) * 100)


def test_future_row_has_no_target_and_uses_last_session(market):
    targets, prices = market
    fr = F.build_frame("SEC", targets, prices, ["EWY", "USDKRW"], future=True)
    hist = F.build_frame("SEC", targets, prices, ["EWY", "USDKRW"])
    assert len(fr) == len(hist) + 1 and np.isnan(fr["y"].iloc[-1])
    assert fr["date"].iloc[-1] > targets["SEC"].index[-1] and fr["date"].iloc[-1].weekday() < 5
    assert fr["own_prev_ret"].iloc[-1] == pytest.approx(F.cc_ret(targets["SEC"]).iloc[-1])
    assert fr["EWY"].iloc[-1] == pytest.approx(F.cc_ret(prices["EWY"]).iloc[-1])
    # 과거 행은 future 여부와 무관하게 동일
    pd.testing.assert_frame_equal(fr.iloc[:-1].reset_index(drop=True), hist, check_dtype=False)


def test_engineered_features(market):
    targets, prices = market
    fr = F.build_frame("KOSPI", targets, prices, ["EWY", "USDKRW"])
    row = fr.iloc[50]
    assert row["EWY_local"] == pytest.approx(row["EWY"] + row["USDKRW"])
    assert row["EWY_excess"] == pytest.approx(row["EWY_local"] - row["kospi_prev_ret"])


def test_intraday_decomposition(market):
    targets, prices = market
    add = F.build_frame("KOSPI", targets, prices, ["EWY"], intraday="add")
    rep = F.build_frame("KOSPI", targets, prices, ["EWY"], intraday="replace")
    assert {"EWY", "EWY_id", "EWY_on"} <= set(add.columns)
    assert "EWY" not in rep.columns and {"EWY_id", "EWY_on"} <= set(rep.columns)
    row = add.iloc[60]  # 일간 ≈ 오버나이트 + 세션 내
    assert row["EWY"] == pytest.approx(row["EWY_on"] + row["EWY_id"], abs=0.1)


def test_calendar_features(market):
    targets, prices = market
    fr = F.build_frame("KOSPI", targets, prices, ["EWY"])
    mon = fr[fr["date"].dt.weekday == 0].iloc[2]
    assert mon["is_mon"] == 1 and mon["gap_days"] == 3
