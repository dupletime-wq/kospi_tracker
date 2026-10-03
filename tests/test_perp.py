import numpy as np
import pandas as pd
import pytest

from src import perp

BAR = perp.BAR


def _closes(start, end, f=lambda i: 100.0):
    idx = pd.date_range(start, end, freq="30min", inclusive="left")
    return pd.Series([f(i) for i in range(len(idx))], index=idx, dtype=float)


# ---------- 수집/폴백 ----------
def test_closed_bars_drop_the_open_one():
    s = _closes("2026-10-01 00:00", "2026-10-01 02:00")
    out = perp._closed(s, pd.Timestamp("2026-10-01 01:10"))
    assert out.index[-1] == pd.Timestamp("2026-10-01 00:30")  # 01:00 봉은 01:30 에 닫힌다


def test_exchange_parsers(monkeypatch):
    now = pd.Timestamp("2026-10-01 02:00")
    gate = [{"t": int(pd.Timestamp(t).timestamp()), "c": str(p)} for t, p in
            (("2026-10-01 00:00", 100), ("2026-10-01 00:30", 101), ("2026-10-01 01:30", 103))]
    monkeypatch.setattr(perp, "_get", lambda url, params, timeout=10.0: gate)
    g = perp.fetch_gate("X", pd.Timestamp("2026-10-01"), now)
    assert list(g.values) == [100.0, 101.0, 103.0]

    ms = lambda t: str(int(pd.Timestamp(t).timestamp() * 1000))  # noqa: E731
    monkeypatch.setattr(perp, "_get", lambda url, params, timeout=10.0: {"data": [
        [ms("2026-10-01 00:00"), 0, 0, 0, "100", 0, 0], [ms("2026-10-01 00:30"), 0, 0, 0, "101", 0, 0]]})
    assert list(perp.fetch_bitget("X", now).values) == [100.0, 101.0]
    monkeypatch.setattr(perp, "_get", lambda url, params, timeout=10.0: {"data": [  # OKX: 내림차순 + confirm
        [ms("2026-10-01 01:30"), 0, 0, 0, "103", 0, 0, 0, "0"],
        [ms("2026-10-01 01:00"), 0, 0, 0, "102", 0, 0, 0, "1"],
        [ms("2026-10-01 00:30"), 0, 0, 0, "101", 0, 0, 0, "1"]]})
    o = perp.fetch_okx("X", now)
    assert list(o.values) == [101.0, 102.0] and o.index.is_monotonic_increasing  # 미확정 봉 제외, 오름차순
    monkeypatch.setattr(perp, "_get", lambda url, params, timeout=10.0: {"data": []})
    with pytest.raises(perp.PerpUnavailable):
        perp.fetch_bitget("X", now)


def test_get_series_falls_back_in_order_and_merges_snapshot(monkeypatch):
    now = pd.Timestamp("2026-10-05 02:00")
    snap = {"SEC": _closes("2026-09-01", "2026-10-02")}
    live = _closes("2026-10-01", "2026-10-05 02:00", lambda i: 200.0)

    def bad(*a, **k):
        raise RuntimeError("blocked")

    monkeypatch.setattr(perp, "fetch_gate", bad)
    monkeypatch.setattr(perp, "fetch_bitget", bad)
    monkeypatch.setattr(perp, "fetch_okx", lambda inst, n: live)
    ps = perp.get_series("SEC", now, snapshot=snap)
    assert ps.source == "okx" and ps.live_end == now
    assert ps.close.index.is_unique and ps.close.index.is_monotonic_increasing
    assert ps.close[pd.Timestamp("2026-09-15")] == 100.0 and ps.close.iloc[-1] == 200.0  # 과거=스냅샷, 최근=실시간
    monkeypatch.setattr(perp, "fetch_okx", bad)
    with pytest.raises(perp.PerpUnavailable) as e:
        perp.get_series("SEC", now, snapshot=snap)
    assert "gate" in str(e.value) and "okx" in str(e.value)


def test_snapshot_roundtrip(tmp_path):
    p = tmp_path / "p.csv.gz"
    a, b = _closes("2026-10-01", "2026-10-02", lambda i: 1 + i), _closes("2026-10-01", "2026-10-02", lambda i: 5.0)
    perp.save_snapshot({"SEC": a, "HYNIX": b}, p)
    got = perp.load_snapshot(p)
    pd.testing.assert_series_equal(got["SEC"], a, check_names=False, check_freq=False)
    assert perp.load_snapshot(tmp_path / "none.csv.gz") == {}


# ---------- 시각 정렬 ----------
def test_price_at_uses_bar_ending_at_timestamp():
    s = pd.Series({pd.Timestamp("2026-10-01 06:00"): 10.0, pd.Timestamp("2026-10-01 06:30"): 11.0})
    assert perp.price_at(s, pd.Timestamp("2026-10-01 06:30")) == 10.0  # 06:30 시점 = 06:00 시작 봉의 종가
    assert perp.price_at(s, pd.Timestamp("2026-10-01 07:00")) == 11.0


def test_perp_return_between_kr_close_and_asof():
    f = lambda i: 100.0 + i  # noqa: E731
    s = _closes("2026-10-01 00:00", "2026-10-03 00:00", f)
    prev, asof = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-01 23:30")
    a, b = perp.price_at(s, perp.kr_close_ts(prev)), perp.price_at(s, asof)
    assert perp.perp_return(s, prev, asof) == pytest.approx((b / a - 1) * 100)
    assert np.isnan(perp.perp_return(s.iloc[:5], prev, asof))  # 봉 누락


def test_live_asof_clamps_and_weekend():
    fri, mon = pd.Timestamp("2026-10-02"), pd.Timestamp("2026-10-05")
    assert perp.live_asof(pd.Timestamp("2026-10-02 06:40"), mon, fri) is None  # 마감 직후 봉이 아직 안 닫힘
    asof, el = perp.live_asof(pd.Timestamp("2026-10-03 16:10"), mon, fri)
    assert asof == pd.Timestamp("2026-10-03 16:00") and el == int(9.5 * 60 + 24 * 60)
    asof, el = perp.live_asof(pd.Timestamp("2026-10-05 03:00"), mon, fri)  # 개장 후에 실행해도 개장 30분 전으로 고정
    assert asof == pd.Timestamp("2026-10-04 23:30")


def test_history_uses_same_elapsed_capped_before_open():
    s = _closes("2026-09-28", "2026-10-07", lambda i: 100.0 + 0.1 * i)
    kr = pd.DatetimeIndex(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"])
    h = perp.history(s, kr, elapsed_min=60 * 33)
    # 금→월: 33시간 경과가 개장 30분 전(65.5시간 뒤)보다 이르므로 그대로, 월→화: 개장 30분 전으로 캡
    exp_mon = perp.perp_return(s, pd.Timestamp("2026-10-02"), perp.kr_close_ts("2026-10-02") + pd.Timedelta(hours=33))
    exp_tue = perp.perp_return(s, pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-05 23:30"))
    assert h[pd.Timestamp("2026-10-05")] == pytest.approx(exp_mon) and h[pd.Timestamp("2026-10-06")] == pytest.approx(exp_tue)


# ---------- 2단계 보정 ----------
def _stack_frame(n=90, perp_beta=1.0, seed=0, noise_perp=False):
    rng = np.random.default_rng(seed)
    base, p = rng.normal(0, 1.5, n), rng.normal(0, 2, n)
    y = 0.3 * base + perp_beta * p + rng.normal(0, 0.4, n)
    if noise_perp:
        p = rng.normal(0, 2, n)
    return pd.DataFrame({"date": pd.bdate_range("2026-06-01", periods=n), "y": y, "base": base, "perp": p})


def test_fit_stack_recovers_coefficients_and_is_useful():
    st = perp.fit_stack(_stack_frame())
    assert st.coef[1] == pytest.approx(0.3, abs=0.1) and st.coef[2] == pytest.approx(1.0, abs=0.1)
    assert st.useful and st.metrics["MAE 개선"] > 0.3
    assert st.predict(1.0, 2.0) == pytest.approx(st.coef[0] + st.coef[1] + 2 * st.coef[2])


def test_stack_off_when_signal_is_noise_and_when_too_few_rows():
    st = perp.fit_stack(_stack_frame(noise_perp=True, perp_beta=0.0))
    assert st is not None and not st.useful  # 선물이 정보가 없으면 표본외에서 개선이 없어 자동으로 꺼진다
    assert perp.fit_stack(_stack_frame(n=perp.MIN_STACK_DAYS - 1)) is None


def test_stack_oos_has_no_lookahead():
    df = _stack_frame()
    a = perp.fit_stack(df).frame["oos"].to_numpy()
    df2 = df.copy()
    df2.loc[60:, "y"] += 50.0
    b = perp.fit_stack(df2).frame["oos"].to_numpy()
    np.testing.assert_allclose(a[:60], b[:60])  # 60행 이전 표본외 예측은 이후 정답에 영향받지 않는다


def test_normal_interval():
    lo, hi, pu = perp.normal_interval(0.0, 2.0)
    assert lo == pytest.approx(-hi) and pu == pytest.approx(0.5)
    assert perp.normal_interval(10.0, 1.0)[2] > 0.99 and perp.normal_interval(1.0, 0.0)[2] == 1.0


def test_build_correction_end_to_end():
    rng = np.random.default_rng(3)
    kr = pd.bdate_range("2026-06-01", "2026-10-02")
    now = pd.Timestamp("2026-10-03 16:10")
    idx = pd.date_range("2026-05-25", "2026-10-03 16:00", freq="30min")
    series, gaps = {}, {}
    for k, vol in (("SEC", 1.0), ("HYNIX", 1.3)):
        g = pd.Series(rng.normal(0, 2, len(kr)), index=kr)  # 날짜별 실제 갭
        px = pd.Series(100.0, index=idx)
        for i in range(1, len(kr)):  # 한국 마감 → 다음날 개장 직전까지 선물이 갭을 (노이즈와 함께) 선반영
            T, P = kr[i], kr[i - 1]
            seg = (idx > perp.kr_close_ts(P)) & (idx <= perp.kr_open_ts(T) - BAR)
            px[seg] = 100.0 * (1 + (g[T] + rng.normal(0, 0.3)) / 100 * np.linspace(0, 1, seg.sum()))
        series[k] = perp.PerpSeries(px, "gate", idx[-1] + BAR)
        gaps[k] = g * vol
    targets = {"KOSPI": pd.DataFrame({"Close": 1.0}, index=kr)}
    bp = {k: pd.DataFrame({"date": kr[1:], "y": gaps.get(k, (gaps["SEC"] + gaps["HYNIX"]) / 2)[kr[1:]].to_numpy(),
                           "pred": rng.normal(0, 1, len(kr) - 1)}) for k in ("SEC", "HYNIX", "KOSPI")}
    c = perp.build_correction(bp, targets, now, series)
    assert c.asof == pd.Timestamp("2026-10-03 16:00") and c.source == "gate"
    assert c.live["KOSPI"] == pytest.approx((c.live["SEC"] + c.live["HYNIX"]) / 2)
    assert set(c.stacks) == {"SEC", "HYNIX", "KOSPI"} and c.stacks["SEC"] is not None


def test_build_correction_raises_when_unavailable():
    kr = pd.bdate_range("2026-06-01", "2026-10-02")
    with pytest.raises(perp.PerpUnavailable):
        perp.build_correction({}, {"KOSPI": pd.DataFrame({"Close": 1.0}, index=kr)}, pd.Timestamp("2026-10-02 06:40"),
                              {k: perp.PerpSeries(pd.Series(dtype=float), "gate", pd.Timestamp("2026-10-02")) for k in perp.SYMBOLS})
