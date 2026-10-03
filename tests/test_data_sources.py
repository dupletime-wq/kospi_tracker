import numpy as np
import pandas as pd
import pytest

from src import data_sources as ds


def _df(start, n, base=100.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = base * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c}, index=idx)


def test_merge_rescales_snapshot_to_live_basis():
    full = _df("2024-01-01", 400)
    snap = full.iloc[:350]
    live = full.iloc[300:] * 0.97  # 그 사이 배당 등으로 수정주가 기준이 3% 달라짐
    out = ds.merge_history(snap, live)
    assert out.index.is_monotonic_increasing and out.index.is_unique and len(out) == 400
    r = out["Close"].pct_change().dropna()
    assert r.abs().max() < 0.06  # 이음매에 가짜 -3% 수익률이 없다
    pd.testing.assert_series_equal(out["Close"].iloc[-50:], live["Close"].iloc[-50:], check_freq=False)


def test_merge_handles_missing_sides():
    d = _df("2024-01-01", 50)
    assert ds.merge_history(None, d) is d and ds.merge_history(d, None) is d
    assert ds.merge_history(d, d.iloc[:0]) is d


def test_snapshot_roundtrip(tmp_path):
    p = tmp_path / "s.csv.gz"
    a, b = _df("2024-01-01", 30, seed=1), _df("2024-01-01", 30, base=50, seed=2)
    ds.save_snapshot({"A": a, "B": b}, p)
    got = ds.load_snapshot(p)
    assert set(got) == {"A", "B"}
    np.testing.assert_allclose(got["A"]["Close"], a["Close"], rtol=1e-5)
    assert ds.load_snapshot(tmp_path / "missing.csv.gz") == {}


def test_fetch_all_falls_back_to_snapshot_when_live_fails(monkeypatch, tmp_path):
    snap = {k: _df("2022-01-03", 500, seed=i) for i, k in enumerate(
        list(ds.TARGETS) + list(ds.ASSETS) + ["DRAM", "SNDK"])}
    p = tmp_path / "s.csv.gz"
    ds.save_snapshot(snap, p)
    monkeypatch.setattr(ds, "SNAPSHOT_PATH", p)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)

    def boom(*a, **k):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(ds.yf, "Ticker", boom)
    mk = ds.fetch_all()
    assert "KOSPI" in mk.targets and len(mk.targets["KOSPI"]) == 500
    assert set(mk.stale) == set(mk.errors) and "429" in mk.errors["KOSPI"]


def test_fetch_all_without_snapshot_reports_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "SNAPSHOT_PATH", tmp_path / "none.csv.gz")
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    monkeypatch.setattr(ds.yf, "Ticker", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked")))
    mk = ds.fetch_all()
    assert "KOSPI" not in mk.targets and "blocked" in mk.errors["KOSPI"] and not mk.stale


def test_fetch_history_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    good = _df("2024-01-01", 10)
    good.index = good.index.tz_localize("Asia/Seoul")

    class T:
        def history(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("rate limited")
            return good

    monkeypatch.setattr(ds.yf, "Ticker", lambda t: T())
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    assert len(ds.fetch_history("X")) == 10 and calls["n"] == 3
    with pytest.raises(RuntimeError):
        monkeypatch.setattr(ds.yf, "Ticker", lambda t: (_ for _ in ()).throw(RuntimeError("down")))
        ds.fetch_history("X")


def test_drop_incomplete_kr_during_session_only():
    d = _df("2026-09-28", 5)  # 9/28~10/2 (금)
    # 금요일 11:00 KST = 02:00 UTC: 오늘 행은 미완성 → 제거
    assert len(ds.drop_incomplete_kr(d, pd.Timestamp("2026-10-02 02:00"))) == 4
    # 금요일 16:00 KST = 07:00 UTC: 마감 후 → 유지
    assert len(ds.drop_incomplete_kr(d, pd.Timestamp("2026-10-02 07:00"))) == 5
    # 토요일: 마지막 행은 금요일 → 유지
    assert len(ds.drop_incomplete_kr(d, pd.Timestamp("2026-10-03 02:00"))) == 5
