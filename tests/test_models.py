import numpy as np
import pandas as pd
import pytest

from src import features as F
from src import models as M
from src import pipeline as P


def _frame(n=700, seed=0, k=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, k))
    y = X[:, 0] * 0.8 - X[:, 1] * 0.3 + rng.normal(0, 0.3, n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
    df.insert(0, "y", y)
    df.insert(0, "date", pd.bdate_range("2020-01-01", periods=n))
    return df, [f"f{i}" for i in range(k)]


@pytest.mark.parametrize("name", ["baseline", "ridge", "lasso", "enet", "huber", "rf", "hgb", "lgbm", "hybrid"])
def test_every_model_fits_and_learns(name):
    df, feats = _frame(500)
    fit = M.fit_all(df.iloc[:400], feats, [name])
    pred = fit.predict(name, df.iloc[400:])
    assert pred.shape == (100,) and np.isfinite(pred).all()
    if name != "baseline":
        assert np.corrcoef(pred, df["y"].iloc[400:])[0, 1] > 0.8


def test_walk_forward_shape_and_skill():
    df, feats = _frame()
    bt = M.walk_forward(df, feats, ["baseline", "ridge", "hybrid"], test_rows=200, min_train=300, refit_every=50)
    assert len(bt) == 200 and not bt.isna().any().any()
    assert M.metrics(bt, "ridge")["MAE 개선율"] > 0.5 and M.metrics(bt, "ridge")["t(vs 기준선)"] > 5


def test_walk_forward_has_no_lookahead():
    df, feats = _frame()
    r0 = 560
    bt1 = M.walk_forward(df, feats, ["ridge", "lgbm"], test_rows=200, min_train=300, refit_every=25)
    df2 = df.copy()
    df2.loc[r0:, "y"] = df2.loc[r0:, "y"] + 100.0  # 미래 정답을 크게 오염
    bt2 = M.walk_forward(df2, feats, ["ridge", "lgbm"], test_rows=200, min_train=300, refit_every=25)
    before = bt1["date"] < df["date"].iloc[r0]
    for m in ("ridge", "lgbm"):
        np.testing.assert_allclose(bt1.loc[before, m], bt2.loc[before, m])


def test_all_nan_feature_is_dropped_not_crash():
    df, feats = _frame(400)
    df["late"] = np.nan
    df.loc[380:, "late"] = 1.0
    fit = M.fit_all(df.iloc[:300], feats + ["late"], ["ridge", "hgb"])
    assert "late" not in fit.features and np.isfinite(fit.predict("hgb", df.iloc[300:])).all()


def test_target_winsorize_and_weights():
    y = np.r_[np.random.default_rng(0).normal(0, 1, 998), 50.0, -50.0]
    c = M.prepare_target(y, 0.005)
    assert c.max() < 10 and c.min() > -10 and (M.prepare_target(y, 0) == y).all()
    w = M.recency_weights(100, 25)
    assert w[-1] == 1.0 and w[-26] == pytest.approx(0.5) and M.recency_weights(10, None) is None


def test_hac_t_sign():
    rng = np.random.default_rng(1)
    assert M._hac_t(rng.normal(0.5, 1, 300)) > 3
    assert M._hac_t(rng.normal(-0.5, 1, 300)) < -3


def test_vol_weights_downweight_volatile_days():
    df = pd.DataFrame({"kospi_vol20": [1.0] * 50 + [4.0] * 50})
    w = M.vol_weights(df)
    assert w[:50].mean() > 10 * w[50:].mean() and w.mean() == pytest.approx(1.0)


def _synthetic_market(n=700):
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2021-01-01", periods=n)

    def asset(vol, drift=0.0):
        r = rng.normal(drift, vol, n)
        c = pd.Series(100 * np.cumprod(1 + r), index=days)
        o = c.shift(1).fillna(c.iloc[0]) * (1 + rng.normal(0, vol / 3, n))
        return pd.DataFrame({"Open": o, "High": c, "Low": c, "Close": c}, index=days)

    prices = {k: asset(0.01) for k in ["EWY", "FLXK", "USDKRW", "SOXX", "SMH", "MU", "SMSN"]}
    # 타깃: 이전 날 EWY 세션내 수익률에 반응하는 시가 갭
    e = prices["EWY"]
    e_id = ((e["Close"] / e["Open"] - 1) * 100).shift(1).fillna(0)
    kclose = pd.Series(1000 * np.cumprod(1 + rng.normal(0, 0.005, n)), index=days)
    gap = 0.4 * e_id.to_numpy() + rng.normal(0, 0.2, n)
    kopen = kclose.shift(1).fillna(kclose.iloc[0]) * (1 + gap / 100)
    kospi = pd.DataFrame({"Open": kopen, "High": kclose, "Low": kclose, "Close": kclose}, index=days)
    targets = {"KOSPI": kospi, "SEC": kospi * 50}
    return targets, prices


@pytest.fixture(scope="module")
def analysis():
    targets, prices = _synthetic_market()
    cfg = P.Config(models=("baseline", "ridge", "hybrid", "ens"), test_rows=150, refit_every=50)
    return P.analyze("KOSPI", targets, prices, pd.Series(dtype=float), cfg), targets, prices, cfg


def test_analyze_end_to_end(analysis):
    ta, targets, *_ = analysis
    assert ta.target_date == F.next_kr_session(targets["KOSPI"].index[-1])
    assert {"ridge", "hybrid", "ens"} <= set(ta.preds) and all(np.isfinite(v) for v in ta.preds.values())
    assert ta.board.loc[M.MODEL_LABELS["ridge"], "MAE 개선율"] > 0.3  # 심어둔 EWY 세션내 신호를 포착
    assert "EWY_id" in ta.features


def test_interval_and_probability(analysis):
    ta, *_ = analysis
    iv = P.interval(ta, "ridge")
    assert iv["lo"] < iv["pred"] < iv["hi"] and 0 <= iv["p_up"] <= 1
    shifted = P.interval(ta, "ridge", shift=1.0)
    assert shifted["pred"] == pytest.approx(iv["pred"] + 1.0) and shifted["hi"] - shifted["lo"] == pytest.approx(iv["hi"] - iv["lo"])


def test_explain_is_additive_for_linear(analysis):
    ta, *_ = analysis
    c = P.explain(ta, "ridge")
    reg = ta.fit.models["ridge"].steps[-1][1]
    assert c.sum() == pytest.approx(ta.preds["ridge"] - reg.intercept_, abs=1e-6)
    assert c.index[0] in ("EWY_id", "EWY_excess", "EWY", "EWY_on")  # 심어둔 신호가 최상위
    assert P.explain(ta, "hybrid") is not None and P.explain(ta, "ens") is not None


def test_ablation_flags_the_planted_feature(analysis):
    _, targets, prices, cfg = analysis
    ab = P.ablation("KOSPI", targets, prices, pd.Series(dtype=float), cfg)
    row = ab.set_index("변형").loc["− 세션내/오버나이트 분해"]
    assert row["Δ MAE"] > 0.01  # 분해를 빼면 신호 일부를 잃는다 (일간 수익률에도 세션내 성분이 섞여 있어 폭은 작다)


def test_blend_and_gap_beta(analysis):
    _, targets, *_ = analysis
    assert P.blend_with_futures(1.0, None, 0.6) == 1.0 and P.blend_with_futures(1.0, 2.0, 0.5) == pytest.approx(1.5)
    assert P.gap_beta(targets["SEC"], targets["KOSPI"]) == pytest.approx(1.0, abs=0.05)  # 가격만 50배 → 갭 동일


def test_dram_pure_with_korean_holdings():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2025-01-01", periods=150)
    mu, sec, own = rng.normal(0, 2, 150), rng.normal(0, 1.5, 150), rng.normal(0, 1, 150)
    dram = 0.7 * mu + 0.3 * sec + own
    mk = lambda r: pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,  # noqa: E731
                                 "Close": 100 * np.cumprod(1 + r / 100)}, index=days)
    pure, info = F.dram_pure({"DRAM": mk(dram), "MU": mk(mu)}, ["MU"], {"SEC": mk(sec)})
    assert info["betas"]["MU"] == pytest.approx(0.7, abs=0.15) and info["betas"]["KR_SEC"] == pytest.approx(0.3, abs=0.15)
    assert abs(np.corrcoef(pure.to_numpy(), own[1:])[0, 1]) > 0.9
    assert F.dram_pure({"DRAM": mk(dram)}, [], None)[0].empty
