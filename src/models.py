"""모델 동물원 + 워크포워드 평가.

모든 모델은 같은 피처 행렬/같은 분할로 비교한다. 훈련 시 타깃은 윈저라이즈(이상값 완화)하고,
최근 데이터에 지수 감쇠 가중을 줄 수 있다. 평가 지표는 원본(미조정) 타깃 기준이다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LassoCV, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODEL_LABELS = {
    "baseline": "기준선 (과거 중앙값)",
    "ridge": "Ridge",
    "lasso": "Lasso (CV)",
    "enet": "ElasticNet (CV)",
    "huber": "Huber 회귀",
    "rf": "Random Forest",
    "hgb": "Gradient Boosting (sklearn)",
    "lgbm": "LightGBM",
    "hybrid": "Ridge + LightGBM 잔차 (하이브리드)",
    "ens": "앙상블 (Ridge·Huber·하이브리드·LGBM 평균)",
}
LINEAR = ("ridge", "lasso", "enet", "huber")
TREES = ("rf", "hgb", "lgbm")
ENSEMBLE_MEMBERS = ("ridge", "huber", "hybrid", "lgbm")
ALPHAS = np.logspace(-3, 0.3, 30)  # 표준화된 피처, 타깃 단위 %


class _Hybrid:
    """선형 모델(Ridge)로 주신호를 잡고, 강하게 규제한 LightGBM으로 잔차의 비선형·상호작용만 보정."""

    def __init__(self, seed: int = 0):
        import lightgbm as lgb

        self.lin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
        self.gbm = lgb.LGBMRegressor(objective="huber", alpha=1.0, n_estimators=150, learning_rate=0.03, num_leaves=4,
                                     min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                     reg_lambda=20.0, random_state=seed, n_jobs=1, verbose=-1)

    def fit(self, X, y, sample_weight=None):
        self.lin.fit(X, y, ridge__sample_weight=sample_weight)
        self.gbm.fit(X, y - self.lin.predict(X), sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.lin.predict(X) + self.gbm.predict(X)


class _Baseline:
    def fit(self, X, y, sample_weight=None):
        self.m_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


def make_model(name: str, seed: int = 0):
    """이름으로 새 모델 인스턴스를 만든다. 선형 모델은 결측 대치+표준화 파이프라인."""
    pre = lambda: [SimpleImputer(strategy="median"), StandardScaler()]  # noqa: E731
    cv = TimeSeriesSplit(n_splits=4)
    if name == "baseline":
        return _Baseline()
    if name == "hybrid":
        return _Hybrid(seed)
    if name == "ridge":
        return make_pipeline(*pre(), Ridge(alpha=100.0))
    if name == "lasso":
        return make_pipeline(*pre(), LassoCV(alphas=ALPHAS, cv=cv, max_iter=5000, random_state=seed))
    if name == "enet":
        return make_pipeline(*pre(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=ALPHAS, cv=cv, max_iter=5000,
                                                  random_state=seed))
    if name == "huber":
        return make_pipeline(*pre(), HuberRegressor(epsilon=1.35, alpha=50.0, max_iter=500))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=120, min_samples_leaf=25, max_features=0.4, max_samples=0.6, n_jobs=1, random_state=seed))
    if name == "hgb":  # 이 크기(수천 행)에서는 히스토그램 방식보다 고전 GBM이 빠르고 안정적
        return make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(
            loss="huber", n_estimators=150, learning_rate=0.04, max_depth=3, subsample=0.7, max_features=0.5,
            min_samples_leaf=30, random_state=seed))
    if name == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(objective="huber", alpha=1.0, n_estimators=250, learning_rate=0.03, num_leaves=7,
                                 min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                 reg_lambda=10.0, random_state=seed, n_jobs=1, verbose=-1)
    raise KeyError(name)


def _fit(name: str, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, seed: int = 0):
    m = make_model(name, seed)
    kw: dict = {}
    if w is not None and name != "baseline":
        kw = {f"{m.steps[-1][0]}__sample_weight": w} if hasattr(m, "steps") else {"sample_weight": w}
    # 작은 데이터에서 OpenMP/BLAS 스레드가 과다 생성되면 수 배 느려지고 병렬 분석끼리 간섭하므로 1개로 제한
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        m.fit(X, y, **kw)
    return m


def prepare_target(y: np.ndarray, clip_q: float = 0.005) -> np.ndarray:
    """이상값(분할/이벤트성 갭)을 훈련에서만 윈저라이즈."""
    if clip_q <= 0 or len(y) < 50:
        return y
    lo, hi = np.quantile(y, [clip_q, 1 - clip_q])
    return np.clip(y, lo, hi)


def recency_weights(n: int, half_life: float | None) -> np.ndarray | None:
    if not half_life:
        return None
    return 0.5 ** ((n - 1 - np.arange(n)) / half_life)


@dataclass
class FitResult:
    models: dict
    features: list[str]

    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        if name == "ens":
            return np.mean([self.models[m].predict(X[self.features]) for m in ENSEMBLE_MEMBERS if m in self.models], axis=0)
        return self.models[name].predict(X[self.features])


def vol_weights(frame: pd.DataFrame, col: str = "kospi_vol20") -> np.ndarray | None:
    """변동성이 큰 날의 영향력을 낮추는 WLS 가중 (1/σ²). 레짐별 갭 분산 차이(이분산성) 완화."""
    if col not in frame:
        return None
    v = frame[col].astype(float)
    v = v.fillna(v.median()).clip(lower=v.quantile(0.05))
    w = 1.0 / (v**2)
    return (w / w.mean()).to_numpy()


def fit_all(frame: pd.DataFrame, features: list[str], names: list[str], half_life: float | None = None,
            clip_q: float = 0.005, seed: int = 0, vol_wls: bool = False) -> FitResult:
    tr = frame.dropna(subset=["y"])
    features = [f for f in features if tr[f].notna().sum() >= 20]  # 훈련 구간에 값이 거의 없는 피처(상장 전 등)는 제외
    y = prepare_target(tr["y"].to_numpy(), clip_q)
    w = recency_weights(len(tr), half_life)
    if vol_wls and (vw := vol_weights(tr)) is not None:
        w = vw if w is None else w * vw
    need = set(names) | (set(ENSEMBLE_MEMBERS) if "ens" in names else set())
    models = {n: _fit(n, tr[features], y, w, seed) for n in need if n != "ens"}
    return FitResult(models, features)


def walk_forward(frame: pd.DataFrame, features: list[str], names: list[str], test_rows: int = 250,
                 min_train: int = 400, refit_every: int = 25, half_life: float | None = None,
                 clip_q: float = 0.005, seed: int = 0, vol_wls: bool = False) -> pd.DataFrame:
    """확장 윈도우 워크포워드(미래 누수 없음). 열: date, y, <모델명>…  표본 부족하면 빈 DataFrame."""
    fr = frame.dropna(subset=["y"]).reset_index(drop=True)
    n = len(fr)
    start = max(min_train, n - test_rows)
    if n - start < 20:
        return pd.DataFrame()
    preds = {m: np.full(n, np.nan) for m in names}
    for i in range(start, n, refit_every):
        j = min(i + refit_every, n)
        fit = fit_all(fr.iloc[:i], features, names, half_life, clip_q, seed, vol_wls)
        for m in names:
            preds[m][i:j] = fit.predict(m, fr.iloc[i:j])
    out = pd.DataFrame({"date": fr["date"], "y": fr["y"], **preds}).iloc[start:].reset_index(drop=True)
    return out


def metrics(bt: pd.DataFrame, name: str, base: str = "baseline") -> dict:
    e = bt["y"] - bt[name]
    b = bt["y"] - bt[base]
    mae, mae_b = float(e.abs().mean()), float(b.abs().mean())
    d = b.abs() - e.abs()  # 양수 = 기준선보다 좋음
    return {
        "MAE": mae, "RMSE": float(np.sqrt((e**2).mean())), "MAE 개선율": 1 - mae / mae_b,
        "방향적중": float((np.sign(bt["y"]) == np.sign(bt[name])).mean()), "상관": float(bt["y"].corr(bt[name])),
        "t(vs 기준선)": _hac_t(d.to_numpy()),
    }


def _hac_t(d: np.ndarray, lag: int = 5) -> float:
    """손실 차이의 평균에 대한 Newey-West t값 (Diebold-Mariano 근사)."""
    n = len(d)
    if n < 30 or np.allclose(d.std(), 0):
        return float("nan")
    x = d - d.mean()
    var = (x @ x) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    return float(d.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def leaderboard(bt: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = {MODEL_LABELS[m]: metrics(bt, m) for m in names}
    return pd.DataFrame(rows).T.sort_values("MAE")
