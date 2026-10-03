"""다음 거래일 KOSPI 시가 갭 추정 모델.

타깃: 갭(%) = (T일 시가 / T-1일 종가 - 1) * 100
지표: T일 이전 가장 최근 해외(미국/런던/환율) 거래일의 종가 대비 등락률(%).
 해외 T-1일 세션은 한국 T-1일 마감(15:30 KST) 이후에 열려 T일 개장 전에 끝나므로
 '한국 마감 이후의 움직임'을 담는다. (환율 KRW=X는 24시간 호가라 근사치.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def pct_change(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change() * 100


def build_dataset(kospi: pd.DataFrame, prices: dict[str, pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """KOSPI 거래일 T마다 T 이전 최근 해외 거래일의 등락률을 붙인 학습표 (y 포함)."""
    base = pd.DataFrame({"date": kospi.index})
    base["y"] = (kospi["Open"] / kospi["Close"].shift(1) - 1).to_numpy() * 100
    base = base.dropna().sort_values("date")
    for key in keys:
        ret = pct_change(prices[key]).dropna().rename(key).reset_index()
        ret.columns = ["date", key]
        base = pd.merge_asof(
            base, ret.sort_values("date"), on="date", direction="backward", allow_exact_matches=False,
            tolerance=pd.Timedelta(days=5),
        )
    return base.dropna().reset_index(drop=True)


def latest_features(prices: dict[str, pd.DataFrame], keys: list[str]) -> pd.Series:
    """각 지표의 가장 최근 등락률(%)."""
    return pd.Series({k: pct_change(prices[k]).dropna().iloc[-1] for k in keys})


def next_kr_session(last_kospi_date: pd.Timestamp) -> pd.Timestamp:
    """마지막 KOSPI 종가일 다음 평일 (공휴일은 반영하지 않음)."""
    d = last_kospi_date + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


@dataclass
class Fit:
    keys: list[str]
    scaler: StandardScaler
    model: Ridge

    def predict(self, x: pd.DataFrame | pd.Series) -> np.ndarray:
        X = pd.DataFrame(x).T if isinstance(x, pd.Series) else x
        return self.model.predict(self.scaler.transform(X[self.keys].to_numpy()))

    def coefficients(self) -> pd.Series:
        """원 단위 계수: 지표가 1%p 움직일 때 갭 추정치(%p) 변화."""
        return pd.Series(self.model.coef_ / self.scaler.scale_, index=self.keys)


def fit_model(data: pd.DataFrame, keys: list[str], alpha: float = 10.0) -> Fit:
    scaler = StandardScaler().fit(data[keys].to_numpy())
    model = Ridge(alpha=alpha).fit(scaler.transform(data[keys].to_numpy()), data["y"].to_numpy())
    return Fit(keys, scaler, model)


@dataclass
class Backtest:
    frame: pd.DataFrame  # date, actual, pred
    mae: float
    mae_baseline: float
    hit_rate: float
    corr: float
    resid_std: float


def walk_forward(data: pd.DataFrame, keys: list[str], alpha: float = 10.0, test_days: int = 120,
                 min_train: int = 60, refit_every: int = 10) -> Backtest | None:
    """확장 윈도우 워크포워드(미래 데이터 누수 없음). 표본이 부족하면 None."""
    n = len(data)
    start = max(min_train, n - test_days)
    if n - start < 20:
        return None
    preds = np.full(n, np.nan)
    fit = None
    for i in range(start, n):
        if fit is None or (i - start) % refit_every == 0:
            fit = fit_model(data.iloc[:i], keys, alpha)
        preds[i] = fit.predict(data.iloc[[i]])[0]
    out = pd.DataFrame({"date": data["date"], "actual": data["y"], "pred": preds}).iloc[start:]
    err = out["actual"] - out["pred"]
    return Backtest(
        frame=out.reset_index(drop=True),
        mae=float(err.abs().mean()),
        mae_baseline=float(out["actual"].abs().mean()),  # 항상 0%로 예측했을 때
        hit_rate=float((np.sign(out["actual"]) == np.sign(out["pred"])).mean()),
        corr=float(out["actual"].corr(out["pred"])),
        resid_std=float(err.std(ddof=1)),
    )


def blend_with_futures(model_pct: float, futures_pct: float | None, weight: float) -> float:
    """야간선물 등락률이 있으면 모델 추정치와 가중평균 (weight = 선물 비중)."""
    if futures_pct is None:
        return model_pct
    return (1 - weight) * model_pct + weight * futures_pct
