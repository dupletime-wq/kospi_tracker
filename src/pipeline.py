"""분석 파이프라인: 피처 → 워크포워드 모델 비교 → 최종 적합·예측 → 구간/기여도."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features as F
from . import models as M


FAST_MODELS = ("baseline", "ridge", "huber", "lgbm", "hybrid", "ens")
ALL_MODELS = ("baseline", "ridge", "lasso", "enet", "huber", "rf", "hgb", "lgbm", "hybrid", "ens")


@dataclass(frozen=True)
class Config:
    groups: tuple[str, ...] = ("core", "kr_gdr")
    intraday: str = "add"
    own: bool = True
    engineered: bool = True
    calendar: bool = True
    dram: bool = True
    peers: tuple[str, ...] = ("MU", "WDC", "SNDK", "STX")
    half_life: float | None = 500
    clip_q: float = 0.0
    vol_wls: bool = False
    test_rows: int = 500
    refit_every: int = 25
    models: tuple[str, ...] = FAST_MODELS


@dataclass
class TargetAnalysis:
    key: str
    hist: pd.DataFrame            # y가 있는 학습 행
    x_new: pd.Series              # 다음 거래일 입력
    features: list[str]
    bt: pd.DataFrame              # 워크포워드 예측 (date, y, 모델별)
    board: pd.DataFrame           # 리더보드
    fit: M.FitResult
    preds: dict[str, float]
    last_close: float
    last_date: pd.Timestamp
    target_date: pd.Timestamp
    bt_nodram: pd.DataFrame | None = None
    cfg: Config = field(default_factory=Config)


def cross_list(cfg: Config) -> list[str]:
    return [a for g in cfg.groups for a in F.GROUPS[g]]


def analyze(tkey: str, targets: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame],
            pure: pd.Series, cfg: Config) -> TargetAnalysis | None:
    extra = {"DRAM_PURE": pure} if (cfg.dram and len(pure)) else None
    full = F.build_frame(tkey, targets, prices, cross_list(cfg), cfg.own, cfg.engineered, cfg.calendar, extra,
                         future=True, intraday=cfg.intraday)
    hist = full.dropna(subset=["y"]).reset_index(drop=True)
    if len(hist) < 450:
        return None
    feats = F.feature_columns(full)
    kw = dict(half_life=cfg.half_life, clip_q=cfg.clip_q, vol_wls=cfg.vol_wls)
    bt = M.walk_forward(hist, feats, list(cfg.models), cfg.test_rows, refit_every=cfg.refit_every, **kw)
    if bt.empty:
        return None
    fit = M.fit_all(hist, feats, list(cfg.models), **kw)
    x_new = full.iloc[[-1]]
    preds = {m: float(fit.predict(m, x_new)[0]) for m in cfg.models}
    bt_nd = None
    if "DRAM_PURE" in feats:
        nd = [f for f in feats if f != "DRAM_PURE"]
        bt_nd = M.walk_forward(hist, nd, ["ridge", "lgbm"], cfg.test_rows, refit_every=cfg.refit_every, **kw)
    tdf = targets[tkey]
    return TargetAnalysis(tkey, hist, full.iloc[-1], feats, bt, M.leaderboard(bt, list(cfg.models)), fit, preds,
                          float(tdf["Close"].iloc[-1]), tdf.index[-1], full["date"].iloc[-1], bt_nd, cfg)


def best_model(ta: TargetAnalysis) -> str:
    """워크포워드 MAE가 가장 낮은 모델(기준선 제외). 선택 편향이 있으니 참고용."""
    inv = {v: k for k, v in M.MODEL_LABELS.items()}
    return inv[ta.board.drop(index=M.MODEL_LABELS["baseline"]).index[0]]


def interval(ta: TargetAnalysis, model: str, shift: float = 0.0, lo_q: float = 0.1, hi_q: float = 0.9):
    """변동성으로 정규화한 워크포워드 잔차의 분위수로 만든 예측 구간과 상승확률.

    shift: 야간선물 반영 등으로 점추정이 이동한 양(%p). 구간은 같은 폭으로 이동한다."""
    pred = ta.preds[model] + shift
    scale = ta.hist.set_index("date")["own_vol20"].reindex(ta.bt["date"]).to_numpy()
    z = ((ta.bt["y"] - ta.bt[model]).to_numpy() / scale)
    z = z[np.isfinite(z)]
    s_now = float(ta.x_new["own_vol20"]) if np.isfinite(ta.x_new["own_vol20"]) else float(np.nanmedian(scale))
    if len(z) < 30:
        return None
    lo, hi = pred + np.quantile(z, lo_q) * s_now, pred + np.quantile(z, hi_q) * s_now
    return {"pred": pred, "lo": float(lo), "hi": float(hi), "p_up": float(((pred + z * s_now) > 0).mean())}


def _contrib(model, features: list[str], X: pd.DataFrame) -> pd.Series | None:
    """모델 하나의 피처별 기여(%p, 평균 입력 대비). 지원 안 하면 None."""
    if isinstance(model, M._Hybrid):
        a, b = _contrib(model.lin, features, X), _contrib(model.gbm, features, X)
        return a.add(b, fill_value=0.0)
    if hasattr(model, "steps"):
        reg = model.steps[-1][1]
        if hasattr(reg, "coef_"):
            Z = model[:-1].transform(X[features])
            return pd.Series(reg.coef_ * Z[0], index=features)
        return None
    if model.__class__.__name__ == "LGBMRegressor":
        c = model.predict(X[features], pred_contrib=True)[0][:-1]
        return pd.Series(c, index=features)
    return None


def explain(ta: TargetAnalysis, model: str) -> pd.Series | None:
    X = pd.DataFrame([ta.x_new])
    members = M.ENSEMBLE_MEMBERS if model == "ens" else (model,)
    parts = [c for m in members if m in ta.fit.models and (c := _contrib(ta.fit.models[m], ta.fit.features, X)) is not None]
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1).sort_values(key=np.abs, ascending=False)


def blend_with_futures(model_pct: float, futures_pct: float | None, weight: float) -> float:
    """야간선물 등락률이 있으면 모델 추정치와 가중평균 (weight = 선물 비중)."""
    if futures_pct is None:
        return model_pct
    return (1 - weight) * model_pct + weight * futures_pct


def gap_beta(target: pd.DataFrame, kospi: pd.DataFrame) -> float:
    """대상 종목 시가 갭을 KOSPI 시가 갭에 회귀한 기울기 (야간선물 충격을 개별주에 전달할 때 사용)."""
    g = lambda d: ((d["Open"] / d["Close"].shift(1) - 1) * 100).rename("g")  # noqa: E731
    df = pd.concat([g(target), g(kospi)], axis=1, keys=["s", "k"], sort=True).dropna().tail(750)
    if len(df) < 60 or df["k"].var() == 0:
        return 1.0
    return float(np.clip(df["s"].cov(df["k"]) / df["k"].var(), 0.0, 3.0))


# feature 그룹 ablation: 라벨 -> Config 변경(제거) 또는 추가할 그룹
def _without(cfg: Config, group: str) -> Config:
    return Config(**{**cfg.__dict__, "groups": tuple(g for g in cfg.groups if g != group)})


def _with(cfg: Config, group: str) -> Config:
    return Config(**{**cfg.__dict__, "groups": tuple(dict.fromkeys(cfg.groups + (group,)))})


def _replace(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


def ablation(tkey: str, targets: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame], pure: pd.Series,
             cfg: Config, model: str = "ridge") -> pd.DataFrame:
    """같은 워크포워드 구간에서 feature 묶음을 하나씩 빼거나 더했을 때 MAE 변화. 음수(Δ<0)가 개선."""
    variants: dict[str, Config] = {"현재 설정": cfg}
    if "kr_gdr" in cfg.groups:
        variants["− 삼성전자 런던 GDR"] = _without(cfg, "kr_gdr")
    variants["− 자기 시장 래그"] = _replace(cfg, own=False)
    variants["− 가공 지표(원화환산·증분)"] = _replace(cfg, engineered=False)
    variants["− 달력"] = _replace(cfg, calendar=False)
    variants["− 세션내/오버나이트 분해"] = _replace(cfg, intraday="none")
    if cfg.dram and len(pure):
        variants["− DRAM 순수 수급"] = _replace(cfg, dram=False)
    for g, label in (("europe", "+ 유럽 지수"), ("semis_ext", "+ 반도체 확장(NVDA·TSM·AMAT…)"),
                     ("us_macro", "+ 미국 매크로(SPY·VIX·금리…)"), ("asia_cmd", "+ 일본·중국·원자재")):
        if g not in cfg.groups:
            variants[label] = _with(cfg, g)
    rows = []
    for label, c in variants.items():
        extra = {"DRAM_PURE": pure} if (c.dram and len(pure)) else None
        fr = F.build_frame(tkey, targets, prices, cross_list(c), c.own, c.engineered, c.calendar, extra,
                           intraday=c.intraday).dropna(subset=["y"])
        bt = M.walk_forward(fr, F.feature_columns(fr), [model], c.test_rows, refit_every=c.refit_every,
                            half_life=c.half_life, clip_q=c.clip_q, vol_wls=c.vol_wls)
        if bt.empty:
            continue
        rows.append({"변형": label, "피처 수": len(F.feature_columns(fr)),
                     "MAE": float((bt["y"] - bt[model]).abs().mean())})
    out = pd.DataFrame(rows)
    out["Δ MAE"] = out["MAE"] - float(out.loc[out["변형"] == "현재 설정", "MAE"].iloc[0])
    return out
