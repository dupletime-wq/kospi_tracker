"""24시간 거래되는 삼성전자·SK하이닉스 무기한 선물 → 한국 마감 후 가격 신호(야간선물 대용).

거래소(Gate → Bitget → OKX 순 폴백)의 30분봉으로 "한국 전일 종가(15:30 KST)부터 지금(최대 개장 30분 전)까지"
수익률을 만들고, 모델 예측 위에 2단계(stacking) 회귀로 얹어 보정한다.

시각 규칙(모두 UTC): 한국 마감 06:30, 한국 개장 00:00(=09:00 KST). 30분봉의 타임스탬프는 *시작* 시각이므로
시각 t 의 가격 = 시작이 t-30분인 봉의 종가.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .features import next_kr_session

BAR = pd.Timedelta("30min")
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "perp_snapshot.csv.gz"
EXCHANGES = ("gate", "bitget", "okx")
SYMBOLS: dict[str, dict[str, str]] = {
    "SEC": {"gate": "SAMSUNG_USDT", "bitget": "SAMSUNGUSDT", "okx": "SAMSUNG-USDT-SWAP"},
    "HYNIX": {"gate": "SKHYNIX_USDT", "bitget": "SKHYNIXUSDT", "okx": "SKHYNIX-USDT-SWAP"},
}
GATE_LISTING = pd.Timestamp("2026-06-02")  # 두 선물 모두 이 무렵 상장
MIN_STACK_DAYS = 30
_HEADERS = {"User-Agent": "kospi-tracker/1.0"}


class PerpUnavailable(RuntimeError):
    pass


# ---------------- 거래소 수집 ----------------
def _closed(s: pd.Series, now: pd.Timestamp) -> pd.Series:
    """아직 닫히지 않은 마지막 봉을 버린다."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[s.index + BAR <= now]


def _get(url: str, params: dict, timeout: float = 10.0):
    r = requests.get(url, params=params, timeout=timeout, headers=_HEADERS)
    r.raise_for_status()
    return r.json()


def fetch_gate(contract: str, start: pd.Timestamp, now: pd.Timestamp) -> pd.Series:
    out: list[dict] = []
    t, end_all = int(start.timestamp()), int(now.timestamp())
    for _ in range(40):  # 요청당 최대 2000봉(약 41일)
        if t >= end_all:
            break
        end = min(t + 1900 * 1800, end_all)
        d = _get("https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                 {"contract": contract, "interval": "30m", "from": t, "to": end})
        if not isinstance(d, list):
            raise PerpUnavailable(f"gate: {str(d)[:100]}")
        out += d
        t = end + 1
    if not out:
        raise PerpUnavailable("gate: 빈 응답")
    s = pd.Series({pd.Timestamp(x["t"], unit="s"): float(x["c"]) for x in out})
    return _closed(s, now)


def fetch_bitget(symbol: str, now: pd.Timestamp) -> pd.Series:
    d = _get("https://api.bitget.com/api/v2/mix/market/candles",
             {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": "30m", "limit": 1000}).get("data") or []
    if not d:
        raise PerpUnavailable("bitget: 빈 응답")
    s = pd.Series({pd.Timestamp(int(x[0]), unit="ms"): float(x[4]) for x in d})
    return _closed(s, now)


def fetch_okx(inst: str, now: pd.Timestamp) -> pd.Series:
    d = _get("https://www.okx.com/api/v5/market/candles", {"instId": inst, "bar": "30m", "limit": 300}).get("data") or []
    if not d:
        raise PerpUnavailable("okx: 빈 응답")
    s = pd.Series({pd.Timestamp(int(x[0]), unit="ms"): float(x[4]) for x in d if x[8] == "1"})
    return _closed(s, now)


# ---------------- 스냅샷 ----------------
def save_snapshot(series: dict[str, pd.Series], path: Path | None = None) -> None:
    path = path or SNAPSHOT_PATH
    long = pd.concat({k: v.rename("c") for k, v in series.items()}, names=["key", "t"]).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(path, index=False, compression="gzip")


def load_snapshot(path: Path | None = None) -> dict[str, pd.Series]:
    path = path or SNAPSHOT_PATH
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, parse_dates=["t"])
    except Exception:  # noqa: BLE001
        return {}
    return {k: g.set_index("t")["c"].sort_index() for k, g in df.groupby("key")}


@dataclass
class PerpSeries:
    close: pd.Series      # 30분봉 종가(UTC 시작 시각 인덱스): 스냅샷 + 실시간
    source: str           # 실시간 수집에 성공한 거래소
    live_end: pd.Timestamp  # 마지막으로 닫힌 봉의 종료 시각


def get_series(key: str, now: pd.Timestamp | None = None, exchanges=EXCHANGES,
               snapshot: dict[str, pd.Series] | None = None) -> PerpSeries:
    """실시간 최신 구간을 거래소 폴백으로 받아 스냅샷(긴 과거) 위에 이어 붙인다. 모두 실패하면 PerpUnavailable."""
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    snap = (snapshot if snapshot is not None else load_snapshot()).get(key)
    errors = []
    for ex in exchanges:
        sym = SYMBOLS[key][ex]
        try:
            if ex == "gate":  # 스냅샷 이후 빈 구간까지 메꿈 (최소 5일)
                start = max(GATE_LISTING, (snap.index[-1] - pd.Timedelta(days=1)) if snap is not None and len(snap)
                            else GATE_LISTING, now - pd.Timedelta(days=41))
                start = min(start, now - pd.Timedelta(days=5))
                live = fetch_gate(sym, start, now)
            elif ex == "bitget":
                live = fetch_bitget(sym, now)
            else:
                live = fetch_okx(sym, now)
        except Exception as exc:  # noqa: BLE001  네트워크·형식 오류는 다음 거래소로
            errors.append(f"{ex}: {exc}")
            continue
        if live.empty:
            errors.append(f"{ex}: 닫힌 봉 없음")
            continue
        close = live if snap is None else pd.concat([snap[snap.index < live.index[0]], live])
        return PerpSeries(close, ex, live.index[-1] + BAR)
    raise PerpUnavailable("; ".join(errors))


# ---------------- 시각 정렬 · 수익률 ----------------
def kr_close_ts(date) -> pd.Timestamp:
    return pd.Timestamp(date).normalize() + pd.Timedelta(hours=6, minutes=30)


def kr_open_ts(date) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def price_at(close: pd.Series, ts: pd.Timestamp) -> float:
    return float(close.get(ts - BAR, np.nan))


def perp_return(close: pd.Series, prev_date, asof: pd.Timestamp) -> float:
    """한국 prev_date 종가 → asof 까지 선물 수익률(%)."""
    a, b = price_at(close, kr_close_ts(prev_date)), price_at(close, asof)
    return (b / a - 1) * 100 if a > 0 and b > 0 else float("nan")


def live_asof(now: pd.Timestamp, target_date, last_date) -> tuple[pd.Timestamp, int] | None:
    """지금 쓸 수 있는 가장 최근 시각(30분 단위, 개장 30분 전 이내)과 한국 마감 후 경과 분(elapsed).
    마감 직후 봉이 아직 닫히지 않았으면 None."""
    close = kr_close_ts(last_date)
    asof = min(now.floor("30min"), kr_open_ts(target_date) - BAR)
    if asof <= close:
        return None
    return asof, int((asof - close) / pd.Timedelta("1min"))


def history(close: pd.Series, kr_dates: pd.DatetimeIndex, elapsed_min: int) -> pd.Series:
    """과거 각 한국 거래일 T 에 대해, 전일 종가 후 elapsed 분(단 개장 30분 전을 넘지 않음) 시점까지의 선물 수익률.
    라이브와 같은 정보량으로 보정 회귀를 맞추기 위한 것이며, 주말 낀 월요일은 더 긴 구간이 허용된다."""
    rows = {}
    dates = list(pd.DatetimeIndex(kr_dates))
    for prev, T in zip(dates[:-1], dates[1:]):
        asof = min(kr_close_ts(prev) + pd.Timedelta(minutes=elapsed_min), kr_open_ts(T) - BAR)
        rows[T] = perp_return(close, prev, asof)
    return pd.Series(rows, dtype=float).dropna()


# ---------------- 2단계 보정(stacking) ----------------
def _ols(base: np.ndarray, perp: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(len(y)), base, perp]
    return np.linalg.lstsq(X, y, rcond=None)[0]


@dataclass
class Stack:
    coef: np.ndarray          # [절편, 모델 계수, 선물 계수]
    sigma: float              # 잔차 표준편차(%p)
    n: int
    frame: pd.DataFrame       # date, y, base, perp, oos(확장 윈도우 표본외 보정 예측)
    metrics: dict

    @property
    def useful(self) -> bool:
        """표본외(확장 윈도우)에서 모델 단독과 '모델만 재보정한 것' 둘 다보다 MAE가 줄 때만 보정을 적용한다.
        (이른 시각엔 선물 정보가 약해 해로울 수 있고, 2단계 회귀가 단지 모델 스케일을 고치는 효과와 구분하기 위함)"""
        m = self.metrics
        return m["n_oos"] >= 20 and m["MAE 개선"] > 0 and m["MAE 개선(재보정 대비)"] > 0

    def predict(self, base: float, perp: float) -> float:
        return float(self.coef[0] + self.coef[1] * base + self.coef[2] * perp)


def fit_stack(df: pd.DataFrame, min_train: int = 25) -> Stack | None:
    """df: date, y, base(모델 예측), perp(선물 수익률). 표본이 MIN_STACK_DAYS 미만이면 None.

    표본외 성능은 확장 윈도우(과거만으로 학습해 다음 날 예측)로 계산해 낙관 편향을 피한다."""
    d = df.dropna(subset=["y", "base", "perp"]).sort_values("date").reset_index(drop=True)
    n = len(d)
    if n < MIN_STACK_DAYS:
        return None
    y, b, p = d["y"].to_numpy(), d["base"].to_numpy(), d["perp"].to_numpy()
    oos, oos_b = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(min_train, n):
        c = _ols(b[:i], p[:i], y[:i])
        oos[i] = c[0] + c[1] * b[i] + c[2] * p[i]
        cb = np.polyfit(b[:i], y[:i], 1)  # 선물 없이 모델 예측만 재보정(스케일·절편)한 비교 대상
        oos_b[i] = cb[1] + cb[0] * b[i]
    coef = _ols(b, p, y)
    resid = y - (coef[0] + coef[1] * b + coef[2] * p)
    sigma = float(np.sqrt((resid**2).sum() / max(n - 3, 1)))
    d["oos"], d["oos_base"] = oos, oos_b
    m = d.dropna(subset=["oos"])
    mae_recal = float((m["y"] - m["oos_base"]).abs().mean())
    e_s, e_b = m["y"] - m["oos"], m["y"] - m["base"]
    # 기준선: 같은 날들에 대해 그 시점까지의 과거 중앙값으로 예측했을 때
    med = np.array([np.median(y[:i]) for i in range(min_train, n)])
    mae0 = float(np.abs(m["y"].to_numpy() - med).mean())
    metrics = {
        "MAE(기준선)": mae0, "MAE(모델 재보정)": mae_recal,
        "n": n, "n_oos": len(m), "MAE(모델)": float(e_b.abs().mean()), "MAE(보정)": float(e_s.abs().mean()),
        "MAE 개선": 1 - float(e_s.abs().mean() / e_b.abs().mean()),
        "방향(모델)": float((np.sign(m["y"]) == np.sign(m["base"])).mean()),
        "방향(보정)": float((np.sign(m["y"]) == np.sign(m["oos"])).mean()),
        "MAE 개선(재보정 대비)": 1 - float(e_s.abs().mean()) / mae_recal,
        "상관(모델)": float(m["y"].corr(m["base"])), "상관(보정)": float(m["y"].corr(m["oos"])),
        "상관(선물)": float(m["y"].corr(m["perp"])),
    }
    return Stack(coef, sigma, n, d[["date", "y", "base", "perp", "oos", "oos_base"]], metrics)


@dataclass
class Correction:
    asof: pd.Timestamp
    elapsed: int          # 한국 마감 후 경과 분
    source: str
    live: dict[str, float]            # 대상별 현재 선물 수익률(%)
    stacks: dict[str, Stack | None]   # 대상별 보정 회귀(None이면 표본 부족)


def build_correction(base_preds: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame],
                     now: pd.Timestamp | None = None, series: dict[str, PerpSeries] | None = None,
                     ) -> Correction:
    """base_preds[key] = 모델 워크포워드 예측 DataFrame(date, y, pred). 보정 대상: SEC, HYNIX, KOSPI(두 선물 평균).

    PerpUnavailable 은 호출자가 잡아 '모델 단독'으로 폴백한다."""
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    series = series or {k: get_series(k, now) for k in SYMBOLS}
    kr = targets["KOSPI"].index
    last_date, target = kr[-1], next_kr_session(kr[-1])
    ao = live_asof(now, target, last_date)
    if ao is None:
        raise PerpUnavailable("한국 마감 직후 봉이 아직 닫히지 않았습니다")
    asof, elapsed = ao
    live = {k: perp_return(series[k].close, last_date, asof) for k in SYMBOLS}
    if any(np.isnan(v) for v in live.values()):
        raise PerpUnavailable("현재 선물 가격을 계산할 수 없습니다(봉 누락)")
    live["KOSPI"] = float(np.mean([live["SEC"], live["HYNIX"]]))
    hist = {k: history(series[k].close, kr, elapsed) for k in SYMBOLS}
    hist["KOSPI"] = pd.concat([hist["SEC"], hist["HYNIX"]], axis=1).mean(axis=1).dropna()
    stacks = {}
    for k, bp in base_preds.items():
        if k not in hist:
            continue
        df = bp.rename(columns={"pred": "base"})[["date", "y", "base"]].merge(
            hist[k].rename("perp").rename_axis("date").reset_index(), on="date", how="inner")
        stacks[k] = fit_stack(df)
    return Correction(asof, elapsed, series["SEC"].source, live, stacks)


def normal_interval(pred: float, sigma: float, q: float = 1.2816) -> tuple[float, float, float]:
    """정규 근사 80% 구간과 상승확률."""
    from math import erf, sqrt

    return pred - q * sigma, pred + q * sigma, 0.5 * (1 + erf(pred / (sigma * sqrt(2)))) if sigma > 0 else float(pred > 0)
