"""시장 데이터 수집: yfinance(가격) + esignal(야간선물, best-effort)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

# 예측 대상: key -> (티커, 라벨)
TARGETS: dict[str, tuple[str, str]] = {
    "KOSPI": ("^KS11", "KOSPI"),
    "SEC": ("005930.KS", "삼성전자"),
    "HYNIX": ("000660.KS", "SK하이닉스"),
}
# 모델 지표 풀: key -> 티커. 그룹은 src.features.GROUPS 와 대응한다.
ASSETS: dict[str, str] = {
    "EWY": "EWY", "FLXK": "FLXK.L", "USDKRW": "KRW=X", "SOXX": "SOXX", "SMH": "SMH", "MU": "MU",
    "NVDA": "NVDA", "TSM": "TSM", "AMAT": "AMAT", "WDC": "WDC", "STX": "STX",
    "SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB", "EEM": "EEM",
    "EWJ": "EWJ", "FXI": "FXI", "WTI": "CL=F", "GOLD": "GC=F",
    "DAX": "^GDAXI", "SX5E": "^STOXX50E", "FTSE": "^FTSE", "SMSN": "SMSN.IL",
}
# DRAM ETF (2026-04 상장, 이력 짧음) 와, 순수 수급 성분을 만들기 위해 제거할 구성종목
DRAM_TICKER = "DRAM"
DRAM_PEERS: dict[str, tuple[str, str]] = {
    "MU": ("MU", "마이크론"),
    "WDC": ("WDC", "웨스턴디지털"),
    "SNDK": ("SNDK", "샌디스크"),
    "STX": ("STX", "씨게이트"),
}
ESIGNAL_URL = "https://esignal.co.kr/kospi200-futures-night/"
HISTORY_PERIOD = "12y"
RETRY_BACKOFF = 1.5  # 초, 재시도마다 2배


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 인덱스(tz 제거, 일 단위)로 정리하고 종가가 없는 봉(장중 미완성 등)은 버린다."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    out = df[["Open", "High", "Low", "Close"]].copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["Close"])


def fetch_history(ticker: str, period: str = HISTORY_PERIOD, tries: int = 3) -> pd.DataFrame:
    """yfinance 일봉. Yahoo의 일시적 제한(429 등)에 대비해 지수 백오프로 재시도한다."""
    last: Exception | None = None
    for i in range(tries):
        try:
            df = _clean(yf.Ticker(ticker).history(period=period, auto_adjust=True))
            if len(df):
                return df
            last = RuntimeError("빈 응답")
        except Exception as exc:  # noqa: BLE001  yfinance 는 다양한 예외를 던진다
            last = exc
        if i < tries - 1:
            time.sleep(RETRY_BACKOFF * 2**i)
    raise RuntimeError(f"{ticker}: {last}")


# ---------------- 스냅샷 (실시간 수집 실패 시 폴백) ----------------
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "snapshot.csv.gz"


def save_snapshot(frames: dict[str, pd.DataFrame], path: Path | None = None) -> None:
    path = path or SNAPSHOT_PATH
    long = pd.concat({k: v[["Open", "High", "Low", "Close"]] for k, v in frames.items()}, names=["key", "date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    long.round(6).reset_index().to_csv(path, index=False, compression="gzip")


def load_snapshot(path: Path | None = None) -> dict[str, pd.DataFrame]:
    path = path or SNAPSHOT_PATH
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:  # noqa: BLE001  손상된 스냅샷은 없는 것으로 취급
        return {}
    return {k: g.set_index("date").drop(columns="key").sort_index() for k, g in df.groupby("key")}


def merge_history(snap: pd.DataFrame | None, live: pd.DataFrame | None) -> pd.DataFrame | None:
    """스냅샷(긴 과거) 위에 실시간(최근)을 이어 붙인다.

    수정주가(배당 등)는 시점마다 달라지므로 겹치는 구간의 종가 비율로 스냅샷을 live 기준으로 재조정해
    이음매에서 가짜 수익률이 생기지 않게 한다."""
    if live is None or live.empty:
        return snap
    if snap is None or snap.empty:
        return live
    live = live.copy()
    live.index = pd.DatetimeIndex(live.index).astype(snap.index.dtype)
    both = snap.index.intersection(live.index)
    scale = 1.0
    if len(both) >= 3:
        ratio = (live.loc[both, "Close"] / snap.loc[both, "Close"]).tail(20)
        scale = float(ratio.median())
        if not (0.5 < scale < 2.0):  # 액면분할 등 비정상 비율이면 재조정하지 않고 live만 신뢰
            return live
    old = snap[snap.index < live.index[0]] * scale
    return pd.concat([old, live])


@dataclass
class Market:
    targets: dict[str, pd.DataFrame]
    prices: dict[str, pd.DataFrame]
    errors: dict[str, str]   # 수집 실패 사유 (실패해도 스냅샷으로 대체됐을 수 있음)
    stale: dict[str, str]    # 실시간 수집 실패 → 스냅샷으로 대체한 키: 마지막 데이터 날짜


def fetch_all(period: str = HISTORY_PERIOD, workers: int = 4) -> Market:
    """대상·지표 일봉 수집. 스냅샷이 있으면 최근 1년만 받아 이어 붙여 요청량을 줄이고,
    개별 수집이 실패하면 스냅샷으로 대체한다. 동시 요청은 Yahoo 제한을 피하려고 적게 둔다."""
    from concurrent.futures import ThreadPoolExecutor

    snap = load_snapshot()
    jobs = (
        [("t", k, t) for k, (t, _) in TARGETS.items()]
        + [("p", k, t) for k, t in ASSETS.items()]
        + [("p", "DRAM", DRAM_TICKER), ("p", "SNDK", "SNDK")]
    )

    def one(job):
        kind, key, ticker = job
        have = snap.get(key)
        live_period = "1y" if have is not None and len(have) > 250 else period
        try:
            live = fetch_history(ticker, live_period)
            return kind, key, merge_history(have, live), None, False
        except Exception as exc:  # noqa: BLE001  네트워크/비공식 API 오류는 앱을 죽이지 않는다
            return kind, key, have, str(exc), have is not None

    targets: dict[str, pd.DataFrame] = {}
    prices: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    stale: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for kind, key, df, err, fell_back in ex.map(one, jobs):
            if err:
                errors[key] = err
            if df is None or df.empty:
                continue
            if fell_back:
                stale[key] = f"{df.index[-1]:%Y-%m-%d}"
            (targets if kind == "t" else prices)[key] = df
    return Market(targets, prices, errors, stale)


@dataclass
class NightFutures:
    change_pct: float
    source: str


_PCT_RE = re.compile(r"([+-]?\d{1,2}\.\d{1,2})\s*%")


def fetch_night_futures(timeout: float = 8.0) -> Optional[NightFutures]:
    """esignal 야간선물 등락률 자동 수집 시도. 실패하면 None.

    esignal은 실시간 값을 웹소켓(세션 티켓 인증)으로 내려주므로 정적 HTML에
    값이 없는 것이 일반적이다. 인증 우회는 하지 않고, 페이지에 값이 노출될
    때만 읽는다. None이면 UI에서 수동 입력을 받는다.
    """
    try:
        res = requests.get(
            ESIGNAL_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (kospi-tracker)"}
        )
        res.raise_for_status()
    except requests.RequestException:
        return None
    return parse_night_futures(res.text)


def parse_night_futures(html: str) -> Optional[NightFutures]:
    """HTML 안의 `class="night-change"`/`data-night-change` 같은 노출 값을 찾는다."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    for node in soup.select("[data-night-change], .night-change, #night-change"):
        text = node.get("data-night-change") or node.get_text(" ", strip=True)
        m = _PCT_RE.search(text if "%" in text else text + "%")
        if m:
            return NightFutures(float(m.group(1)), "esignal.co.kr")
    return None


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401

        return True
    except ImportError:
        return False
