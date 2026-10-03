"""시장 데이터 수집: yfinance(가격) + esignal(야간선물, best-effort)."""
from __future__ import annotations

import re
from dataclasses import dataclass
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


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 인덱스(tz 제거, 일 단위)로 정리하고 종가가 없는 봉(장중 미완성 등)은 버린다."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    out = df[["Open", "High", "Low", "Close"]].copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["Close"])


def fetch_history(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    return _clean(yf.Ticker(ticker).history(period=period, auto_adjust=True))


def fetch_all(period: str = HISTORY_PERIOD) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str]]:
    """({대상키: 일봉}, {지표키: 일봉(DRAM·SNDK 포함)}, {키: 오류메시지}). 병렬 수집."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = (
        [("t", k, t) for k, (t, _) in TARGETS.items()]
        + [("p", k, t) for k, t in ASSETS.items()]
        + [("p", "DRAM", DRAM_TICKER), ("p", "SNDK", "SNDK")]
    )

    def one(job):
        kind, key, ticker = job
        try:
            df = fetch_history(ticker, period)
            return kind, key, df, (None if len(df) else f"{ticker}: 데이터 없음")
        except Exception as exc:  # 네트워크/비공식 API 오류는 앱을 죽이지 않는다
            return kind, key, None, f"{ticker}: {exc}"

    targets: dict[str, pd.DataFrame] = {}
    prices: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for kind, key, df, err in ex.map(one, jobs):
            if err:
                errors[key] = err
            elif df is not None:
                (targets if kind == "t" else prices)[key] = df
    return targets, prices, errors


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
