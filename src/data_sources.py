"""시장 데이터 수집: yfinance(가격) + esignal(야간선물, best-effort)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

# key -> (yfinance 티커, 한글 라벨, 기본 학습 포함 여부)
FEATURES: dict[str, tuple[str, str, bool]] = {
    "EWY": ("EWY", "EWY (미국상장 한국 ETF)", True),
    "FLXK": ("FLXK.L", "Franklin FTSE Korea UCITS (런던)", True),
    "USDKRW": ("KRW=X", "USD/KRW 환율", True),
    "SOXX": ("SOXX", "SOXX (반도체 ETF)", True),
    "SMH": ("SMH", "SMH (반도체 ETF)", True),
    "MU": ("MU", "마이크론 (MU)", True),
    # 2026-04 상장으로 이력이 짧아 기본 제외
    "DRAM": ("DRAM", "DRAM ETF (메모리, 상장 6개월 내외)", False),
}
KOSPI_TICKER = "^KS11"
ESIGNAL_URL = "https://esignal.co.kr/kospi200-futures-night/"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 인덱스(tz 제거, 일 단위)로 정리하고 종가가 없는 봉(장중 미완성 등)은 버린다."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    out = df[["Open", "High", "Low", "Close"]].copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["Close"])


def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    return _clean(yf.Ticker(ticker).history(period=period, auto_adjust=True))


def fetch_all(period: str = "5y") -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, str]]:
    """(KOSPI 일봉, {지표키: 일봉}, {지표키: 오류메시지})."""
    kospi = fetch_history(KOSPI_TICKER, period)
    prices: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for key, (ticker, _label, _default) in FEATURES.items():
        try:
            df = fetch_history(ticker, period)
            if df.empty:
                errors[key] = f"{ticker}: 데이터 없음"
            else:
                prices[key] = df
        except Exception as exc:  # 네트워크/비공식 API 오류는 앱을 죽이지 않는다
            errors[key] = f"{ticker}: {exc}"
    return kospi, prices, errors


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
