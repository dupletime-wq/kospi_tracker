"""Feature engineering: 한국 개장(T일 09:00 KST) 시점에 이미 알 수 있는 정보만 사용한다.

시점 규칙
- 해외 자산: T일 이전 가장 최근 거래일의 등락률 (미국/런던 세션은 한국 T-1 마감 이후에 열려 T 개장 전에 끝남).
- 자기 시장 래그: T-1 세션까지의 값만 사용.
- 타깃: (T일 시가 / T-1일 종가 - 1) * 100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 지표 그룹. 앱/벤치마크에서 그룹 단위로 켜고 끄며 ablation 한다.
GROUPS: dict[str, list[str]] = {
    "core": ["EWY", "FLXK", "USDKRW", "SOXX", "SMH", "MU"],
    "semis_ext": ["NVDA", "TSM", "AMAT", "WDC", "STX"],
    "us_macro": ["SPY", "QQQ", "VIX", "TNX", "DXY", "EEM"],
    "asia_cmd": ["EWJ", "FXI", "WTI", "GOLD"],
    # 유럽 세션(한국 시간 16:00~익일 00:30)은 한국 마감 직후에 열려 세션내 수익률이 깨끗한 마감 후 정보
    "europe": ["DAX", "SX5E", "FTSE"],
    # 삼성전자 런던 GDR(USD, 25주=1 GDR): 삼성전자 자체를 한국 마감 후에 거래하는 직접 지표
    "kr_gdr": ["SMSN"],
}
LABELS = {
    "EWY": "EWY", "FLXK": "Franklin Korea(런던)", "USDKRW": "USD/KRW", "SOXX": "SOXX", "SMH": "SMH", "MU": "마이크론",
    "NVDA": "엔비디아", "TSM": "TSMC ADR", "AMAT": "어플라이드머티어리얼즈", "WDC": "웨스턴디지털", "STX": "씨게이트",
    "SPY": "S&P500(SPY)", "QQQ": "나스닥100(QQQ)", "VIX": "VIX", "TNX": "미 10년물 금리", "DXY": "달러인덱스",
    "DAX": "DAX", "SX5E": "유로스톡스50", "FTSE": "FTSE100", "SMSN": "삼성전자 런던 GDR",
    "EEM": "신흥국(EEM)", "EWJ": "일본(EWJ)", "FXI": "중국(FXI)", "WTI": "WTI 원유", "GOLD": "금",
}
ENGINEERED = ["EWY_local", "FLXK_local", "EWY_excess", "FLXK_excess", "SEMI_vs_MKT", "MU_vs_SEMI",
              "SMSN_local", "SMSN_excess"]
OWN = ["own_prev_ret", "own_prev_gap", "own_prev_intraday", "own_mom5", "own_mom20", "own_vol20", "own_dist_ma20",
       "kospi_prev_ret", "kospi_vol20"]
CALENDAR = ["is_mon", "is_fri", "gap_days"]
DRAM_FEATURES = ["DRAM_PURE"]

# 등락률 대신 변화량(차분)을 쓰는 자산: 금리는 %p 단위 차분(bp)
_DIFF_ASSETS = {"TNX"}


def cc_ret(df: pd.DataFrame) -> pd.Series:
    return df["Close"].pct_change() * 100


# 세션 내(시가→종가)/오버나이트(전일 종가→시가)로 분해할 자산.
# 해외 세션의 시가→종가는 한국 마감 이후 정보만 담고, 오버나이트 갭에는 한국 장중 움직임이 섞여 있다.
INTRADAY_ASSETS = ["EWY", "FLXK", "SOXX", "SMH", "MU", "NVDA", "TSM", "SPY", "QQQ", "DAX", "SX5E", "FTSE", "SMSN"]


def asset_series(prices: dict[str, pd.DataFrame], key: str, intraday: str = "none") -> dict[str, pd.Series]:
    """자산의 자체 거래일 캘린더 기준 피처 시계열. 키 -> {컬럼명: 시계열}.

    intraday: "none" 일간 수익률만 / "add" 일간 + (세션 내, 오버나이트) / "replace" 대상 자산은 (세션 내, 오버나이트)만.
    """
    df = prices[key]
    if key in _DIFF_ASSETS:
        return {key: df["Close"].diff() * 100}  # bp
    out = {key: cc_ret(df)}
    if intraday != "none" and key in INTRADAY_ASSETS:
        out[f"{key}_id"] = (df["Close"] / df["Open"] - 1) * 100
        out[f"{key}_on"] = (df["Open"] / df["Close"].shift(1) - 1) * 100
        if intraday == "replace":
            del out[key]
    if key == "VIX":
        out["VIX_lvl"] = df["Close"]  # 수준 자체도 레짐 정보
    return out


def _own_features(df: pd.DataFrame, prefix: str, ext_index: pd.DatetimeIndex) -> pd.DataFrame:
    """대상 시장 자체 래그 피처. T행의 값 = T 이전 마지막 세션까지의 정보."""
    c, o = df["Close"], df["Open"]
    r = c.pct_change() * 100
    f = pd.DataFrame({
        "prev_ret": r,
        "prev_gap": (o / c.shift(1) - 1) * 100,
        "prev_intraday": (c / o - 1) * 100,
        "mom5": (c / c.shift(5) - 1) * 100,
        "mom20": (c / c.shift(20) - 1) * 100,
        "vol20": r.rolling(20).std(),
        "dist_ma20": (c / c.rolling(20).mean() - 1) * 100,
    })
    f.columns = [f"{prefix}_{col}" for col in f.columns]
    return f.reindex(ext_index.union(f.index)).shift(1).reindex(ext_index)  # 한 세션 뒤로 밀어 T 이전 정보만


def _attach(base: pd.DataFrame, name: str, s: pd.Series) -> pd.DataFrame:
    r = s.dropna().rename(name).reset_index()
    r.columns = ["date", name]
    r["date"] = r["date"].astype("datetime64[ns]")  # 날짜 해상도가 다르면 merge_asof가 거부한다
    base = base.assign(date=base["date"].astype("datetime64[ns]"))
    return pd.merge_asof(base, r.sort_values("date"), on="date", direction="backward",
                         allow_exact_matches=False, tolerance=pd.Timedelta(days=5))


def next_kr_session(last_date: pd.Timestamp) -> pd.Timestamp:
    """마지막 KR 거래일 다음 평일 (공휴일 미반영)."""
    d = last_date + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def build_frame(target_key: str, targets: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame],
                cross: list[str], own: bool = True, engineered: bool = True, calendar: bool = True,
                extra: dict[str, pd.Series] | None = None, future: bool = False,
                intraday: str = "none") -> pd.DataFrame:
    """피처 행렬. future=True면 다음 거래일 행(y=NaN)을 마지막에 붙여 예측용 입력을 만든다.

    반환 컬럼: date, y, 피처들. 이력이 짧은 지표(extra, 상장 전 NaN)는 그대로 NaN 유지.
    """
    tdf = targets[target_key]
    idx = tdf.index
    if future:
        idx = idx.append(pd.DatetimeIndex([next_kr_session(idx[-1])]))
    base = pd.DataFrame({"date": idx})
    base["y"] = ((tdf["Open"] / tdf["Close"].shift(1) - 1) * 100).reindex(idx).to_numpy()

    for key in cross:
        if key not in prices:
            continue
        for name, s in asset_series(prices, key, intraday).items():
            base = _attach(base, name, s)
    for name, s in (extra or {}).items():
        base = _attach(base, name, s)

    kospi = targets["KOSPI"]
    kp = _own_features(kospi, "kospi", idx)
    base["kospi_prev_ret"] = kp["kospi_prev_ret"].to_numpy()
    base["kospi_vol20"] = kp["kospi_vol20"].to_numpy()
    if own:
        for col, v in _own_features(tdf, "own", idx).items():
            base[col] = v.to_numpy()
    else:
        base = base.drop(columns=["kospi_prev_ret", "kospi_vol20"])

    if engineered:
        have = base.columns
        if "EWY" in have and "USDKRW" in have:
            base["EWY_local"] = base["EWY"] + base["USDKRW"]  # 원화 환산: 원화 약세(USDKRW↑)면 달러 ETF는 그만큼 하락
            if "kospi_prev_ret" in have:
                # EWY 미국 세션 수익률은 한국 T-1 세션 움직임을 포함하므로, 그걸 뺀 마감 이후 증분
                base["EWY_excess"] = base["EWY_local"] - base["kospi_prev_ret"]
        if "FLXK" in have and "USDKRW" in have:
            base["FLXK_local"] = base["FLXK"] + base["USDKRW"]
            if "kospi_prev_ret" in have:
                base["FLXK_excess"] = base["FLXK_local"] - base["kospi_prev_ret"]
        if "SMSN" in have and "USDKRW" in have:
            base["SMSN_local"] = base["SMSN"] + base["USDKRW"]
            if "SEC" in targets:  # GDR 일간 수익률에 섞인 삼성전자 한국장 전일 움직임을 제거
                sec_prev = _own_features(targets["SEC"], "sec", idx)["sec_prev_ret"].to_numpy()
                base["SMSN_excess"] = base["SMSN_local"] - sec_prev
        if "SOXX" in have and "SPY" in have:
            base["SEMI_vs_MKT"] = base["SOXX"] - base["SPY"]
        if "MU" in have and "SOXX" in have:
            base["MU_vs_SEMI"] = base["MU"] - base["SOXX"]

    if calendar:
        d = pd.Series(idx)
        base["is_mon"] = (d.dt.weekday == 0).astype(float).to_numpy()
        base["is_fri"] = (d.dt.weekday == 4).astype(float).to_numpy()
        base["gap_days"] = d.diff().dt.days.fillna(1).to_numpy()

    return base


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("date", "y")]


def dram_pure(prices: dict[str, pd.DataFrame], peers: list[str],
              kr_holdings: dict[str, pd.DataFrame] | None = None) -> tuple[pd.Series, dict]:
    """DRAM ETF 등락률에서 구성종목의 선형 영향을 제거한 '순수 수급' 성분(잔차, %p).

    peers: 미국 상장 구성종목(MU, WDC, SNDK, STX…) 일간 수익률.
    kr_holdings: 한국 상장 구성종목(삼성전자, SK하이닉스) 일봉. 미국 D일 세션 수익률에는 한국 D일 장중
      움직임(이미 한국 종가에 반영된 정보)이 포함되므로 *같은 날짜*의 한국 종목 수익률도 함께 제거한다.
    """
    from sklearn.linear_model import LinearRegression

    peers = [p for p in peers if p in prices]
    kr = {f"KR_{k}": cc_ret(v) for k, v in (kr_holdings or {}).items()}
    if "DRAM" not in prices or not (peers or kr):
        return pd.Series(dtype=float), {}
    X = pd.concat({**{p: cc_ret(prices[p]) for p in peers}, **kr}, axis=1)
    X.index = pd.DatetimeIndex(X.index).astype("datetime64[ns]")
    y = cc_ret(prices["DRAM"]).rename("DRAM")
    y.index = pd.DatetimeIndex(y.index).astype("datetime64[ns]")
    df = pd.concat([y, X], axis=1, join="inner").dropna()
    cols = list(X.columns)
    if len(df) < 40:
        return pd.Series(dtype=float), {}
    lr = LinearRegression().fit(df[cols], df["DRAM"])
    resid = df["DRAM"] - lr.predict(df[cols])
    return resid.rename("DRAM_PURE"), {
        "r2": float(lr.score(df[cols], df["DRAM"])), "betas": dict(zip(cols, map(float, lr.coef_))), "n": len(df),
    }


def describe(col: str) -> str:
    """피처 컬럼명을 사람이 읽는 한글 라벨로."""
    fixed = {
        "VIX_lvl": "VIX 수준", "EWY_local": "EWY 원화환산(일간)", "FLXK_local": "Franklin Korea 원화환산(일간)",
        "EWY_excess": "EWY 한국장 제외 증분", "FLXK_excess": "Franklin Korea 한국장 제외 증분",
        "SEMI_vs_MKT": "반도체 − 시장(SOXX−SPY)", "MU_vs_SEMI": "마이크론 − 반도체", "SMSN_local": "삼성전자 GDR 원화환산(일간)",
        "SMSN_excess": "삼성전자 GDR 한국장 제외 증분",
        "own_prev_ret": "자기 종목 전일 수익률", "own_prev_gap": "자기 종목 전일 갭", "own_prev_intraday": "자기 종목 전일 장중",
        "own_mom5": "자기 종목 5일 모멘텀", "own_mom20": "자기 종목 20일 모멘텀", "own_vol20": "자기 종목 20일 변동성",
        "own_dist_ma20": "자기 종목 20일선 괴리", "kospi_prev_ret": "KOSPI 전일 수익률", "kospi_vol20": "KOSPI 20일 변동성",
        "is_mon": "월요일", "is_fri": "금요일", "gap_days": "휴장 간격(일)", "DRAM_PURE": "DRAM 순수 수급",
    }
    if col in fixed:
        return fixed[col]
    for suf, txt in (("_id", "세션내(시가→종가)"), ("_on", "오버나이트 갭")):
        if col.endswith(suf) and col[: -len(suf)] in LABELS:
            return f"{LABELS[col[:-len(suf)]]} {txt}"
    return f"{LABELS[col]} 일간" if col in LABELS else col


def theme_of(col: str) -> str:
    """기여도를 묶어 보여줄 테마."""
    if col.endswith("_id"):
        return "미국·런던 세션내(마감 후 정보)"
    if col.endswith("_on"):
        return "오버나이트 갭"
    if col in ENGINEERED:
        return "가공 지표(원화환산·증분)"
    if col.startswith("own_") or col.startswith("kospi_"):
        return "한국장 래그·변동성"
    if col == "DRAM_PURE":
        return "DRAM 순수 수급"
    if col in ("is_mon", "is_fri", "gap_days"):
        return "달력"
    return "해외 일간 수익률"
