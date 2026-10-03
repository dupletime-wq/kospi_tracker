"""KOSPI 시가 갭 추정기 (Streamlit)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_sources import FEATURES, fetch_all, fetch_night_futures
from src.model import (
    blend_with_futures, build_dataset, fit_model, latest_features, next_kr_session, pct_change,
    walk_forward,
)

st.set_page_config(page_title="KOSPI 시가 추정", page_icon="📈", layout="wide")


@st.cache_data(ttl=600, show_spinner="시장 데이터 불러오는 중…")
def load_market():
    return fetch_all("5y")


@st.cache_data(ttl=300, show_spinner=False)
def load_night():
    return fetch_night_futures()


st.title("📈 KOSPI 다음 거래일 시가 추정")
st.caption("한국장 마감 이후 해외·야간 지표의 움직임으로 시가 갭(전일 종가 대비)을 통계적으로 추정합니다. "
           "투자 조언이 아닌 참고용입니다.")

kospi, prices, errors = load_market()
if kospi.empty:
    st.error("KOSPI 데이터를 불러오지 못했습니다. 잠시 후 새로고침해 주세요.")
    st.stop()

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("설정")
    if st.button("🔄 데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()
    avail = [k for k in FEATURES if k in prices]
    chosen = st.multiselect(
        "모델에 사용할 지표", avail,
        default=[k for k in avail if FEATURES[k][2]],
        format_func=lambda k: FEATURES[k][1],
        help="DRAM ETF는 상장 6개월 내외라 켜면 학습 표본이 크게 줄어듭니다.",
    )
    years = st.slider("학습 기간(년)", 1, 5, 3)
    alpha = st.slider("Ridge 규제 강도", 0.1, 100.0, 10.0, help="클수록 계수가 보수적으로 줄어듭니다.")
    test_days = st.slider("백테스트 기간(거래일)", 40, 250, 120)
    st.divider()
    st.subheader("KOSPI 야간선물")
    auto = load_night()
    if auto:
        st.success(f"자동 수집 성공: {auto.change_pct:+.2f}% ({auto.source})")
    else:
        st.info("자동 수집 실패 → 직접 입력해 주세요. "
                "esignal 야간선물 화면의 등락률(%)을 입력하면 됩니다.")
    use_fut = st.checkbox("야간선물 반영", value=auto is not None)
    fut_pct = st.number_input("야간선물 등락률(%)", value=float(auto.change_pct) if auto else 0.0,
                              step=0.05, format="%.2f", disabled=not use_fut)
    fut_w = st.slider("야간선물 비중", 0.0, 1.0, 0.6, 0.05, disabled=not use_fut,
                      help="야간선물은 과거 이력을 구하기 어려워 학습에 쓰지 못합니다. "
                           "모델 추정치와의 가중평균 비중을 직접 정합니다.")

for key, msg in errors.items():
    st.sidebar.warning(f"{FEATURES[key][1]}: 수집 실패 ({msg})")

if not chosen:
    st.warning("사이드바에서 지표를 하나 이상 선택해 주세요.")
    st.stop()

# ---------------- 학습 ----------------
cutoff = kospi.index[-1] - pd.DateOffset(years=years)
data = build_dataset(kospi[kospi.index >= cutoff], prices, chosen)
if len(data) < 80:
    st.error(f"학습 표본이 {len(data)}개로 너무 적습니다. 지표를 줄이거나 학습 기간을 늘려 주세요.")
    st.stop()

fit = fit_model(data, chosen, alpha)
x_now = latest_features(prices, chosen)
model_pct = float(fit.predict(x_now)[0])
bt = walk_forward(data, chosen, alpha, test_days)
final_pct = blend_with_futures(model_pct, fut_pct if use_fut else None, fut_w)

last_close = float(kospi["Close"].iloc[-1])
target_date = next_kr_session(kospi.index[-1])
est_open = last_close * (1 + final_pct / 100)

# ---------------- 결과 ----------------
st.subheader(f"{target_date:%Y-%m-%d (%a)} 시가 추정")
c1, c2, c3, c4 = st.columns(4)
c1.metric("예상 시가 등락률", f"{final_pct:+.2f}%")
c2.metric("예상 KOSPI 시가", f"{est_open:,.1f}", f"{est_open - last_close:+,.1f}p")
c3.metric("직전 종가", f"{last_close:,.1f}", f"{kospi.index[-1]:%m/%d}", delta_color="off")
c4.metric("모델만의 추정", f"{model_pct:+.2f}%")
if bt:
    lo, hi = final_pct - 1.28 * bt.resid_std, final_pct + 1.28 * bt.resid_std
    st.info(f"**80% 예측 구간**: {lo:+.2f}% ~ {hi:+.2f}%  (백테스트 오차 표준편차 {bt.resid_std:.2f}%p 기준 근사)")
if use_fut:
    st.caption(f"야간선물 {fut_pct:+.2f}% × {fut_w:.0%} + 모델 {model_pct:+.2f}% × {1 - fut_w:.0%}")

st.subheader("지표 현황과 기여도")
coef = fit.coefficients()
rows = []
for k in FEATURES:
    if k in prices:
        s = pct_change(prices[k]).dropna()
        rows.append({
            "지표": FEATURES[k][1], "기준일": f"{prices[k].index[-1]:%m/%d}",
            "종가": round(float(prices[k]["Close"].iloc[-1]), 2), "등락률(%)": round(float(s.iloc[-1]), 2),
            "모델 사용": "✅" if k in chosen else "—",
            "계수": round(float(coef[k]), 3) if k in chosen else None,
            "기여(%p)": round(float(coef[k] * x_now[k]), 3) if k in chosen else None,
        })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(f"기여(%p) 합계 + 절편 {fit.model.intercept_:+.3f} = 모델 추정치 (표준화 전 단위로 환산). "
           f"학습 표본 {len(data)}개 ({data['date'].iloc[0]:%Y-%m-%d} ~ {data['date'].iloc[-1]:%Y-%m-%d})")

st.subheader("백테스트 (워크포워드)")
if bt:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{bt.mae:.2f}%p", f"기준선(0% 예측) {bt.mae_baseline:.2f}", delta_color="off")
    m2.metric("방향 적중률", f"{bt.hit_rate:.0%}")
    m3.metric("상관계수", f"{bt.corr:.2f}")
    m4.metric("검증 구간", f"{len(bt.frame)}일")
    fig = go.Figure()
    fig.add_bar(x=bt.frame["date"], y=bt.frame["actual"], name="실제 갭", opacity=0.55)
    fig.add_scatter(x=bt.frame["date"], y=bt.frame["pred"], name="예측", mode="lines+markers")
    fig.update_layout(yaxis_title="시가 갭 (%)", height=380, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
else:
    st.warning("백테스트에 필요한 표본이 부족합니다.")

with st.expander("방법론과 한계"):
    st.markdown(
        """
- **타깃**: (T일 시가 ÷ T-1일 종가 − 1). 시가는 `^KS11` 일봉의 Open.
- **지표**: T일 이전 가장 최근 해외 거래일의 종가 대비 등락률. 이 세션은 한국 마감 이후에 열리므로 마감 후 정보만 담습니다.
- **환율**(`KRW=X`)은 24시간 호가라 한국 마감 시점과 정확히 맞지 않는 근사치입니다.
- **야간선물**은 이력 확보가 어려워 학습에 쓰지 못하고 가중평균으로만 반영합니다. 비중은 직접 정하는 값이며 데이터로 검증된 값이 아닙니다.
- 한국 공휴일은 반영하지 않아, 휴장일 다음 날은 갭이 누적될 수 있습니다.
- 데이터는 yfinance(비공식)이며 지연·누락이 있을 수 있습니다.
"""
    )
