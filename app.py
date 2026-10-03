"""KOSPI · 삼성전자 · SK하이닉스 시가 추정기 (Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_sources import DRAM_PEERS, FEATURES, TARGETS, fetch_all, fetch_night_futures
from src.model import (
    blend_with_futures, build_dataset, dram_pure, fit_staged, gap_beta, latest_extra, latest_features,
    next_kr_session, pct_change, walk_forward,
)

UP, DOWN, INK, MUTED = "#e03131", "#1c7ed6", "#212529", "#868e96"  # 한국 증시 관례: 상승 빨강 / 하락 파랑
EXTRA_LABELS = {"DRAM_PURE": "DRAM 순수 수급 (구성종목 제거)", "DRAM_RAW": "DRAM 원지표"}

st.set_page_config(page_title="시가 추정 · KOSPI/삼성전자/SK하이닉스", page_icon="📈", layout="wide")

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
.hero {{ background: linear-gradient(120deg,#0b1f3a 0%,#1b3a6b 55%,#3b5bdb 100%); color:#fff;
  border-radius: 20px; padding: 26px 30px; margin-bottom: 18px; }}
.hero h1 {{ font-size: 1.7rem; margin: 0 0 6px 0; color:#fff; }}
.hero p {{ margin:0; opacity:.82; font-size:.92rem; }}
.hero .pill {{ display:inline-block; background:rgba(255,255,255,.14); border-radius:999px;
  padding:3px 12px; font-size:.78rem; margin:10px 6px 0 0; }}
.card {{ background:#fff; border:1px solid #e9ecef; border-radius:18px; padding:20px 22px;
  box-shadow:0 4px 18px rgba(16,24,40,.06); height:100%; }}
.card .name {{ font-size:.95rem; color:{MUTED}; font-weight:600; letter-spacing:.2px; }}
.card .big {{ font-size:2.35rem; font-weight:800; line-height:1.15; margin:6px 0 2px 0; }}
.card .sub {{ font-size:.9rem; color:{INK}; }}
.card .row {{ display:flex; flex-wrap:wrap; gap:2px 10px; justify-content:space-between; font-size:.82rem; color:{MUTED};
  border-top:1px dashed #e9ecef; margin-top:12px; padding-top:10px; }}
.card .row b {{ color:{INK}; }}
.bar {{ position:relative; height:8px; border-radius:6px; background:#f1f3f5; margin-top:14px; }}
.bar i {{ position:absolute; top:0; height:8px; border-radius:6px; opacity:.35; }}
.bar em {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:3px; }}
.up {{ color:{UP}; }} .down {{ color:{DOWN}; }}
.note {{ background:#fff9db; border:1px solid #ffe066; border-radius:14px; padding:12px 16px;
  font-size:.88rem; color:#5c4a00; }}
.note.bad {{ background:#fff5f5; border-color:#ffc9c9; color:#7a1f1f; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #e9ecef; border-radius:14px; padding:10px 14px; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner="시장 데이터 불러오는 중…")
def load_market():
    return fetch_all("5y")


@st.cache_data(ttl=300, show_spinner=False)
def load_night():
    return fetch_night_futures()


def cls(v: float) -> str:
    return "up" if v > 0 else "down" if v < 0 else ""


def fmt_px(key: str, v: float) -> str:
    return f"{v:,.1f}" if key == "KOSPI" else f"{v:,.0f}원"


targets, prices, errors = load_market()
if "KOSPI" not in targets:
    st.error("KOSPI 데이터를 불러오지 못했습니다. 잠시 후 새로고침해 주세요.")
    st.stop()
kospi = targets["KOSPI"]

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("🔄 데이터 새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    avail = [k for k in FEATURES if k in prices]
    chosen = st.multiselect("1단계 지표 (긴 이력)", avail, default=avail, format_func=lambda k: FEATURES[k][1])
    st.markdown("**DRAM ETF 반영 (2단계)**")
    peers = st.multiselect("DRAM에서 제거할 구성종목", [k for k in DRAM_PEERS if k in prices],
                           default=[k for k in DRAM_PEERS if k in prices],
                           format_func=lambda k: f"{DRAM_PEERS[k][1]} ({k})")
    extra_sel = st.multiselect("2단계 지표", list(EXTRA_LABELS), default=["DRAM_PURE"],
                               format_func=EXTRA_LABELS.get,
                               help="이력이 짧은 DRAM(상장 6개월)을 1단계 모델의 오차 설명에 씁니다.")
    years = st.slider("학습 기간(년)", 1, 5, 3)
    alpha = st.slider("Ridge 규제 강도", 0.1, 100.0, 10.0)
    test_days = st.slider("백테스트 기간(거래일)", 40, 250, 120)
    st.divider()
    st.markdown("**KOSPI 야간선물**")
    auto = load_night()
    if auto:
        st.success(f"자동 수집: {auto.change_pct:+.2f}% ({auto.source})")
    else:
        st.info("자동 수집 실패 → esignal 야간선물 등락률(%)을 직접 입력하세요.")
    use_fut = st.checkbox("야간선물 반영", value=auto is not None)
    fut_pct = st.number_input("야간선물 등락률(%)", value=float(auto.change_pct) if auto else 0.0,
                              step=0.05, format="%.2f", disabled=not use_fut)
    fut_w = st.slider("야간선물 비중", 0.0, 1.0, 0.6, 0.05, disabled=not use_fut,
                      help="KOSPI 추정치와의 가중평균 비중(임의 설정). 삼성전자·SK하이닉스는 "
                           "KOSPI 갭 대비 민감도(베타)만큼 같은 충격을 전달합니다.")

for key, msg in errors.items():
    st.sidebar.warning(f"{key}: 수집 실패 ({msg})")
if not chosen:
    st.warning("사이드바에서 1단계 지표를 하나 이상 선택해 주세요.")
    st.stop()

# ---------------- DRAM 분해 + 학습 ----------------
pure, dinfo = dram_pure(prices, peers)
extra_series: dict[str, pd.Series] = {}
if "DRAM_PURE" in extra_sel and len(pure):
    extra_series["DRAM_PURE"] = pure
if "DRAM_RAW" in extra_sel and "DRAM" in prices:
    extra_series["DRAM_RAW"] = pct_change(prices["DRAM"])
extra_keys = list(extra_series)


@st.cache_data(ttl=600, show_spinner="모델 학습·백테스트 중…")
def run(tkey, chosen, extra_keys, years, alpha, test_days, _targets, _prices, _extra, sig):
    df = _targets[tkey]
    d = build_dataset(df[df.index >= df.index[-1] - pd.DateOffset(years=years)], _prices, list(chosen), _extra)
    if len(d) < 80:
        return None
    fit = fit_staged(d, list(chosen), list(extra_keys), alpha)
    x = pd.concat([latest_features(_prices, list(chosen)), latest_extra(_extra)])
    X = pd.DataFrame([x]).reindex(columns=fit.keys)
    return dict(
        data=d, fit=fit, x=x, pred=float(fit.predict(X)[0]),
        bt=walk_forward(d, list(chosen), alpha, test_days, extra_keys=list(extra_keys)),
        bt0=walk_forward(d, list(chosen), alpha, test_days) if extra_keys else None,
        last=float(df["Close"].iloc[-1]), last_date=df.index[-1],
    )


sig = (kospi.index[-1], len(pure), tuple(peers))  # 캐시 키: 데이터가 바뀌면 재계산
res = {k: run(k, tuple(chosen), tuple(extra_keys), years, alpha, test_days, targets, prices, extra_series, sig)
       for k in TARGETS if k in targets}
if res.get("KOSPI") is None:
    st.error("학습 표본이 너무 적습니다. 지표를 줄이거나 학습 기간을 늘려 주세요.")
    st.stop()

# 야간선물 반영
k_model = res["KOSPI"]["pred"]
k_final = blend_with_futures(k_model, fut_pct if use_fut else None, fut_w)
final = {}
for k, r in res.items():
    if r is None:
        continue
    if k == "KOSPI":
        final[k] = k_final
    else:
        final[k] = r["pred"] + gap_beta(targets[k], kospi) * (k_final - k_model)
target_date = next_kr_session(kospi.index[-1])

# ---------------- 헤더 ----------------
dram_today = float(pct_change(prices["DRAM"]).dropna().iloc[-1]) if "DRAM" in prices else None
pills = f'<span class="pill">기준 {target_date:%Y-%m-%d (%a)} 시가</span>'
if dram_today is not None:
    pills += f'<span class="pill">DRAM ETF {dram_today:+.2f}%</span>'
pills += f'<span class="pill">야간선물 {fut_pct:+.2f}%</span>' if use_fut else '<span class="pill">야간선물 미반영</span>'
st.markdown(
    f"""<div class="hero"><h1>📈 다음 거래일 시가 추정</h1>
<p>한국장 마감 이후 EWY · Franklin Korea · 환율 · 반도체 ETF · DRAM ETF의 움직임으로
KOSPI, 삼성전자, SK하이닉스의 시가 갭을 추정합니다.</p>{pills}</div>""",
    unsafe_allow_html=True,
)


def card(k: str) -> str:
    r, name = res[k], TARGETS[k][1]
    g = final[k]
    est = r["last"] * (1 + g / 100)
    bt = r["bt"]
    rng_html, lo_hi = "", ""
    diff = est - r["last"]
    delta = f"{diff:+,.1f}p" if k == "KOSPI" else f"{diff:+,.0f}원"
    if bt:
        lo, hi = g - 1.28 * bt.resid_std, g + 1.28 * bt.resid_std
        lim = max(abs(lo), abs(hi), 1e-9) * 1.15
        pos = lambda v: 50 + 50 * v / lim
        color = UP if g >= 0 else DOWN
        rng_html = (f'<div class="bar"><i style="left:{pos(lo):.1f}%;width:{pos(hi) - pos(lo):.1f}%;background:{color}"></i>'
                    f'<em style="left:{pos(g) - 0.5:.1f}%;background:{color}"></em></div>')
        lo_hi = f"80% 구간 <b>{lo:+.1f} ~ {hi:+.1f}%</b>"
    return f"""<div class="card"><div class="name">{name}</div>
<div class="big {cls(g)}">{g:+.2f}%</div>
<div class="sub">예상 시가 <b>{fmt_px(k, est)}</b> <span class="{cls(g)}">({delta})</span></div>
{rng_html}
<div class="row"><span>직전 종가 <b>{fmt_px(k, r['last'])}</b></span><span>{lo_hi or '구간 계산 불가'}</span></div></div>"""


cols = st.columns(len(final))
for c, k in zip(cols, final):
    c.markdown(card(k), unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["📊 요약", "🧠 DRAM 수급 분석", "🧪 백테스트"])

# ---------------- 요약 ----------------
with tab1:
    vals = [final[k] for k in final]
    names = [f"{TARGETS[k][1]}  {v:+.2f}%" for k, v in zip(final, vals)]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker_color=[UP if v >= 0 else DOWN for v in vals],
        error_x=dict(type="data", visible=True, color=MUTED, thickness=1.5, width=6,
                     array=[1.28 * res[k]["bt"].resid_std if res[k]["bt"] else 0 for k in final]),
        hovertemplate="%{y}<extra></extra>",
    ))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="예상 시가 갭 (%) · 막대 끝 선 = 80% 구간",
                      yaxis=dict(autorange="reversed"), plot_bgcolor="rgba(0,0,0,0)")
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    st.plotly_chart(fig, width="stretch")

    who = st.radio("지표별 기여도 대상", [TARGETS[k][1] for k in final], horizontal=True)
    k = next(k for k in final if TARGETS[k][1] == who)
    r = res[k]
    coef = r["fit"].coefficients()
    rows = []
    for fk in r["fit"].keys:
        label = FEATURES[fk][1] if fk in FEATURES else EXTRA_LABELS.get(fk, fk)
        rows.append({"지표": label, "최근 등락률(%)": round(float(r["x"][fk]), 2), "계수": round(float(coef[fk]), 3),
                     "기여(%p)": round(float(coef[fk] * r["x"][fk]), 3)})
    df_c = pd.DataFrame(rows)
    st.dataframe(df_c, hide_index=True, width="stretch")
    st.caption(f"기여 합계 {df_c['기여(%p)'].sum():+.3f} + 절편 {r['fit'].intercept():+.3f} = 모델 추정 {r['pred']:+.2f}%  ·  "
               f"학습 표본 {len(r['data'])}개 ({r['data']['date'].iloc[0]:%Y-%m-%d} ~ {r['data']['date'].iloc[-1]:%Y-%m-%d})")

# ---------------- DRAM 분석 ----------------
with tab2:
    if not dinfo:
        st.warning("DRAM ETF 분해에 필요한 데이터가 부족합니다 (최소 40거래일, 구성종목 1개 이상).")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            last_d = pure.index[-1]
            raw = float(pct_change(prices["DRAM"]).loc[last_d])
            parts = {f"{DRAM_PEERS[p][1]}": dinfo["betas"][p] * float(pct_change(prices[p]).loc[last_d]) for p in dinfo["betas"]}
            pure_v = float(pure.iloc[-1])
            labels = list(parts) + ["순수 수급(잔차)"]
            vals = list(parts.values()) + [pure_v]
            fig = go.Figure(go.Waterfall(
                x=labels + ["DRAM ETF 등락"], y=vals + [0], measure=["relative"] * len(vals) + ["total"],
                increasing=dict(marker=dict(color=UP)), decreasing=dict(marker=dict(color=DOWN)),
                totals=dict(marker=dict(color="#495057")), connector=dict(line=dict(color=MUTED, width=1)),
                hovertemplate="%{x}: %{y:+.2f}%p<extra></extra>",
            ))
            fig.update_layout(title=f"{last_d:%m/%d} DRAM ETF 등락 {raw:+.2f}% 분해 (%p)", height=360,
                              margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.metric("DRAM ETF 등락", f"{raw:+.2f}%")
            st.metric("구성종목으로 설명되는 부분", f"{raw - pure_v:+.2f}%p")
            st.metric("순수 수급 성분", f"{pure_v:+.2f}%p")
            st.caption(f"구성종목 회귀 R² = {dinfo['r2']:.2f} (표본 {dinfo['n']}일). "
                       "베타: " + ", ".join(f"{DRAM_PEERS[p][1]} {b:.2f}" for p, b in dinfo["betas"].items()))

        st.markdown("##### 순수 수급 성분은 실제로 갭을 설명하나?")
        rows = []
        for k, r in res.items():
            if r is None:
                continue
            sub = r["data"].dropna(subset=["DRAM_PURE"]) if "DRAM_PURE" in r["data"] else pd.DataFrame()
            row = {"대상": TARGETS[k][1], "겹치는 표본": len(sub)}
            if len(sub) > 30:
                resid = sub["y"] - r["fit"].base.predict(sub)
                cr = float(sub["DRAM_PURE"].corr(resid))
                row["1단계 오차와 상관"] = round(cr, 2)
                row["t값"] = round(cr * np.sqrt((len(sub) - 2) / (1 - cr**2)), 1)
            if r["bt"] and r["bt0"]:
                row["MAE (DRAM 포함)"] = round(r["bt"].mae, 3)
                row["MAE (DRAM 제외)"] = round(r["bt0"].mae, 3)
                row["개선(%p)"] = round(r["bt0"].mae - r["bt"].mae, 3)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        better = [row for row in rows if row.get("개선(%p)", 0) > 0]
        bad = len(better) == 0
        st.markdown(
            f'<div class="note {"bad" if bad else ""}">{"⚠️ " if bad else "✅ "}'
            + ("현재 표본(DRAM 상장 후 약 6개월)에서는 순수 수급 성분이 1단계 오차를 유의하게 줄이지 못합니다 "
               "(|t| &lt; 2, 백테스트 MAE 개선 없음). 모델에는 포함돼 있지만 영향이 작게 추정됩니다. 표본이 쌓이면 달라질 수 있습니다."
               if bad else "일부 대상에서 DRAM 포함 시 백테스트 MAE가 개선됩니다. 다만 표본이 짧아 과신은 금물입니다.")
            + "</div>", unsafe_allow_html=True)
        st.caption("DRAM 원지표는 갭과 상관이 높지만, 그 정보 대부분이 MU·SOXX·SMH 등 1단계 지표와 겹칩니다. "
                   "구성종목 베타는 DRAM 전체 이력으로 한 번 추정해 백테스트에 약간의 사후 정보(look-ahead)가 포함됩니다.")

# ---------------- 백테스트 ----------------
with tab3:
    who = st.radio("대상", [TARGETS[k][1] for k in final], horizontal=True, key="bt_who")
    k = next(k for k in final if TARGETS[k][1] == who)
    bt = res[k]["bt"]
    if not bt:
        st.warning("백테스트에 필요한 표본이 부족합니다.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{bt.mae:.2f}%p", f"기준선(0% 예측) {bt.mae_baseline:.2f}", delta_color="off")
        m2.metric("방향 적중률", f"{bt.hit_rate:.0%}")
        m3.metric("상관계수", f"{bt.corr:.2f}")
        m4.metric("검증 구간", f"{len(bt.frame)}일")
        fig = go.Figure()
        fig.add_bar(x=bt.frame["date"], y=bt.frame["actual"], name="실제 갭", marker_color="#ced4da")
        fig.add_scatter(x=bt.frame["date"], y=bt.frame["pred"], name="예측", mode="lines",
                        line=dict(color="#3b5bdb", width=2))
        fig.update_layout(yaxis_title="시가 갭 (%)", height=380, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h"), plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

with st.expander("방법론과 한계"):
    st.markdown(
        """
- **타깃**: (T일 시가 ÷ T-1일 종가 − 1). KOSPI, 삼성전자(005930.KS), SK하이닉스(000660.KS) 각각 별도 모델.
- **지표**: T일 이전 가장 최근 해외 거래일의 등락률 (한국 마감 이후에 열린 세션). 환율은 24시간 호가라 근사입니다.
- **2단계 모델**: 이력이 긴 지표(EWY, FLXK, 환율, SOXX, SMH, MU)로 먼저 Ridge 학습 → DRAM ETF(상장 6개월)는 그 오차를 설명하는 2단계 지표로 사용.
- **DRAM 순수 수급**: DRAM ETF 등락률을 MU·WDC·SNDK·STX 등락률에 회귀해 남은 잔차.
- **야간선물**: 이력이 없어 학습에 쓰지 못하고 KOSPI 추정치와 가중평균(비중은 사용자 지정)으로만 반영, 개별주는 KOSPI 갭 베타만큼 전달합니다.
- 한국 공휴일 미반영, yfinance(비공식) 지연·누락 가능. 투자 조언이 아닌 참고용입니다.
"""
    )
