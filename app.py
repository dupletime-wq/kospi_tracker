"""KOSPI · 삼성전자 · SK하이닉스 시가 추정기 (Streamlit)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import features as F
from src import models as M
from src import pipeline as P
from src.data_sources import DRAM_PEERS, TARGETS, fetch_all, fetch_night_futures

UP, DOWN, INK, MUTED, ACCENT = "#e03131", "#1c7ed6", "#212529", "#868e96", "#3b5bdb"  # 한국 관례: 상승 빨강/하락 파랑
OPTIONAL_GROUPS = {
    "kr_gdr": "삼성전자 런던 GDR (한국 마감 후 삼성전자 직접 가격)",
    "europe": "유럽 지수 (DAX·유로스톡스·FTSE)",
    "semis_ext": "반도체 확장 (NVDA·TSM·AMAT·WDC·STX)",
    "us_macro": "미국 매크로 (SPY·QQQ·VIX·금리·달러·EEM)",
    "asia_cmd": "일본·중국·원자재 (EWJ·FXI·WTI·금)",
}
KR_HOLD = {"SEC": "삼성전자(한국 보유)", "HYNIX": "SK하이닉스(한국 보유)"}

st.set_page_config(page_title="시가 추정 · KOSPI/삼성전자/SK하이닉스", page_icon="📈", layout="wide")
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
.hero {{ background: linear-gradient(120deg,#0b1f3a 0%,#1b3a6b 55%,{ACCENT} 100%); color:#fff;
  border-radius: 20px; padding: 26px 30px; margin-bottom: 18px; }}
.hero h1 {{ font-size: 1.7rem; margin: 0 0 6px 0; color:#fff; }}
.hero p {{ margin:0; opacity:.82; font-size:.92rem; }}
.hero .pill {{ display:inline-block; background:rgba(255,255,255,.14); border-radius:999px;
  padding:3px 12px; font-size:.78rem; margin:10px 6px 0 0; }}
.card {{ background:#fff; border:1px solid #e9ecef; border-radius:18px; padding:20px 22px;
  box-shadow:0 4px 18px rgba(16,24,40,.06); height:100%; }}
.card .name {{ font-size:.95rem; color:{MUTED}; font-weight:600; letter-spacing:.2px; display:flex; justify-content:space-between; }}
.card .chip {{ font-size:.72rem; background:#f1f3f5; color:#495057; border-radius:999px; padding:2px 9px; font-weight:600; }}
.card .big {{ font-size:2.35rem; font-weight:800; line-height:1.15; margin:6px 0 2px 0; }}
.card .sub {{ font-size:.9rem; color:{INK}; }}
.card .row {{ display:flex; flex-wrap:wrap; gap:2px 10px; justify-content:space-between; font-size:.82rem; color:{MUTED};
  border-top:1px dashed #e9ecef; margin-top:12px; padding-top:10px; }}
.card .row b {{ color:{INK}; }}
.bar {{ position:relative; height:8px; border-radius:6px; background:#f1f3f5; margin-top:14px; }}
.bar i {{ position:absolute; top:0; height:8px; border-radius:6px; opacity:.35; }}
.bar em {{ position:absolute; top:-3px; width:4px; height:14px; border-radius:3px; }}
.up {{ color:{UP}; }} .down {{ color:{DOWN}; }}
.note {{ background:#fff9db; border:1px solid #ffe066; border-radius:14px; padding:12px 16px; font-size:.88rem; color:#5c4a00; }}
.note.bad {{ background:#fff5f5; border-color:#ffc9c9; color:#7a1f1f; }}
.note.ok {{ background:#ebfbee; border-color:#b2f2bb; color:#1b5e20; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #e9ecef; border-radius:14px; padding:10px 14px; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner="시장 데이터 불러오는 중…")
def load_market():
    return fetch_all()


@st.cache_data(ttl=300, show_spinner=False)
def load_night():
    return fetch_night_futures()


@st.cache_resource(ttl=1800, show_spinner="모델 학습·워크포워드 검증 중… (최초 1회, 이후 캐시)")
def run_all(cfg: P.Config, sig: tuple, _targets, _prices, _pure):
    """대상 3종목을 병렬로 분석한다. 시장 데이터가 바뀌면 sig가 달라져 캐시가 갱신된다."""
    keys = [k for k in TARGETS if k in _targets]
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lambda k: P.analyze(k, _targets, _prices, _pure, cfg), keys))
    return dict(zip(keys, res))


@st.cache_data(ttl=1800, show_spinner="feature ablation 실행 중…")
def run_ablation(cfg: P.Config, tkey: str, sig: tuple, _targets, _prices, _pure):
    return P.ablation(tkey, _targets, _prices, _pure, cfg, "ridge")


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
        st.cache_resource.clear()
        st.rerun()
    full_zoo = st.checkbox("전체 모델 비교 (Lasso·ElasticNet·RF·GBM 포함)", value=False,
                           help="약 1.5분(끄면 약 25초). 끄면 Ridge·Huber·LightGBM·하이브리드·앙상블만 비교합니다.")
    zoo = P.ALL_MODELS if full_zoo else P.FAST_MODELS
    choices = ["auto"] + [m for m in zoo if m != "baseline"]
    model_sel = st.selectbox(
        "예측에 쓸 모델", choices, index=choices.index("hybrid"),
        format_func=lambda m: "자동 (검증 MAE 최저)" if m == "auto" else M.MODEL_LABELS[m],
        help="기본값은 Ridge+LightGBM 하이브리드: 사전 실험에서 3개 대상 모두 최상위권이었습니다. "
             "'자동'은 같은 검증 구간에서 고르므로 성능이 낙관적으로 보일 수 있습니다.")
    st.markdown("**Feature 그룹** (핵심 지표는 항상 포함)")
    opt = st.multiselect("추가 그룹", list(OPTIONAL_GROUPS), default=["kr_gdr"], format_func=OPTIONAL_GROUPS.get)
    with st.expander("고급 feature 옵션"):
        intraday = st.checkbox("미국·런던 세션내/오버나이트 분해", value=True,
                               help="세션내(시가→종가)는 한국 마감 이후 정보만 담고, 오버나이트 갭에는 한국 장중 움직임이 섞여 있습니다.")
        eng = st.checkbox("가공 지표 (원화환산, 한국장 제외 증분)", value=True)
        own = st.checkbox("한국장 래그·변동성", value=True)
        cal = st.checkbox("달력 (월·금, 휴장 간격)", value=True)
    st.markdown("**DRAM ETF**")
    use_dram = st.checkbox("DRAM 순수 수급 포함", value=True)
    peer_opts = [k for k in DRAM_PEERS if k in prices] + [f"KR_{k}" for k in KR_HOLD]
    peers_sel = st.multiselect("DRAM에서 제거할 구성종목", peer_opts, default=peer_opts,
                               format_func=lambda k: KR_HOLD[k[3:]] if k.startswith("KR_") else f"{DRAM_PEERS[k][1]} ({k})")
    with st.expander("학습·검증 설정"):
        half_life = st.select_slider("최근 데이터 가중 반감기(거래일)", [0, 250, 500, 1000], value=500,
                                     help="0이면 가중 없음. 변동성 레짐이 바뀌어 최근을 더 중시하면 소폭 개선됩니다.")
        test_rows = st.slider("워크포워드 검증 기간(거래일)", 250, 750, 500, 50)
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
                      help="KOSPI 추정치와의 가중평균 비중(임의 설정, 학습 불가). 삼성전자·SK하이닉스에는 "
                           "KOSPI 갭 대비 베타만큼 같은 충격을 전달합니다.")

for key, msg in errors.items():
    st.sidebar.warning(f"{key}: 수집 실패 ({msg})")

# ---------------- 분석 실행 ----------------
us_peers = [p for p in peers_sel if not p.startswith("KR_")]
kr_peers = {p[3:]: targets[p[3:]] for p in peers_sel if p.startswith("KR_") and p[3:] in targets}
pure, dinfo = F.dram_pure(prices, us_peers, kr_peers) if use_dram else (pd.Series(dtype=float), {})
cfg = P.Config(groups=("core", *opt), intraday="add" if intraday else "none", own=own, engineered=eng, calendar=cal,
               dram=use_dram and len(pure) > 0, peers=tuple(peers_sel), half_life=half_life or None,
               test_rows=test_rows, models=zoo)
sig = (kospi.index[-1], len(kospi), tuple(sorted(peers_sel)), use_dram)
res = run_all(cfg, sig, targets, prices, pure)
if res.get("KOSPI") is None:
    st.error("학습 표본이 부족합니다. 설정을 확인해 주세요.")
    st.stop()
res = {k: v for k, v in res.items() if v is not None}
chosen = {k: (P.best_model(ta) if model_sel == "auto" else model_sel) for k, ta in res.items()}

# 야간선물 반영: KOSPI는 가중평균, 개별주는 KOSPI 갭 베타만큼 충격 전달
k_model = res["KOSPI"].preds[chosen["KOSPI"]]
k_final = P.blend_with_futures(k_model, fut_pct if use_fut else None, fut_w)
shift = {"KOSPI": k_final - k_model}
for k in res:
    if k != "KOSPI":
        shift[k] = P.gap_beta(targets[k], kospi) * (k_final - k_model)
iv = {k: P.interval(ta, chosen[k], shift[k]) for k, ta in res.items()}
final = {k: (iv[k]["pred"] if iv[k] else res[k].preds[chosen[k]] + shift[k]) for k in res}
target_date = res["KOSPI"].target_date

# ---------------- 헤더 ----------------
pills = f'<span class="pill">기준 {target_date:%Y-%m-%d (%a)} 시가</span>'
pills += f'<span class="pill">모델 {M.MODEL_LABELS[chosen["KOSPI"]] if model_sel != "auto" else "자동 선택"}</span>'
pills += f'<span class="pill">야간선물 {fut_pct:+.2f}%</span>' if use_fut else '<span class="pill">야간선물 미반영</span>'
if cfg.dram:
    pills += '<span class="pill">DRAM 순수 수급 반영</span>'
st.markdown(
    f"""<div class="hero"><h1>📈 다음 거래일 시가 추정</h1>
<p>한국장 마감 이후 해외 세션(EWY·반도체·삼성전자 GDR·DRAM 등)의 <b>세션내 움직임</b>을 분리해 KOSPI, 삼성전자, SK하이닉스의 시가 갭을 추정합니다.
워크포워드 검증으로 여러 모델을 비교합니다.</p>{pills}</div>""",
    unsafe_allow_html=True,
)


def card(k: str) -> str:
    ta, name, g = res[k], TARGETS[k][1], final[k]
    est = ta.last_close * (1 + g / 100)
    diff = est - ta.last_close
    delta = f"{diff:+,.1f}p" if k == "KOSPI" else f"{diff:+,.0f}원"
    m = M.metrics(ta.bt, chosen[k])
    bar, rng, chip = "", "구간 계산 불가", ""
    if iv[k]:
        lo, hi = iv[k]["lo"], iv[k]["hi"]
        lim = max(abs(lo), abs(hi), 1e-9) * 1.15
        pos = lambda v: 50 + 50 * v / lim  # noqa: E731
        color = UP if g >= 0 else DOWN
        bar = (f'<div class="bar"><i style="left:{pos(lo):.1f}%;width:{pos(hi) - pos(lo):.1f}%;background:{color}"></i>'
               f'<em style="left:{pos(g) - 0.5:.1f}%;background:{color}"></em></div>')
        rng = f"80% 구간 <b>{lo:+.1f} ~ {hi:+.1f}%</b>"
        chip = f'<span class="chip">상승확률 {iv[k]["p_up"]:.0%}</span>'
    return f"""<div class="card"><div class="name"><span>{name}</span>{chip}</div>
<div class="big {cls(g)}">{g:+.2f}%</div>
<div class="sub">예상 시가 <b>{fmt_px(k, est)}</b> <span class="{cls(g)}">({delta})</span></div>
{bar}
<div class="row"><span>직전 종가 <b>{fmt_px(k, ta.last_close)}</b></span><span>{rng}</span></div>
<div class="row"><span>검증 방향적중 <b>{m['방향적중']:.0%}</b></span><span>MAE <b>{m['MAE']:.2f}%p</b> (기준선 대비 −{m['MAE 개선율']:.0%})</span></div></div>"""


for c, k in zip(st.columns(len(res)), res):
    c.markdown(card(k), unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 요약·근거", "🏁 모델 비교", "🧪 Feature 분석", "🧠 DRAM 수급", "📖 방법론"])


def pick(label: str, key: str) -> str:
    who = st.radio(label, [TARGETS[k][1] for k in res], horizontal=True, key=key)
    return next(k for k in res if TARGETS[k][1] == who)


# ---------------- 요약·근거 ----------------
with tab1:
    ks = list(res)
    vals = [final[k] for k in ks]
    err = [(iv[k]["hi"] - iv[k]["lo"]) / 2 if iv[k] else 0 for k in ks]
    fig = go.Figure(go.Bar(
        x=vals, y=[f"{TARGETS[k][1]}  {v:+.2f}%" for k, v in zip(ks, vals)], orientation="h",
        marker_color=[UP if v >= 0 else DOWN for v in vals],
        error_x=dict(type="data", visible=True, color=MUTED, thickness=1.5, width=6, array=err),
        hovertemplate="%{y}<extra></extra>"))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                      xaxis_title="예상 시가 갭 (%) · 막대 끝 선 = 80% 구간", plot_bgcolor="rgba(0,0,0,0)")
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    st.plotly_chart(fig, width="stretch")

    k = pick("예측 근거 대상", "why")
    ta = res[k]
    contrib = P.explain(ta, chosen[k])
    st.markdown("##### 오늘의 예측은 어디서 나왔나 (평균적인 날 대비 기여, %p)")
    if contrib is None:
        st.info("선택한 모델은 기여도 분해를 지원하지 않습니다 (Random Forest·sklearn GBM). 선형·LightGBM·하이브리드·앙상블을 고르세요.")
    else:
        c1, c2 = st.columns([2, 3])
        theme = contrib.groupby(contrib.index.map(F.theme_of)).sum().sort_values()
        f1 = go.Figure(go.Bar(x=theme.values, y=theme.index, orientation="h",
                              marker_color=[UP if v >= 0 else DOWN for v in theme.values],
                              hovertemplate="%{y}: %{x:+.3f}%p<extra></extra>"))
        f1.update_layout(title="테마별 합계", height=330, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
        c1.plotly_chart(f1, width="stretch")
        top = contrib.head(12).iloc[::-1]
        f2 = go.Figure(go.Bar(x=top.values, y=[F.describe(c) for c in top.index], orientation="h",
                              marker_color=[UP if v >= 0 else DOWN for v in top.values],
                              customdata=[[float(ta.x_new[c]) if np.isfinite(ta.x_new[c]) else np.nan] for c in top.index],
                              hovertemplate="%{y}<br>입력값 %{customdata[0]:+.2f} → 기여 %{x:+.3f}%p<extra></extra>"))
        f2.update_layout(title="개별 feature 상위 12", height=330, margin=dict(l=10, r=10, t=40, b=10),
                         plot_bgcolor="rgba(0,0,0,0)")
        c2.plotly_chart(f2, width="stretch")
        st.caption("선형 모델은 계수×표준화 입력, LightGBM은 SHAP(pred_contrib), 앙상블/하이브리드는 구성 모델의 평균/합입니다. "
                   "야간선물 반영분은 포함되어 있지 않습니다.")
    st.markdown("##### 모델별 오늘 추정 (모델 간 불일치 = 불확실성의 단서)")
    rows = [{"모델": M.MODEL_LABELS[m], "오늘 추정(%)": round(v, 2),
             "검증 MAE(%p)": round(float(ta.board.loc[M.MODEL_LABELS[m], "MAE"]), 3)}
            for m, v in ta.preds.items() if m != "baseline"]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------- 모델 비교 ----------------
with tab2:
    k = pick("대상", "cmp")
    ta = res[k]
    b = ta.board.copy()
    b.insert(0, "순위", range(1, len(b) + 1))
    st.markdown(f"##### 워크포워드 리더보드 · 최근 {len(ta.bt)}거래일 · 훈련은 항상 과거 데이터만 사용")
    st.dataframe(b.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAE 개선율": "{:.1%}", "방향적중": "{:.1%}",
                                 "상관": "{:.3f}", "t(vs 기준선)": "{:.1f}"}, na_rep="—"), width="stretch")
    st.caption("MAE 개선율 = 1 − MAE/기준선 MAE (기준선은 과거 중앙값 예측). t는 절대오차 차이의 Newey-West t값(Diebold-Mariano 근사)으로 "
               "기준선 대비 유의성입니다. 상위 모델끼리의 차이는 대개 통계적으로 유의하지 않으니 순위를 과신하지 마세요.")
    labels = list(ta.board.index)
    maes = ta.board["MAE"].tolist()
    f = go.Figure(go.Bar(x=maes, y=labels, orientation="h",
                         marker_color=[ACCENT if l == M.MODEL_LABELS[chosen[k]] else "#ced4da" for l in labels],
                         hovertemplate="%{y}: MAE %{x:.3f}%p<extra></extra>"))
    f.update_layout(height=60 + 34 * len(labels), margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                    xaxis_title="MAE (%p, 낮을수록 좋음) · 파랑 = 현재 선택", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f, width="stretch")
    st.markdown(f"##### 실제 갭 vs 예측 — {M.MODEL_LABELS[chosen[k]]}")
    fig = go.Figure()
    fig.add_bar(x=ta.bt["date"], y=ta.bt["y"], name="실제 갭", marker_color="#ced4da")
    fig.add_scatter(x=ta.bt["date"], y=ta.bt[chosen[k]], name="예측", mode="lines", line=dict(color=ACCENT, width=2))
    fig.update_layout(yaxis_title="시가 갭 (%)", height=360, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h"), plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    if not full_zoo:
        st.info("사이드바에서 '전체 모델 비교'를 켜면 Lasso·ElasticNet·Random Forest·sklearn GBM도 같은 조건으로 비교합니다.")

# ---------------- Feature 분석 ----------------
with tab3:
    k = pick("대상", "feat")
    ta = res[k]
    st.markdown("##### Feature 묶음 ablation (Ridge, 같은 워크포워드 구간)")
    st.caption("묶음을 하나씩 빼거나 더했을 때 MAE 변화입니다. Δ<0이면 개선. 일부 그룹은 추가해도 도움이 안 되어 기본값에서 제외했습니다.")
    if st.button("ablation 실행 (약 5~10초)", key="abl_btn"):
        st.session_state["abl"] = (k, run_ablation(cfg, k, sig, targets, prices, pure))
    got = st.session_state.get("abl")
    if got and got[0] == k:
        ab = got[1]
        st.dataframe(ab.style.format({"MAE": "{:.3f}", "Δ MAE": "{:+.3f}"}), hide_index=True, width="stretch")
        sub = ab[ab["변형"] != "현재 설정"].sort_values("Δ MAE")
        f = go.Figure(go.Bar(x=sub["Δ MAE"], y=sub["변형"], orientation="h",
                             marker_color=[DOWN if v < 0 else UP for v in sub["Δ MAE"]],
                             hovertemplate="%{y}: Δ MAE %{x:+.3f}<extra></extra>"))
        f.update_layout(height=60 + 32 * len(sub), margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Δ MAE (파랑 = 개선, 빨강 = 악화)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, width="stretch")
    st.markdown("##### 모델이 학습한 전반적 중요도")
    c1, c2 = st.columns(2)
    reg = ta.fit.models["ridge"].steps[-1][1] if "ridge" in ta.fit.models else None
    if reg is not None:
        s = pd.Series(np.abs(reg.coef_), index=ta.fit.features).sort_values().tail(12)
        f = go.Figure(go.Bar(x=s.values, y=[F.describe(c) for c in s.index], orientation="h", marker_color=ACCENT))
        f.update_layout(title="Ridge |표준화 계수| 상위 12", height=360, margin=dict(l=10, r=10, t=40, b=10),
                        plot_bgcolor="rgba(0,0,0,0)")
        c1.plotly_chart(f, width="stretch")
    gb = ta.fit.models.get("lgbm")
    if gb is not None:
        imp = pd.Series(gb.booster_.feature_importance("gain"), index=ta.fit.features)
        s = (imp / imp.sum()).sort_values().tail(12)
        f = go.Figure(go.Bar(x=s.values, y=[F.describe(c) for c in s.index], orientation="h", marker_color="#868e96"))
        f.update_layout(title="LightGBM gain 중요도 상위 12", height=360, margin=dict(l=10, r=10, t=40, b=10),
                        plot_bgcolor="rgba(0,0,0,0)", xaxis_tickformat=".0%")
        c2.plotly_chart(f, width="stretch")

# ---------------- DRAM ----------------
with tab4:
    if not dinfo:
        st.warning("DRAM 순수 수급 성분을 만들 수 없습니다 (사이드바에서 DRAM 포함·구성종목을 확인하세요).")
    else:
        c1, c2 = st.columns([3, 2])
        last_d = pure.index[-1]
        raw = float(F.cc_ret(prices["DRAM"]).loc[last_d])
        parts = {}
        for key, beta in dinfo["betas"].items():
            if key.startswith("KR_"):
                r = F.cc_ret(targets[key[3:]]).get(last_d, np.nan)
                lab = KR_HOLD[key[3:]]
            else:
                r = F.cc_ret(prices[key]).get(last_d, np.nan)
                lab = DRAM_PEERS[key][1] if key in DRAM_PEERS else key
            parts[lab] = beta * (0.0 if not np.isfinite(r) else float(r))
        pure_v = float(pure.iloc[-1])
        with c1:
            fig = go.Figure(go.Waterfall(
                x=list(parts) + ["순수 수급(잔차)", "DRAM ETF 등락"], y=list(parts.values()) + [pure_v, 0],
                measure=["relative"] * (len(parts) + 1) + ["total"],
                increasing=dict(marker=dict(color=UP)), decreasing=dict(marker=dict(color=DOWN)),
                totals=dict(marker=dict(color="#495057")), connector=dict(line=dict(color=MUTED, width=1)),
                hovertemplate="%{x}: %{y:+.2f}%p<extra></extra>"))
            fig.update_layout(title=f"{last_d:%m/%d} DRAM ETF 등락 {raw:+.2f}% 분해 (%p)", height=380,
                              margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.metric("DRAM ETF 등락", f"{raw:+.2f}%")
            st.metric("구성종목으로 설명되는 부분", f"{raw - pure_v:+.2f}%p")
            st.metric("순수 수급 성분", f"{pure_v:+.2f}%p")
            st.caption(f"회귀 R² = {dinfo['r2']:.2f} (표본 {dinfo['n']}일). 한국 보유종목은 같은 날짜 수익률로 제거합니다 — "
                       "미국 세션 수익률에는 이미 한국 종가에 반영된 한국 장중 움직임이 섞여 있기 때문입니다.")

        st.markdown("##### 순수 수급 성분은 실제로 갭을 설명하나? (DRAM 상장 이후 구간, 같은 모델 DRAM 포함/제외)")
        rows = []
        start = pure.index[0]
        for k, ta in res.items():
            row = {"대상": TARGETS[k][1]}
            if ta.bt_nodram is not None and "DRAM_PURE" in ta.features:
                w = ta.bt[ta.bt["date"] >= start]
                w0 = ta.bt_nodram[ta.bt_nodram["date"] >= start]
                row["비교 표본"] = len(w)
                for m, name in (("ridge", "Ridge"), ("lgbm", "LightGBM")):
                    a = float((w["y"] - w[m]).abs().mean())
                    b0 = float((w0["y"] - w0[m]).abs().mean())
                    if m == "ridge":
                        row["Ridge MAE(포함)"], row["Ridge MAE(제외)"] = round(a, 3), round(b0, 3)
                    row[f"{name} 개선"] = round(b0 - a, 3)
                reg = ta.fit.models["ridge"].steps[-1][1]
                j = ta.fit.features.index("DRAM_PURE")
                row["Ridge 계수(표준화)"] = round(float(reg.coef_[j]), 3)
            rows.append(row)
        dfv = pd.DataFrame(rows)
        st.dataframe(dfv, hide_index=True, width="stretch")
        st.caption("개선 = (DRAM 제외 MAE) − (DRAM 포함 MAE). 양수면 DRAM이 도움, 음수면 오히려 해가 됩니다.")
        gains = [r.get("Ridge 개선", 0) for r in rows] + [r.get("LightGBM 개선", 0) for r in rows]
        helps = sum(g > 0.005 for g in gains)
        if rows and "비교 표본" in dfv:
            if helps >= 4:
                st.markdown('<div class="note ok">✅ DRAM 순수 수급을 넣으면 대부분의 비교에서 오차가 줄었습니다. 다만 상장 후 표본이 짧아 과신은 금물입니다.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="note bad">⚠️ 현재 표본(DRAM 상장 후 약 6개월)에서는 순수 수급 성분이 오차를 뚜렷하게 줄이지 못합니다. '
                            '모델에는 포함돼 있고 영향은 작게 추정됩니다. 표본이 쌓이면 달라질 수 있어 계속 추적할 가치는 있습니다.</div>',
                            unsafe_allow_html=True)

# ---------------- 방법론 ----------------
with tab5:
    st.markdown(
        """
**타깃** (T일 시가 ÷ T-1일 종가 − 1). KOSPI, 삼성전자, SK하이닉스 각각 별도 모델.

**핵심 아이디어 — 정보의 시점 분리.** 미국·런던 자산의 일간 수익률에는 이미 알려진 *한국 장중 움직임*이 섞여 있습니다.
세션내(시가→종가) 수익률은 한국 마감 이후 정보만 담고, 오버나이트 갭에는 한국 장중 움직임이 섞입니다. 둘을 분리하고,
EWY·Franklin Korea·삼성전자 GDR에서는 한국장 전일 수익률을 뺀 *증분*을 가공 지표로 만듭니다. 사전 실험에서 이 분해가 가장 큰 개선(MAE 개선율 약 +7%p)이었습니다.

**피처 그룹.** 핵심(EWY·Franklin·환율·SOXX·SMH·MU) + 한국장 래그·변동성 + 가공 지표 + 달력 + 삼성전자 런던 GDR(+DRAM 순수 수급).
반도체 확장·미국 매크로·일본/중국/원자재·유럽 지수는 추가해도 일관된 개선이 없어 기본값에서 제외했고, 사이드바에서 켜 볼 수 있습니다.

**모델.** Ridge, Lasso, ElasticNet, Huber, Random Forest, sklearn GBM, LightGBM, Ridge+LightGBM 잔차 하이브리드, 평균 앙상블.
훈련 시 최근 데이터에 지수 감쇠 가중(반감기 500일). 이상값 윈저라이즈와 변동성 가중(WLS)은 실험에서 이득이 없어 끕니다.
이 데이터는 신호가 대부분 선형이고 표본이 작아, 트리 모델 단독은 선형 모델보다 약하고 선형이 잡은 뒤 잔차를 보정하는 하이브리드가 가장 안정적이었습니다.

**검증.** 확장 윈도우 워크포워드(미래 누수 없음, 25일마다 재학습). 피처 시점 정렬은 단위 테스트로 검증합니다.
예측 구간은 변동성(20일)으로 정규화한 워크포워드 잔차의 분위수로 만들고, 상승확률은 그 분포에서 계산합니다.

**한계.**
- 야간선물은 이력이 없어 학습에 쓰지 못하고 가중평균으로만 반영합니다(비중은 임의). 자동 수집은 esignal의 웹소켓 인증 때문에 보통 실패합니다.
- 한국 공휴일은 반영하지 않습니다. yfinance는 비공식이라 지연·누락이 있을 수 있습니다.
- 검증 구간의 최고 모델을 고르면 성능이 낙관적으로 보입니다. 모델 간 차이는 대개 통계적으로 유의하지 않습니다.
- 통계적 참고용이며 투자 조언이 아닙니다.
"""
    )
