"""전체 모델·feature ablation 벤치마크를 라이브 데이터로 재현하고 Markdown 으로 출력한다.

    python scripts/benchmark.py > docs/BENCHMARK.md
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features as F  # noqa: E402
from src import models as M  # noqa: E402
from src import pipeline as P  # noqa: E402
from src.data_sources import TARGETS, fetch_all  # noqa: E402


def md(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [fmt.get(c, "{}").format(row[c]) if pd.notna(row[c]) else "—" for c in cols]
        out.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(out)


def main() -> None:
    mk = fetch_all()
    targets, prices = mk.targets, mk.prices
    pure, _ = F.dram_pure(prices, ["MU", "WDC", "SNDK", "STX"], {k: targets[k] for k in ("SEC", "HYNIX")})
    cfg = P.Config(models=P.ALL_MODELS)
    keys = list(TARGETS)
    with ThreadPoolExecutor(3) as ex:
        res = dict(zip(keys, ex.map(lambda k: P.analyze(k, targets, prices, pure, cfg), keys)))

    last = targets["KOSPI"].index[-1].date()
    first = res["KOSPI"].bt["date"].iloc[0].date()
    print(f"# 벤치마크 ({date.today()} 실행, 데이터 ~{last})\n")
    print(f"- 워크포워드: 확장 윈도우, 최근 {cfg.test_rows}거래일({first}~{last}), {cfg.refit_every}일마다 재학습, 훈련은 항상 과거만 사용")
    print(f"- 설정: 그룹 {cfg.groups}, 세션내 분해 '{cfg.intraday}', 반감기 {cfg.half_life}일, DRAM 순수 수급 포함")
    print("- MAE 개선율 = 1 − MAE/기준선(과거 중앙값 예측) MAE. t = Newey-West t (기준선 대비 절대오차 차이)")
    print("- 모델 간 순위 차이는 대부분 통계적으로 유의하지 않다. 최상위를 골랐다는 사실 자체가 낙관 편향을 만든다.\n")
    fmt = {"MAE": "{:.3f}", "RMSE": "{:.3f}", "MAE 개선율": "{:.1%}", "방향적중": "{:.1%}", "상관": "{:.3f}",
           "t(vs 기준선)": "{:.1f}"}
    for k, ta in res.items():
        print(f"## {TARGETS[k][1]} — 모델 리더보드\n")
        print(md(ta.board, fmt) + "\n")
        print(f"오늘({ta.target_date.date()}) 모델별 추정(%): " + ", ".join(
            f"{M.MODEL_LABELS[m]} {v:+.2f}" for m, v in ta.preds.items() if m != "baseline") + "\n")
    for k in keys:
        print(f"## {TARGETS[k][1]} — feature 묶음 ablation (Ridge)\n")
        ab = P.ablation(k, targets, prices, pure, cfg, "ridge").set_index("변형")
        print(md(ab, {"MAE": "{:.3f}", "Δ MAE": "{:+.3f}"}) + "\n")
    print("## DRAM 순수 수급 — 상장 이후 구간 포함/제외 비교 (MAE)\n")
    rows = {}
    start = pure.index[0]
    for k, ta in res.items():
        if ta.bt_nodram is None:
            continue
        w, w0 = ta.bt[ta.bt["date"] >= start], ta.bt_nodram[ta.bt_nodram["date"] >= start]
        row = {"표본": len(w)}
        for m in ("ridge", "lgbm"):
            row[f"{m} 포함"] = float((w["y"] - w[m]).abs().mean())
            row[f"{m} 제외"] = float((w0["y"] - w0[m]).abs().mean())
        rows[TARGETS[k][1]] = row
    print(md(pd.DataFrame(rows).T, {c: "{:.3f}" for c in ("ridge 포함", "ridge 제외", "lgbm 포함", "lgbm 제외")}))


if __name__ == "__main__":
    main()
