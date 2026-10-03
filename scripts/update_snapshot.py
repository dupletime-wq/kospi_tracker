"""실시간 수집이 막힐 때(예: Streamlit Cloud 의 Yahoo 제한) 쓰는 데이터 스냅샷을 갱신한다.

    python scripts/update_snapshot.py   # data/snapshot.csv.gz 갱신 후 커밋
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_sources as ds  # noqa: E402
from src import perp  # noqa: E402


def update_perp() -> None:
    """삼성전자·SK하이닉스 24시간 선물 30분봉 전체 이력(Gate)을 스냅샷으로 저장한다."""
    import pandas as pd

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    series = {k: perp.fetch_gate(sym["gate"], perp.GATE_LISTING - pd.Timedelta(days=1), now) for k, sym in perp.SYMBOLS.items()}
    perp.save_snapshot(series)
    print("perp snapshot:", {k: f"{len(v)}봉 ~{v.index[-1]}" for k, v in series.items()}, "->", perp.SNAPSHOT_PATH)


def main() -> None:
    if "--perp-only" in sys.argv:
        return update_perp()
    ds.SNAPSHOT_PATH.unlink(missing_ok=True)  # 스냅샷에 기대지 않고 전체 이력을 새로 받는다
    mk = ds.fetch_all(workers=2)
    if mk.errors:
        sys.exit(f"수집 실패가 있어 스냅샷을 갱신하지 않았습니다: {mk.errors}")
    frames = {**mk.targets, **mk.prices}
    ds.save_snapshot(frames)
    update_perp()
    print(f"snapshot 저장: {len(frames)}개 종목, KOSPI 마지막 {mk.targets['KOSPI'].index[-1]:%Y-%m-%d} -> {ds.SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
