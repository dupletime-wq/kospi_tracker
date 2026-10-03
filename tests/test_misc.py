import pandas as pd

from src.data_sources import parse_night_futures
from src.features import describe, next_kr_session, theme_of


def test_next_session_skips_weekend():
    assert next_kr_session(pd.Timestamp("2026-10-02")) == pd.Timestamp("2026-10-05")  # 금 -> 월
    assert next_kr_session(pd.Timestamp("2026-10-05")) == pd.Timestamp("2026-10-06")


def test_parse_night_futures():
    assert parse_night_futures("<div>아무것도 없음</div>") is None
    got = parse_night_futures('<span class="night-change">-0.45%</span>')
    assert got is not None and got.change_pct == -0.45


def test_feature_labels_cover_all_forms():
    assert describe("EWY_id") == "EWY 세션내(시가→종가)" and describe("SMSN_on") == "삼성전자 런던 GDR 오버나이트 갭"
    assert describe("own_prev_ret") == "자기 종목 전일 수익률" and describe("DRAM_PURE") == "DRAM 순수 수급"
    assert describe("MU") == "마이크론 일간"
    assert theme_of("EWY_id").startswith("미국") and theme_of("SOXX_on") == "오버나이트 갭" and theme_of("is_mon") == "달력"
