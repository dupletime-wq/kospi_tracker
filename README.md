# KOSPI 시가 추정기

한국장 마감 이후 **EWY, Franklin FTSE Korea UCITS ETF(런던), USD/KRW, SOXX·SMH·MU(반도체), DRAM ETF, KOSPI 야간선물**의 움직임으로
다음 거래일 KOSPI **시가 갭(전일 종가 대비 %)** 을 Ridge 회귀로 추정하는 Streamlit 앱입니다. 투자 조언이 아닙니다.

## 구성
| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit 한국어 UI |
| `src/data_sources.py` | yfinance 가격 수집, esignal 야간선물 자동 수집 시도 |
| `src/model.py` | 데이터 정렬(미래 데이터 누수 방지), Ridge 학습, 워크포워드 백테스트 |
| `tests/` | 모델·파서 단위 테스트 (`pip install -r requirements-dev.txt && pytest`) |

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## streamlit.app(Community Cloud) 배포
이 저장소는 배포용 설정(`requirements.txt`, `runtime.txt`, `.streamlit/config.toml`)을 갖춘 상태입니다. 앱 생성은 본인 계정 로그인이 필요합니다.
1. https://share.streamlit.io 에 GitHub 계정으로 로그인하고, 저장소(`dupletime-wq/kospi_tracker`) 접근을 허용합니다.
2. **Create app → Deploy a public app from GitHub** 선택.
3. Repository: `dupletime-wq/kospi_tracker`, Branch: 배포할 브랜치(PR 병합 후 `main` 권장), Main file path: `app.py`.
4. Deploy. 배포 후 코드를 push하면 자동 반영됩니다.

## 데이터와 한계
- **DRAM ETF**는 2026-04 상장이라 이력이 약 6개월입니다. 기본값은 학습 제외(현황 표에는 표시)이며, 사이드바에서 켜면 학습 표본이 ~120개로 줄어듭니다.
- **야간선물(esignal.co.kr)**: 실시간 값이 웹소켓(세션 티켓 인증)으로 전달돼 자동 수집이 보통 실패합니다. 인증 우회는 하지 않으며, 실패하면 사이드바에 **수동 입력**합니다. 이력이 없어 학습에는 쓰지 않고 모델 추정치와 가중평균(비중은 사용자가 지정)으로만 반영합니다.
- 환율(`KRW=X`)은 24시간 호가라 한국 마감 시점과 정확히 맞지 않는 근사입니다.
- 한국 공휴일은 반영하지 않습니다. yfinance는 비공식 API라 지연·누락이 있을 수 있습니다.
