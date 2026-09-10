# timetable_app.importers

외부 학사 시스템의 강의시간표를 이 프로젝트의 `courses.csv` 형식으로 변환하는 스크립트 모음. 현재는 가천대학교(`gachon.py`) 하나만 있다.

## 이 모듈은 앱 런타임의 일부가 아니다

`cli.py`/`web.py`/`recommender.py`는 이 패키지를 **import하지 않는다**. 여기 있는 스크립트는 사람이 학기당 한 번 수동으로 실행해서 `courses.csv`를 만들어내는 **준비 단계 도구**다. 흐름은 다음과 같다.

```
[학기 시작 전, 딱 한 번]
gachon-import 실행 → courses.csv 생성 → 사람이 확인/보정(score 등 채워넣기) → git commit
                                                                              │
[학기 내내, 앱 실행할 때마다]                                                    ▼
timetable.py / app.py 실행 → courses.csv 읽음 (커밋해둔 그 파일)
```

**실시간으로 학교 서버를 호출하지 않는 이유:**
1. 이 API는 학교가 공식 문서화한 것이 아니라 브라우저 개발자도구로 역추적한 내부 엔드포인트다. 매 요청마다 의존하면 학교 쪽 개편에 앱 전체가 바로 영향받는다.
2. 학교 서버에 불필요한 트래픽을 반복해서 보내지 않기 위함이다.
3. 학교 API에는 `score`/`difficulty`/`rating`처럼 AI 학습에 필요한 값이 없다 — 받아온 데이터를 사람이 다듬을 시간이 필요하다.

## gachon.py

`info.gachon.ac.kr/Ssu1000q/mainSearch.do`(로그인 불필요, 브라우저 Network 탭으로 확인한 내부 API)를 호출해 학과별 개설 과목을 가져온다.

```bash
python -m timetable_app.importers.gachon --dept CS3120 CS2170 --year 2026 --term 20 --out courses.csv
# 또는 pip install -e . 이후:
gachon-import --dept CS3120 CS2170 --year 2026 --term 20 --out courses.csv
```

| 옵션 | 설명 |
|---|---|
| `--dept` | 학과 코드 (공백으로 여러 개 나열 가능). 같은 과목이 여러 학과 코드로 겹쳐 조회되면 학수번호 기준으로 자동 중복 제거됨 |
| `--year` | 조회 연도 |
| `--term` | 학기 코드 (예: `20` — 정확한 규칙은 아직 다른 학기로 검증 전) |
| `--univ` | 단과대학 코드 (기본값 `CS0000`) |
| `--out` | 출력 CSV 경로 (기본값 `courses.csv`) |

### 알아둘 점

- **학과 코드/단과대학 코드**를 모르면 브라우저에서 `deptList.do` 요청을 캡처해 확인해야 한다 (아직 이 모듈에 전체 학과 목록 조회 기능은 없음).
- **50분 교시 → 실제 시각 변환**: 가천대는 1교시 09:00~09:50부터 매 정시 50분 수업이다(`_period_to_range`). 같은 요일의 연속 교시(예: 수8,수9)는 쉬는시간 없이 이어지는 한 블록으로 자동 병합한다(`_parse_time_field`).
- **현재 수강인원**은 이 API 응답에 없어 항상 0으로 채워진다 (정원 대비 실시간 신청 인원 반영 불가).
- **다른 학교로 확장**하려면 같은 인터페이스(원본 레코드 dict → `courses.csv` 행)로 새 모듈(`importers/<학교>.py`)을 추가하면 된다. 이 파일 하나가 깨져도 나머지 앱(`models`/`recommender`/`persistence`)은 영향받지 않도록 격리되어 있다.

### 테스트

`tests/test_gachon_importer.py`는 실제 네트워크 요청 없이 파싱 로직(교시 병합, 학점 파싱, 학과 간 중복 제거)만 검증한다. CI에서 학교 서버에 요청을 보내지 않는다.
