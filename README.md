# AI-Programming: AI 기반 시간표 추천 시스템

학생의 학점 범위, 선호 요일, 선호 교수, 난이도·평점 선호도를 입력하면 AI 모델(RandomForest)이 각 시간표의 점수를 예측하여 **시간 충돌 없는 최적의 시간표**를 자동으로 추천해주는 프로그램입니다. 추천된 시간표는 다크 테마의 HTML 페이지로 시각화되어 브라우저에서 바로 확인할 수 있습니다.

---

## 주요 기능

- **CSV 한 파일로 구동**: `courses.csv` 하나에 과목 정보 + 학습용 점수를 함께 담아, 이 파일만 통째로 교체하면 다른 학기/커리큘럼 데이터로 바로 전환 가능
- **과목 데이터 검증**: CSV로 과목 정보를 불러오면서 요일·시간·학점 등의 유효성을 자동 검사
- **AI 기반 점수 예측**: 과목의 기본 정보, 시간대, 담당 교수 통계, 사용자 선호도를 특성(feature)으로 추출해 RandomForestRegressor로 시간표 점수를 예측
- **백트래킹 시간표 생성**: 시간 충돌을 피하면서 학점 조건을 만족하는 모든 조합을 탐색 후 상위 5개 시간표 반환
- **시간표 시각화**: 과목별 색상이 구분된 다크 테마 HTML 시간표 자동 생성
- **시간표 저장/불러오기**: 마음에 드는 시간표를 이름 붙여 JSON으로 저장, 이후 재사용 가능
- **특성 중요도 분석**: 모델 학습 후 어떤 요소가 점수에 가장 큰 영향을 미쳤는지 `feature_importance.png`로 시각화

---

## 폴더 구조

```
AI-Programming/
├── timetable.py                 # 실행 진입점 (timetable_app.cli.main 호출)
├── timetable_app/                # 실제 구현 패키지
│   ├── models.py                  # TimeSlot, Course, UserPreferences, 도메인 상수
│   ├── features.py                # 특성(feature) 추출 함수
│   ├── recommender.py             # ScheduleRecommender: 학습/예측/탐색
│   ├── html_report.py             # 시간표 HTML 렌더링
│   ├── persistence.py             # 과목 CSV 입력 + 저장된 시간표 JSON 입출력
│   └── cli.py                     # 대화형 입력 및 main()
├── courses.csv                   # 과목 정보 + 학습용 점수 (이 파일만 갈아 끼우면 됨)
├── schedules/                     # 저장된 시간표 (자동 생성됨)
├── recommended_timetable.html     # 추천 결과 (실행 후 생성됨)
├── feature_importance.png         # 특성 중요도 그래프 (실행 후 생성됨)
└── timetable.log                  # 실행 로그 (자동 생성됨)
```

---

## 요구사항

- Python 3.8 이상
- 필요 라이브러리 (`requirements.txt`):

```bash
pip install -r requirements.txt
```

---

## 설치 및 실행 방법

### 1. 레포 클론

```bash
git clone https://github.com/code5312/AI-Programming.git
cd AI-Programming
```

### 2. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 3. 실행

```bash
python timetable.py
```

### 4. (선택) 테스트 실행

`tests/`에 표준 라이브러리 `unittest` 기반 테스트가 있습니다 (추가 설치 불필요). 데이터 검증 규칙, `courses.csv` 파싱, 시간표 탐색, HTML 렌더링, 그리고 이번에 실제로 고쳤던 버그들의 회귀 테스트를 포함합니다.

```bash
python -m unittest discover -s tests -t .
```

---

## 사용 방법

프로그램을 실행하면 터미널에서 아래 순서로 선호도를 입력받습니다.

1. **최소/최대 학점** 입력 (예: 최소 9, 최대 18)
2. **선호 요일** 선택 (0=월, 1=화, 2=수, 3=목, 4=금 / 없으면 `n`)
3. **선호 교수** 입력 (쉼표로 구분 / 없으면 `n`)
4. **제외할 과목 코드** 입력 (쉼표로 구분 / 없으면 `n`)
5. **선호 난이도** (0~1 사이, 기본값 0.5)
6. **선호 평점** (0~5 사이, 기본값 3.0)

입력이 끝나면:

- `courses.csv`에 `score` 값이 있는 과목이 있으면 AI 모델을 자동 학습, 없으면 규칙 기반 점수로 진행
- 조건에 맞는 시간표 중 예측 점수가 가장 높은 시간표를 선택
- `recommended_timetable.html` 파일을 생성하고 자동으로 브라우저에서 열림
- 저장 여부를 물어보며, `y` 입력 시 `schedules/이름.json`으로 저장

---

## 데이터 파일 형식

### `courses.csv` (과목 정보 + 학습용 점수, 이 파일만 교체하면 됨)

**긴 형식(long format)**: 한 과목이 여러 시간에 수업하면 같은 `code`로 행을 여러 번 반복해서 적습니다 (엑셀에서 행을 복사해 `day`/`start_time`/`end_time`만 바꾸면 됨).

| 컬럼 | 설명 | 필수 |
|---|---|---|
| `code` | 과목 코드 (앞자리 0이 있으면 문자열로 취급되도록 텍스트 서식으로 입력 권장) | ✅ |
| `name` | 과목명 | ✅ |
| `professor` | 담당 교수 | ✅ |
| `credits` | 학점 (1~3) | ✅ |
| `classroom` | 강의실 | ✅ |
| `capacity` | 정원 | ✅ |
| `day` | 요일 (0=월 ~ 4=금) | ✅ |
| `start_time` / `end_time` | 수업 시작/종료 시간 (`HH:MM`, 09:00~18:00 사이, 1~4시간) | ✅ |
| `current_enrolled` | 현재 수강인원 | 선택 (기본 0) |
| `difficulty` | 난이도 0~1 | 선택 (기본 0.5) |
| `rating` | 평점 0~5 | 선택 (기본 3.0) |
| `prerequisites` | 선수과목 코드 목록 (세미콜론 `;`으로 구분, 예: `CODE1;CODE2`) | 선택 |
| `score` | 그 과목의 실제/가상 만족도 점수. 값이 있는 과목만 모델 학습에 쓰이고, 한 과목의 여러 행 중 첫 값만 사용됩니다 | 선택 |

같은 `code`를 가진 행들의 `name`/`professor`/`credits`/`classroom`/`capacity`/`difficulty`/`rating`/`score`는 모두 같은 값이어야 하며(수업 시간만 행마다 다름), 첫 번째 행의 값이 그 과목의 값으로 사용됩니다. `score`가 채워진 과목이 2개 미만이면 모델 학습은 건너뛰고 규칙 기반 점수로만 추천합니다.

과목 후보(제외 과목을 뺀 나머지)가 22개를 넘으면 전수 백트래킹 대신 무작위 다중 시작 그리디 탐색으로 자동 전환되어, `courses.csv`에 과목이 아무리 많아도 응답 시간이 일정 수준으로 유지됩니다 (22개 이하는 항상 최적해를 보장하는 전수 탐색 사용).

---

## 주요 모듈/클래스 구조

| 모듈 | 클래스/함수 | 역할 |
|---|---|---|
| `timetable_app.models` | `TimeSlot`, `Course`, `UserPreferences` | 데이터 검증 및 도메인 규칙(수업 가능 시간대, 학점 범위 등) |
| `timetable_app.features` | `extract_all_features` 등 | 과목 → 모델 입력 특성(feature) 변환 |
| `timetable_app.recommender` | `ScheduleRecommender` | 과목 카탈로그 관리, 모델 학습/예측, 시간표 탐색 |
| `timetable_app.html_report` | `generate_html` | 추천 시간표를 다크 테마 HTML로 렌더링 |
| `timetable_app.persistence` | `load_courses_from_csv`, `save_schedule_json` 등 | 과목 CSV 입력 + 저장된 시간표 JSON 입출력 |
| `timetable_app.cli` | `main`, `get_user_preferences` | 대화형 입력 및 실행 진입점 |

`ScheduleRecommender(data_dir=...)`처럼 `data_dir`을 지정하면 저장된 시간표(`schedules/`)와 특성 중요도 그래프(`feature_importance.png`)를 해당 디렉터리 아래에 쓴다. 지정하지 않으면 기존과 동일하게 프로젝트 루트를 쓰므로 CLI 사용에는 영향이 없다. 여러 사용자가 동시에 쓰는 환경(예: 웹 서비스)에서 사용자/세션별 디렉터리를 넘겨 산출물을 격리하는 용도.

---

## 문제 해결(Trouble Shooting)

- **`FileNotFoundError`**: 실행 위치(현재 디렉터리)에 `courses.csv`가 없는 경우입니다. 프로젝트 루트에서 `python timetable.py`로 실행하세요.
- **`CSV에 필수 컬럼이 없습니다`**: `courses.csv`에 `code, name, professor, credits, classroom, capacity, day, start_time, end_time` 컬럼이 모두 있는지 확인하세요.
- **`추천 시간표를 찾을 수 없습니다`**: 입력한 학점 범위나 제외 과목 조건이 너무 빡빡할 수 있습니다. 조건을 완화해서 다시 시도하세요.
- **`TimeSlotError` / `CourseError`**: `courses.csv`의 시간이 09:00~18:00 범위를 벗어나거나, 수업 길이가 1~4시간을 벗어나거나, 학점이 1~3을 벗어나면 발생합니다. 이런 과목은 오류를 던지지 않고 로그에만 기록한 뒤 건너뛰므로, 추천 과목 수가 예상보다 적다면 `timetable.log`를 확인하세요.
- **모델 학습이 안 됨**: `courses.csv`에 `score`가 채워진 과목이 없거나 2개 미만이면 AI 모델 없이 실행되며, 이 경우 선호도 일치도 기반의 규칙 기반 점수로 추천이 생성됩니다.

---

## 라이선스

MIT LICENSE.
