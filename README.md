# 🗓️ AI-Programming — AI 기반 시간표 추천 시스템

[![Tests](https://github.com/code5312/AI-Programming/actions/workflows/tests.yml/badge.svg)](https://github.com/code5312/AI-Programming/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

학생의 학점 범위, 선호 요일, 선호 교수, 난이도·평점 선호도를 입력하면 AI 모델(RandomForest)이 각 시간표의 점수를 예측하여 **시간 충돌 없는 최적의 시간표**를 자동으로 추천해주는 프로그램입니다. 추천된 시간표는 다크 테마의 HTML 페이지로 시각화되어 브라우저에서 바로 확인할 수 있습니다. 터미널 CLI와 브라우저 기반 웹 UI, 두 가지 방식으로 모두 쓸 수 있습니다.

**🔗 데모: [ai-timetable-recommender.onrender.com](https://ai-timetable-recommender.onrender.com/)** (Render 무료 플랜이라 한동안 요청이 없으면 슬립합니다 — 첫 접속이 느리면 서버가 깨어나는 중입니다)

---

## 목차

- [주요 기능](#주요-기능)
- [폴더 구조](#폴더-구조)
- [요구사항](#요구사항)
- [설치 및 실행 방법](#설치-및-실행-방법)
- [배포 (Render)](#배포-render)
- [사용 방법](#사용-방법)
- [데이터 파일 형식](#데이터-파일-형식)
- [주기적 데이터 갱신 (가천대 강의시간표 자동 가져오기)](#주기적-데이터-갱신-가천대-강의시간표-자동-가져오기)
- [주요 모듈/클래스 구조](#주요-모듈클래스-구조)
- [문제 해결](#문제-해결trouble-shooting)
- [라이선스](#라이선스)

---

## 주요 기능

- 📄 **CSV 한 파일로 구동**: `courses.csv` 하나에 과목 정보 + 학습용 점수를 함께 담아, 이 파일만 통째로 교체하면 다른 학기/커리큘럼 데이터로 바로 전환 가능
- ✅ **과목 데이터 검증**: CSV로 과목 정보를 불러오면서 요일·시간·학점 등의 유효성을 자동 검사
- 🤖 **AI 기반 점수 예측**: 과목의 기본 정보, 시간대, 담당 교수 통계, 사용자 선호도를 특성(feature)으로 추출해 RandomForestRegressor로 시간표 점수를 예측
- 🔍 **백트래킹 시간표 생성**: 시간 충돌을 피하면서 학점 조건을 만족하는 모든 조합을 탐색 후 상위 5개 시간표 반환
- 🎨 **시간표 시각화**: 과목별 색상이 구분된 다크 테마 HTML 시간표 자동 생성
- 💾 **시간표 저장/불러오기**: 마음에 드는 시간표를 이름 붙여 JSON으로 저장, 이후 재사용 가능
- 📊 **특성 중요도 분석**: 모델 학습 후 어떤 요소가 점수에 가장 큰 영향을 미쳤는지 `feature_importance.png`로 시각화
- 🌐 **웹 UI**: Flask 기반 브라우저 UI 제공. 방문자(세션)마다 선호도와 저장 시간표가 서로 격리되어 여러 명이 동시에 써도 안전. 파비콘/OG 메타태그, 테마에 맞춘 404/500 에러 페이지까지 포함
- 📤 **CSV 업로드**: 가천대 데이터가 아니어도, `courses.csv`와 같은 형식이면 웹 UI에서 직접 업로드해 자기 학교/학기 커리큘럼으로 바로 추천받을 수 있음
- 📦 **pip 패키징**: `pip install -e .` 한 번이면 `timetable` / `timetable-web` 커맨드로 바로 실행 가능
- 🎓 **가천대 강의시간표 자동 가져오기**: `gachon-import` 커맨드로 학기당 한 번 실제 개설강좌 데이터를 받아와 `courses.csv`로 저장 가능 (수동 CSV 작성 불필요, 자세한 내용은 [주기적 데이터 갱신](#주기적-데이터-갱신-가천대-강의시간표-자동-가져오기) 참고)

---

## 폴더 구조

```
AI-Programming/
├── timetable.py                  # CLI 실행 진입점 (timetable_app.cli.main 호출)
├── app.py                        # 웹 UI 실행 진입점 (timetable_app.web.main 호출)
├── pyproject.toml                # 패키징 설정 (pip install -e .)
├── requirements.txt              # 의존성 목록
├── timetable_app/                # 실제 구현 패키지
│   ├── models.py                   # TimeSlot, Course, UserPreferences, 도메인 상수
│   ├── features.py                 # 특성(feature) 추출 함수
│   ├── recommender.py              # ScheduleRecommender: 학습/예측/탐색
│   ├── html_report.py              # 시간표 HTML 렌더링
│   ├── persistence.py              # 과목 CSV 입력 + 저장된 시간표 JSON 입출력
│   ├── cli.py                      # 대화형 CLI 입력 및 main()
│   ├── web.py                      # Flask 웹 UI 및 main()
│   └── importers/                  # 외부 학사 시스템 -> courses.csv 변환 스크립트
│       └── gachon.py                 # 가천대 강의시간표 조회 API 임포터
├── tests/                         # unittest 기반 테스트 스위트
├── .github/workflows/tests.yml    # push/PR마다 테스트 자동 실행 (CI)
├── render.yaml                    # Render 배포 설정 (원클릭 배포용)
├── Procfile                       # 프로덕션 시작 명령 (gunicorn app:app)
├── courses.csv                    # 과목 정보 + 학습용 점수 (이 파일만 갈아 끼우면 됨)
├── schedules/                     # CLI로 저장한 시간표 (자동 생성됨)
├── webdata/                       # 웹 세션별 저장 시간표 + 업로드한 CSV (자동 생성됨, git 추적 제외)
├── recommended_timetable.html     # CLI 추천 결과 (실행 후 생성됨)
├── feature_importance.png         # 특성 중요도 그래프 (실행 후 생성됨)
└── timetable.log                  # 실행 로그 (자동 생성됨)
```

---

## 요구사항

- Python 3.8 이상
- 필요 라이브러리 (`requirements.txt`): numpy, pandas, scikit-learn, matplotlib, seaborn, flask

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

패키지로 설치해 `timetable` / `timetable-web` 커맨드를 바로 쓰고 싶다면:

```bash
pip install -e .
```

### 3. 실행

**CLI로 실행** (터미널 대화형 입력):

```bash
python timetable.py
```

**웹 UI로 실행** (브라우저에서 사용, `http://127.0.0.1:5000` 접속):

```bash
python app.py
```

### 4. (선택) 테스트 실행

`tests/`에 표준 라이브러리 `unittest` 기반 테스트가 있습니다 (추가 설치 불필요). 데이터 검증 규칙, `courses.csv` 파싱, 시간표 탐색, HTML 렌더링, 웹 세션 격리, 그리고 이번에 실제로 고쳤던 버그들의 회귀 테스트를 포함합니다. push/PR마다 GitHub Actions에서도 자동으로 실행됩니다.

```bash
python -m unittest discover -s tests -t .
```

---

## 배포 (Render)

로컬에서 `python app.py`로만 쓰던 웹 UI를 링크 하나로 누구나 접속할 수 있게 [Render](https://render.com)에 배포할 수 있습니다. 이 레포에는 이미 `render.yaml`(배포 설정)과 `Procfile`(프로덕션 시작 명령)이 준비되어 있습니다.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/code5312/AI-Programming)

### 수동으로 배포하는 경우

1. Render 대시보드 → **New** → **Blueprint** → 이 GitHub 레포 선택 (`render.yaml`을 자동으로 인식함)
2. 배포 완료 후 발급되는 `https://<서비스이름>.onrender.com` 링크로 누구나 접속 가능

### 알아둘 점

- **프로덕션 WSGI 서버**: `python app.py`(로컬 개발용 Flask 내장 서버) 대신 실제 배포에서는 `gunicorn app:app`을 씁니다. `app.py`가 모듈 레벨에 `app` 객체를 노출해 gunicorn이 바로 가져다 쓸 수 있게 되어 있습니다.
- **세션 서명 키(`FLASK_SECRET_KEY`)**: `render.yaml`의 `generateValue: true`가 배포 시 한 번 자동 생성해 고정값으로 유지합니다. 이 값이 재배포마다 바뀌면 기존 방문자의 세션(및 그 세션에 연결된 저장 시간표 접근)이 끊깁니다.
- **저장 데이터는 재배포 시 초기화될 수 있음**: `webdata/`(웹 세션별 저장 시간표)는 Render 무료 플랜의 임시(ephemeral) 파일시스템에 쓰여서, 재배포하거나 서비스가 슬립 후 깨어날 때 사라질 수 있습니다. 한 세션 안에서 추천받고 저장/조회하는 흐름은 정상 동작하지만, 장기 보관용 저장소는 아닙니다.
- **`courses.csv` 갱신 반영**: `gachon-import`로 새 학기 데이터를 받아 `courses.csv`를 갱신했다면, 커밋 후 GitHub에 push하면 Render가 자동으로 재배포합니다(레포 연동 시 기본 동작).
- **무료 플랜은 일정 시간 요청이 없으면 슬립**합니다. 슬립 상태에서 첫 요청은 서버가 깨어나느라 몇십 초 걸릴 수 있습니다 (포트폴리오 데모 목적이면 보통 괜찮은 수준).

---

## 사용 방법

### CLI

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

### 웹 UI

`python app.py` 실행 후 `http://127.0.0.1:5000` 접속:

1. 폼에서 학점 범위·선호 요일·선호 교수·제외 과목·난이도·평점을 입력하고 "시간표 추천받기" 클릭
2. 추천된 시간표가 다크 테마 HTML로 바로 표시됨
3. 이름을 입력해 "저장" 하면 이후 `/schedules`에서 다시 조회 가능
4. 방문자(브라우저 세션)마다 저장한 시간표가 서로 분리되어 있어, 여러 명이 같은 서버에 동시 접속해도 서로의 결과를 보거나 덮어쓰지 않음
5. 상단 "📤 내 CSV로 추천하기"에서 자기 데이터를 업로드하면, 그 세션은 이후 업로드한 카탈로그로 추천받음 (폼 상단에 배너로 표시됨). "기본 데이터로 돌아가기"를 누르면 원래 카탈로그로 복귀

> 배포 환경(여러 워커/프로세스)에서는 `FLASK_SECRET_KEY` 환경변수로 세션 서명 키를 고정해야 세션이 워커 간에 깨지지 않습니다.

---

## 데이터 파일 형식

아래 형식은 프로젝트 루트의 `courses.csv`(CLI/서버 기본 데이터)와 웹 UI [CSV 업로드](#웹-ui)에 동일하게 적용됩니다.

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
| `start_time` / `end_time` | 수업 시작/종료 시간 (`HH:MM`, 09:00~23:00 사이, 50분~4시간) | ✅ |
| `current_enrolled` | 현재 수강인원 | 선택 (기본 0) |
| `difficulty` | 난이도 0~1 | 선택 (기본 0.5) |
| `rating` | 평점 0~5 | 선택 (기본 3.0) |
| `prerequisites` | 선수과목 코드 목록 (세미콜론 `;`으로 구분, 예: `CODE1;CODE2`) | 선택 |
| `score` | 그 과목의 실제/가상 만족도 점수. 값이 있는 과목만 모델 학습에 쓰이고, 한 과목의 여러 행 중 첫 값만 사용됩니다 | 선택 |

같은 `code`를 가진 행들의 `name`/`professor`/`credits`/`classroom`/`capacity`/`difficulty`/`rating`/`score`는 모두 같은 값이어야 하며(수업 시간만 행마다 다름), 첫 번째 행의 값이 그 과목의 값으로 사용됩니다. `score`가 채워진 과목이 2개 미만이면 모델 학습은 건너뛰고 규칙 기반 점수로만 추천합니다.

과목 후보(제외 과목을 뺀 나머지)가 22개를 넘으면 전수 백트래킹 대신 무작위 다중 시작 그리디 탐색으로 자동 전환되어, `courses.csv`에 과목이 아무리 많아도 응답 시간이 일정 수준으로 유지됩니다 (22개 이하는 항상 최적해를 보장하는 전수 탐색 사용).

---

## 주기적 데이터 갱신 (가천대 강의시간표 자동 가져오기)

학기가 바뀌면 시간대·강의실·교수·과목이 전부 바뀌므로, `courses.csv`를 손으로 매번 새로 만드는 대신 가천대학교 강의시간표 조회 시스템에서 그 학기 개설 과목을 자동으로 받아올 수 있습니다. 자세한 내용은 [`timetable_app/importers/README.md`](timetable_app/importers/README.md) 참고.

```bash
gachon-import --dept CS3120 CS2170 --year 2026 --term 20 --out courses.csv
```

**중요**: 이건 앱 실행 중에 매번 호출하는 실시간 연동이 아니라, **학기당 한 번 수동으로 실행**해서 나온 결과를 `courses.csv`로 저장해두고 그 학기 내내 재사용하는 흐름입니다. 이유는 세 가지입니다.

1. 이 API는 학교가 공식 문서화한 API가 아니라 브라우저 요청을 역추적한 내부 엔드포인트라, 매 요청마다 의존하면 학교 쪽 개편에 앱 전체가 바로 영향받습니다.
2. 학교 서버에 불필요하게 자주 요청을 보내지 않기 위함입니다.
3. 받아온 데이터를 `score`/`difficulty`/`rating` 등 AI 학습용 값을 사람이 직접 채워 넣어 다듬을 시간이 필요합니다 (학교 API에는 이 값들이 없습니다).

받아온 `courses.csv`는 다른 데이터 파일과 마찬가지로 검토 후 커밋해두면 됩니다.

---

## 주요 모듈/클래스 구조

| 모듈 | 클래스/함수 | 역할 |
|---|---|---|
| `timetable_app.models` | `TimeSlot`, `Course`, `UserPreferences` | 데이터 검증 및 도메인 규칙(수업 가능 시간대, 학점 범위 등) |
| `timetable_app.features` | `extract_all_features` 등 | 과목 → 모델 입력 특성(feature) 변환 |
| `timetable_app.recommender` | `ScheduleRecommender` | 과목 카탈로그 관리, 모델 학습/예측, 시간표 탐색 |
| `timetable_app.html_report` | `generate_html` | 추천 시간표를 다크 테마 HTML로 렌더링 |
| `timetable_app.persistence` | `load_courses_from_csv`, `save_schedule_json` 등 | 과목 CSV 입력 + 저장된 시간표 JSON 입출력 |
| `timetable_app.cli` | `main`, `get_user_preferences` | 대화형 CLI 입력 및 실행 진입점 |
| `timetable_app.web` | `create_app`, `main` | Flask 웹 UI 및 실행 진입점 |
| `timetable_app.importers.gachon` | `fetch_courses`, `main` | 가천대 강의시간표 API -> `courses.csv` 변환 (학기당 1회 수동 실행) |

`ScheduleRecommender(data_dir=...)`처럼 `data_dir`을 지정하면 저장된 시간표(`schedules/`)와 특성 중요도 그래프(`feature_importance.png`)를 해당 디렉터리 아래에 쓴다. 지정하지 않으면 기존과 동일하게 프로젝트 루트를 쓰므로 CLI 사용에는 영향이 없다. 웹 UI는 사용자(브라우저 세션)마다 `ScheduleRecommender.clone_with_data_dir()`로 가벼운 복제본을 만들어 이 격리를 이용한다 — 과목 카탈로그와 학습된 모델은 앱 시작 시 한 번만 로드/학습해 공유하고, 저장 경로와 선호도 상태만 세션별로 분리한다.

CSV를 업로드한 세션은 별도로 학습된 자기만의 카탈로그를 쓰는데, 이 캐시는 프로세스 메모리에 있다. gunicorn을 워커 여러 개로 띄우면 요청이 다른 워커로 갈 수 있어, 캐시가 없으면 세션의 `webdata/<세션ID>/uploaded_courses.csv`(업로드 시 저장해둔 원본)를 다시 읽어 그 워커의 캐시를 채운다 — 디스크는 워커 간에 공유되므로 어느 워커가 요청을 받아도 같은 카탈로그로 추천한다.

---

## 문제 해결(Trouble Shooting)

- **`FileNotFoundError`**: 실행 위치(현재 디렉터리)에 `courses.csv`가 없는 경우입니다. 프로젝트 루트에서 `python timetable.py`(또는 `python app.py`)로 실행하세요.
- **`CSV에 필수 컬럼이 없습니다`**: `courses.csv`에 `code, name, professor, credits, classroom, capacity, day, start_time, end_time` 컬럼이 모두 있는지 확인하세요.
- **`추천 시간표를 찾을 수 없습니다`**: 입력한 학점 범위나 제외 과목 조건이 너무 빡빡할 수 있습니다. 조건을 완화해서 다시 시도하세요.
- **`TimeSlotError` / `CourseError`**: `courses.csv`의 시간이 09:00~23:00 범위를 벗어나거나, 수업 길이가 50분~4시간을 벗어나거나, 학점이 1~3을 벗어나면 발생합니다. 이런 과목은 오류를 던지지 않고 로그에만 기록한 뒤 건너뛰므로, 추천 과목 수가 예상보다 적다면 `timetable.log`를 확인하세요.
- **모델 학습이 안 됨**: `courses.csv`에 `score`가 채워진 과목이 없거나 2개 미만이면 AI 모델 없이 실행되며, 이 경우 선호도 일치도 기반의 규칙 기반 점수로 추천이 생성됩니다.
- **웹 UI에서 저장한 시간표가 안 보임**: 브라우저 세션 쿠키를 기준으로 시간표가 격리됩니다. 쿠키를 지웠거나 다른 브라우저/시크릿 모드로 접속하면 이전에 저장한 시간표가 보이지 않는 것이 정상입니다.
- **CSV 업로드가 "CSV를 읽을 수 없습니다" 오류를 냄**: 필수 컬럼 누락, 인코딩 문제, 또는 CSV가 아닌 파일일 가능성이 높습니다. [데이터 파일 형식](#데이터-파일-형식) 표를 참고하세요.
- **CSV 업로드가 "파일이 너무 큽니다" 오류를 냄**: 업로드 용량 제한은 2MB입니다 (일반적인 강의시간표 CSV로는 충분한 크기).

---

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE) 참고.
