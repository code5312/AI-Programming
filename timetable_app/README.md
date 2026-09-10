# timetable_app

AI 시간표 추천 시스템의 실제 구현 패키지. 전체 사용법/설치 방법은 [루트 README](../README.md) 참고.

## 모듈 구성

| 모듈 | 역할 |
|---|---|
| `models.py` | `TimeSlot`, `Course`, `UserPreferences` 데이터클래스와 도메인 상수(수업 가능 시간대, 학점 범위 등) |
| `features.py` | 과목 → AI 모델 입력 특성(feature) 변환 |
| `recommender.py` | `ScheduleRecommender` — 과목 카탈로그 관리, 모델 학습/예측, 시간표 탐색(백트래킹/그리디) |
| `html_report.py` | 추천 시간표를 다크 테마 HTML로 렌더링 |
| `persistence.py` | `courses.csv` 입력 파싱 + 저장된 시간표(JSON) 입출력 |
| `cli.py` | 터미널 대화형 입력 및 CLI 실행 진입점(`main`) |
| `web.py` | Flask 기반 웹 UI 및 실행 진입점(`main`). 기본 `courses.csv` 카탈로그 외에, 방문자가 직접 CSV를 업로드해 자기 카탈로그로 전환하는 기능(`/upload`, `/reset-data`) 포함 |
| `importers/` | 외부 학사 시스템 데이터를 `courses.csv` 형식으로 변환하는 스크립트 모음 — [importers/README.md](importers/README.md) 참고 |

## 의존 방향

`models` ← `features`/`persistence` ← `recommender` ← `cli`/`web`. 상위(`cli`/`web`)는 하위 모듈을 조합만 하고, 핵심 로직(검증 규칙, 특성 추출, 탐색 알고리즘)은 전부 하위 모듈에 있다 — CLI/웹 어느 쪽에서 실행하든 동일한 로직을 거친다.

`importers/`는 이 의존 그래프 밖에 있는 별도 유틸리티다. `courses.csv`와 같은 형식의 파일을 만들어낼 뿐, 런타임에 `recommender`/`cli`/`web`이 직접 호출하지 않는다 — 학기당 한 번 수동으로 실행해 데이터를 미리 준비해두는 용도.

## web.py의 카탈로그 캐싱

`web.py`는 앱 시작 시 로드한 기본 카탈로그(`base_recommender`)를 요청마다 `ScheduleRecommender.clone_with_data_dir()`로 가볍게 복제해 쓴다 — 과목 목록과 학습된 모델은 공유 객체를 그대로 참조하고(읽기 전용이라 안전), `data_dir`(저장 시간표 경로)만 세션별로 새로 지정한다.

업로드된 카탈로그는 세션ID를 키로 하는 인메모리 딕셔너리(`uploaded_recommenders`)에 캐싱되는데, 이건 gunicorn 멀티 워커에서 프로세스마다 따로 존재한다. 그래서 캐시 미스 시 `webdata/<세션ID>/uploaded_courses.csv`(업로드 때 디스크에 저장해둔 원본)가 있으면 그걸로 재구성해 캐시를 다시 채운다 — 디스크는 워커 간에 공유되므로 어느 워커가 요청을 받아도 정합성이 유지된다.
