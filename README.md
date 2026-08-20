# AI-Programming: AI 기반 시간표 추천 시스템

학생의 학점 범위, 선호 요일, 선호 교수, 난이도·평점 선호도를 입력하면 AI 모델(RandomForest)이 각 시간표의 점수를 예측하여 **시간 충돌 없는 최적의 시간표**를 자동으로 추천해주는 프로그램입니다. 추천된 시간표는 다크 테마의 HTML 페이지로 시각화되어 브라우저에서 바로 확인할 수 있습니다.

---

## 주요 기능

- **과목 데이터 검증**: JSON으로 과목 정보를 불러오면서 요일·시간·학점 등의 유효성을 자동 검사
- **AI 기반 점수 예측**: 과목의 기본 정보, 시간대, 담당 교수 통계, 사용자 선호도를 특성(feature)으로 추출해 RandomForestRegressor로 시간표 점수를 예측
- **백트래킹 시간표 생성**: 시간 충돌을 피하면서 학점 조건을 만족하는 모든 조합을 탐색 후 상위 5개 시간표 반환
- **시간표 시각화**: 과목별 색상이 구분된 다크 테마 HTML 시간표 자동 생성
- **시간표 저장/불러오기**: 마음에 드는 시간표를 이름 붙여 JSON으로 저장, 이후 재사용 가능
- **특성 중요도 분석**: 모델 학습 후 어떤 요소가 점수에 가장 큰 영향을 미쳤는지 `feature_importance.png`로 시각화

---

## 폴더 구조

```
AI-Programming/
├── timetable.py           # 메인 프로그램 (실행 파일)
├── timetable.json          # 과목 정보 데이터
├── training_data.csv       # AI 모델 학습용 데이터
├── schedules/               # 저장된 시간표 (자동 생성됨)
├── recommended_timetable.html   # 추천 결과 (실행 후 생성됨)
├── feature_importance.png       # 특성 중요도 그래프 (실행 후 생성됨)
└── timetable.log             # 실행 로그 (자동 생성됨)
```

---

## 요구사항

- Python 3.8 이상
- 필요 라이브러리:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
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
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3. ⚠️ 데이터 경로 확인 (필독)

`timetable.py`의 `main()` 함수 안에서 과목 데이터를 아래처럼 절대경로로 불러오도록 되어 있습니다.

```python
recommender.load_courses_from_json("C:\\Users\\ASUS\\.cursor\\timetable.json")
```

다른 환경에서 실행할 경우 이 경로가 존재하지 않아 오류가 납니다. 실행 전 아래처럼 **같은 폴더의 `timetable.json`을 가리키도록 수정**해주세요.

```python
recommender.load_courses_from_json("timetable.json")
```

### 4. 실행

```bash
python timetable.py
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

- `training_data.csv`가 있으면 AI 모델을 자동 학습
- 조건에 맞는 시간표 중 AI 예측 점수가 가장 높은 시간표를 선택
- `recommended_timetable.html` 파일을 생성하고 자동으로 브라우저에서 열림
- 저장 여부를 물어보며, `y` 입력 시 `schedules/이름.json`으로 저장

---

## 데이터 파일 형식

### `timetable.json` (과목 정보)

| 필드 | 설명 | 필수 |
|---|---|---|
| `code` | 과목 코드 | ✅ |
| `name` | 과목명 | ✅ |
| `professor` | 담당 교수 | ✅ |
| `credits` | 학점 (1~3) | ✅ |
| `time_slots` | `[{day, start_time, end_time}]` 형태의 수업 시간 목록 (day: 0=월 ~ 4=금, 시간은 09:00~18:00 사이) | ✅ |
| `classroom` | 강의실 | ✅ |
| `capacity` | 정원 | ✅ |
| `current_enrolled` | 현재 수강인원 | 선택 (기본 0) |
| `difficulty` | 난이도 0~1 | 선택 (기본 0.5) |
| `rating` | 평점 0~5 | 선택 (기본 3.0) |
| `prerequisites` | 선수과목 코드 목록 | 선택 |

### `training_data.csv` (모델 학습 데이터)

`code, name, professor, credits, difficulty, rating, score` 컬럼으로 구성되며, 각 행은 과목 조합에 대한 실제/가상의 만족도 점수(`score`)를 담고 있어야 모델이 학습됩니다.

---

## 주요 클래스 구조

| 클래스 | 역할 |
|---|---|
| `TimeSlot` | 요일·시작/종료 시간을 검증하는 수업 시간 단위 |
| `Course` | 과목 정보 (JSON ↔ 객체 변환 포함) |
| `UserPreferences` | 학점 범위, 선호 요일/교수/난이도/평점 등 사용자 조건 |
| `ScheduleRecommender` | 특성 추출 → 모델 학습 → 추천 생성 → HTML 시각화까지 담당하는 핵심 클래스 |

---

## 문제 해결(Trouble Shooting)

- **`FileNotFoundError`**: `timetable.json` 경로가 잘못되었을 가능성이 큽니다. 위 "데이터 경로 확인" 항목을 참고하세요.
- **`추천 시간표를 찾을 수 없습니다`**: 입력한 학점 범위나 제외 과목 조건이 너무 빡빡할 수 있습니다. 조건을 완화해서 다시 시도하세요.
- **`TimeSlotError` / `CourseError`**: `timetable.json`의 시간이 09:00~18:00 범위를 벗어나거나, 수업 길이가 1~4시간을 벗어나면 발생합니다. 데이터를 점검하세요.
- **모델 학습이 안 됨**: `training_data.csv`가 없으면 AI 모델 없이 실행되며, 이 경우 `generate_recommendations()`가 규칙 기반 로직(`super()`)을 호출하도록 되어 있으나 현재 부모 클래스가 없어 오류가 날 수 있습니다. 실행 전 `training_data.csv`를 준비하는 것을 권장합니다.

---

## 라이선스

별도 명시 없음.
