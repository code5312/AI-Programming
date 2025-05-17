# AI 기반 시간표 추천 시스템

이 프로젝트는 학생들의 선호도와 과목 정보를 기반으로 최적의 시간표를 추천해주는 시스템입니다.

## 기능

- 과목 정보 JSON 파일 로드
- 사용자 선호도 기반 시간표 추천
- 시간 충돌 검사
- HTML 형식의 시간표 생성
- 시간표 저장 및 로드
- AI 기반 추천 시스템

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 과목 정보 입력
   - `timetable.json` 파일에 과목 정보 입력

2. 프로그램 실행
```bash
python timetable.py
```

3. 선호도 입력
   - 최소/최대 학점
   - 선호 요일
   - 선호 시간대
   - 선호 교수
   - 제외 과목

4. 결과 확인
   - 생성된 `recommended_timetable.html` 파일 확인

## 파일 구조

- `timetable.py`: 메인 프로그램
- `timetable.json`: 과목 정보
- `training_data.csv`: AI 모델 학습 데이터
- `requirements.txt`: 필요한 패키지 목록

## 라이선스

MIT License 