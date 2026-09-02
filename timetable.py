"""
시간표 추천 시스템 - 실행 진입점

실제 구현은 timetable_app 패키지에 있습니다 (models/features/recommender/
html_report/persistence/cli). 이 파일은 `python timetable.py`로 그대로
실행할 수 있도록 남겨둔 얇은 래퍼입니다.
"""
from timetable_app.cli import main

if __name__ == "__main__":
    main()
