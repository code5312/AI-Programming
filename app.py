"""
웹 UI 실행 진입점

실제 구현은 timetable_app.web에 있습니다. 이 파일은 `python app.py`로 그대로
실행할 수 있도록 남겨둔 얇은 래퍼입니다 (timetable.py와 동일한 패턴).
"""
from timetable_app.web import main

if __name__ == "__main__":
    main()
