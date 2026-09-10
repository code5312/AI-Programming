"""
웹 UI 실행 진입점 + WSGI 앱 노출

`python app.py`로 실행하면 로컬 개발 서버가 뜨고, gunicorn 같은 프로덕션 WSGI
서버는 모듈 레벨 `app` 객체를 그대로 가져다 쓸 수 있다 (`gunicorn app:app`).
실제 구현은 timetable_app.web에 있다.
"""
import os

from timetable_app.web import create_app

app = create_app()

if __name__ == "__main__":
    # Render 등 대부분의 PaaS는 PORT 환경변수로 리슨할 포트를 지정하고
    # 0.0.0.0 바인딩을 요구한다. 로컬 개발 시에는 두 값 다 안 넘어오므로
    # 기존과 동일하게 127.0.0.1:5000으로 뜬다.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
        threaded=True,
    )
