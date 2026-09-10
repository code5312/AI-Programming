"""
Flask 기반 웹 UI.

cli.py와 동일한 핵심 로직(ScheduleRecommender/html_report)을 그대로 재사용한다.
과목 카탈로그 로드 + AI 모델 학습(비용이 큰 작업)은 앱 시작 시 딱 한 번만 하고,
요청이 올 때마다 base_recommender.clone_with_data_dir()로 가벼운 복제본을 만들어
쓴다 - 학습된 모델은 공유하되, 선호도 상태와 저장 시간표 디렉터리(schedules/)만
브라우저 세션마다 격리해서 동시 접속자끼리 서로의 시간표를 보거나 덮어쓰지
않도록 한다.
"""
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, redirect, request, session

from . import PROJECT_ROOT
from .html_report import generate_html
from .models import (
    Course,
    DAY_NAMES,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    RATING_MAX,
    RATING_MIN,
    TOTAL_CREDITS_MAX,
    TOTAL_CREDITS_MIN,
    UserPreferences,
)
from .recommender import ScheduleRecommender, build_recommender_from_csv

# 세션별 산출물(schedules/, feature_importance.png)이 쌓이는 위치. 프로젝트
# 소스와 섞이지 않도록 별도 폴더로 두고 .gitignore에서 제외한다.
WEB_DATA_DIR = PROJECT_ROOT / "webdata"

_PAGE_STYLE = """
:root {
    --bg: #0b0d11;
    --bg-card: #14171d;
    --bg-card-2: #171b22;
    --bg-input: #1b1f28;
    --border: #262b35;
    --text: #e9eaee;
    --text-dim: #9aa0ab;
    --accent: #43aa8b;
    --accent-2: #577590;
    --danger: #ef476f;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    min-height: 100vh;
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", Roboto, sans-serif;
    background:
        radial-gradient(circle at 15% -10%, rgba(87, 117, 144, 0.25), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(67, 170, 139, 0.18), transparent 45%),
        var(--bg);
    padding: 56px 20px;
    line-height: 1.5;
}
.container { max-width: 620px; margin: 0 auto; }
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 36px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
}
.eyebrow {
    font-size: 13px; font-weight: 700; letter-spacing: .04em;
    color: var(--accent); text-transform: uppercase; margin-bottom: 8px;
}
h1 { font-size: 26px; margin: 0 0 8px; color: #fff; letter-spacing: -.01em; }
.subtitle { color: var(--text-dim); margin: 0 0 28px; font-size: 14.5px; }
.section {
    background: var(--bg-card-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
}
.section-title {
    font-weight: 600; font-size: 14.5px; color: #fff;
    display: flex; align-items: center; gap: 8px; margin-bottom: 16px;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
label { display: block; font-size: 13px; color: var(--text-dim); margin-bottom: 6px; }
input[type=text], input[type=number] {
    width: 100%; padding: 10px 12px; border-radius: 9px;
    border: 1px solid var(--border); background: var(--bg-input); color: var(--text);
    font-size: 14.5px; transition: border-color .15s, box-shadow .15s;
}
input[type=text]:focus, input[type=number]:focus {
    outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(67, 170, 139, .22);
}
input::placeholder { color: #6b7078; }
input[type=file] { width: 100%; padding: 10px 0; color: var(--text-dim); font-size: 14px; }
input[type=file]::file-selector-button {
    background: var(--bg-input); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; margin-right: 12px; cursor: pointer; font-size: 13.5px;
}
input[type=file]::file-selector-button:hover { border-color: var(--accent-2); }
.banner {
    background: rgba(87, 117, 144, .12); border: 1px solid rgba(87, 117, 144, .35);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 22px; font-size: 13.5px;
    color: var(--text-dim); display: flex; justify-content: space-between; align-items: center; gap: 12px;
}
.banner form { margin: 0; }
.banner button.linklike {
    all: unset; color: var(--accent); cursor: pointer; text-decoration: underline; font-size: 13.5px;
}
.day-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.day-chip { position: relative; }
.day-chip input { position: absolute; opacity: 0; width: 0; height: 0; }
.day-chip span {
    display: inline-block; padding: 9px 18px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--text-dim); cursor: pointer;
    font-size: 14px; transition: all .15s ease; user-select: none;
}
.day-chip span:hover { border-color: var(--accent-2); color: #fff; }
.day-chip input:checked + span {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff; border-color: transparent;
    box-shadow: 0 4px 14px rgba(67, 170, 139, .35);
}
button, .btn {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff; border: none; padding: 13px 24px; border-radius: 11px;
    font-size: 15px; font-weight: 600; cursor: pointer; text-decoration: none;
    display: inline-flex; align-items: center; gap: 6px; width: 100%;
    justify-content: center; margin-top: 6px;
    box-shadow: 0 8px 20px rgba(67, 170, 139, .28);
    transition: transform .12s ease, box-shadow .12s ease;
}
button:hover, .btn:hover { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(67, 170, 139, .4); }
button:active, .btn:active { transform: translateY(0); }
.error {
    background: rgba(239, 71, 111, .1); border: 1px solid rgba(239, 71, 111, .35);
    border-left: 4px solid var(--danger); border-radius: 10px; padding: 14px 16px;
    color: #ffd7de; margin-bottom: 22px; font-size: 14px; display: flex; gap: 10px;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.footer-links { margin-top: 24px; text-align: center; font-size: 13.5px; color: var(--text-dim); }
.footer-links a { margin: 0 6px; }
.schedule-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 8px; }
.schedule-list li a {
    display: flex; justify-content: space-between; align-items: center;
    padding: 15px 18px; border: 1px solid var(--border); border-radius: 12px;
    background: var(--bg-card-2); color: var(--text); font-size: 14.5px;
    transition: border-color .15s, transform .12s;
}
.schedule-list li a:hover { border-color: var(--accent-2); transform: translateX(3px); text-decoration: none; }
.schedule-list li a::after { content: "→"; color: var(--text-dim); }
.empty-state { color: var(--text-dim); text-align: center; padding: 26px 0; font-size: 14px; }
.center-icon { font-size: 42px; text-align: center; margin-bottom: 8px; }
.hint { font-size: 12.5px; color: var(--text-dim); margin-top: 8px; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>{_PAGE_STYLE}</style>
</head>
<body>
    <div class="container">
        <div class="card">
            {body}
        </div>
    </div>
</body>
</html>"""


def _preference_form(error: Optional[str] = None, custom_active: bool = False) -> str:
    error_html = f'<div class="error">⚠️ <div>{error}</div></div>' if error else ""
    banner_html = ""
    if custom_active:
        banner_html = """
        <div class="banner">
            <span>📤 업로드한 나만의 데이터로 추천 중입니다.</span>
            <form method="post" action="/reset-data">
                <button type="submit" class="linklike">기본 데이터로 돌아가기</button>
            </form>
        </div>
        """
    day_chips = "\n".join(
        f'<label class="day-chip"><input type="checkbox" name="preferred_days" value="{i}"><span>{name}</span></label>'
        for i, name in enumerate(DAY_NAMES)
    )
    body = f"""
        <div class="eyebrow">AI Timetable</div>
        <h1>🗓️ 시간표 추천받기</h1>
        <p class="subtitle">선호도를 입력하면 AI가 시간 충돌 없는 최적의 시간표를 찾아드립니다.</p>
        {banner_html}
        {error_html}
        <form method="post" action="/recommend">
            <div class="section">
                <div class="section-title">🎓 학점 범위</div>
                <div class="grid-2">
                    <div>
                        <label>최소 학점 ({TOTAL_CREDITS_MIN}~{TOTAL_CREDITS_MAX})</label>
                        <input type="number" name="min_credits" min="{TOTAL_CREDITS_MIN}" max="{TOTAL_CREDITS_MAX}" value="{TOTAL_CREDITS_MIN}" required>
                    </div>
                    <div>
                        <label>최대 학점 ({TOTAL_CREDITS_MIN}~{TOTAL_CREDITS_MAX})</label>
                        <input type="number" name="max_credits" min="{TOTAL_CREDITS_MIN}" max="{TOTAL_CREDITS_MAX}" value="{TOTAL_CREDITS_MAX}" required>
                    </div>
                </div>
            </div>
            <div class="section">
                <div class="section-title">📅 선호 요일 <span class="hint" style="margin:0;">(선택 사항)</span></div>
                <div class="day-grid">{day_chips}</div>
            </div>
            <div class="section">
                <div class="section-title">👩‍🏫 선호 교수 / 제외 과목</div>
                <label>선호 교수 (쉼표로 구분)</label>
                <input type="text" name="preferred_professors" placeholder="예: 박기성, 김태규">
                <label style="margin-top:12px;">제외 과목 코드 (쉼표로 구분)</label>
                <input type="text" name="excluded_courses" placeholder="예: 14395001">
            </div>
            <div class="section">
                <div class="section-title">⭐ 난이도 / 평점 선호</div>
                <div class="grid-2">
                    <div>
                        <label>선호 난이도 ({DIFFICULTY_MIN}~{DIFFICULTY_MAX})</label>
                        <input type="number" step="0.1" name="preferred_difficulty" min="{DIFFICULTY_MIN}" max="{DIFFICULTY_MAX}" value="0.5">
                    </div>
                    <div>
                        <label>선호 평점 ({RATING_MIN}~{RATING_MAX})</label>
                        <input type="number" step="0.1" name="preferred_rating" min="{RATING_MIN}" max="{RATING_MAX}" value="3.0">
                    </div>
                </div>
            </div>
            <button type="submit">✨ 시간표 추천받기</button>
        </form>
        <div class="footer-links"><a href="/upload">📤 내 CSV로 추천하기</a> · <a href="/schedules">💾 저장된 시간표 보기</a></div>
    """
    return _page("AI 시간표 추천", body)


def _upload_page(error: Optional[str] = None) -> str:
    error_html = f'<div class="error">⚠️ <div>{error}</div></div>' if error else ""
    body = f"""
        <div class="eyebrow">AI Timetable</div>
        <h1>📤 내 강의시간표 CSV 업로드</h1>
        <p class="subtitle">다른 학교/학기 커리큘럼이어도 <code>courses.csv</code>와 같은 형식이면 그대로 쓸 수 있습니다.</p>
        {error_html}
        <div class="section">
            <div class="section-title">필수 컬럼</div>
            <p class="hint" style="margin:0;">code, name, professor, credits, classroom, capacity, day, start_time, end_time</p>
            <p class="hint">선택: current_enrolled, difficulty, rating, prerequisites, score (score가 2개 이상 채워져 있으면 AI 모델을 학습합니다)</p>
        </div>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <div class="section">
                <label>CSV 파일</label>
                <input type="file" name="csv_file" accept=".csv" required>
            </div>
            <button type="submit">업로드</button>
        </form>
        <div class="footer-links"><a href="/">← 돌아가기</a></div>
    """
    return _page("CSV 업로드", body)


def _error_page(message: str) -> str:
    body = f"""
        <div class="center-icon">⚠️</div>
        <h1 style="text-align:center;">오류가 발생했습니다</h1>
        <div class="error" style="justify-content:center;">{message}</div>
        <a href="/" class="btn">← 돌아가기</a>
    """
    return _page("오류", body)


# generate_html()이 만든 (다크 테마) 시간표 페이지에 저장/이동 UI를 덧붙일 때
# 쓰는 스타일. generate_html의 <style>과 클래스명이 겹치지 않도록 tt- 접두어를 쓴다.
_BAR_STYLE = """
.tt-bar {
    max-width: 640px; margin: 20px auto 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", Roboto, sans-serif;
}
.tt-bar form { display: flex; gap: 8px; }
.tt-bar input[type=text] {
    flex: 1; padding: 11px 14px; border-radius: 10px; border: 1px solid #333;
    background: #1b1f28; color: #eee; font-size: 14.5px;
}
.tt-bar input[type=text]:focus { outline: none; border-color: #43aa8b; }
.tt-bar button {
    background: linear-gradient(135deg, #43aa8b, #577590); color: #fff; border: none;
    padding: 11px 22px; border-radius: 10px; font-size: 14.5px; font-weight: 600;
    cursor: pointer;
}
.tt-bar .tt-links { margin-top: 14px; font-size: 13.5px; color: #9aa0ab; text-align: center; }
.tt-bar .tt-links a { color: #43aa8b; text-decoration: none; margin: 0 6px; }
.tt-bar .tt-links a:hover { text-decoration: underline; }
"""


def _with_save_bar(html_content: str) -> str:
    bar = """
        <div class="tt-bar">
            <form method="post" action="/save">
                <input type="text" name="name" placeholder="저장할 이름" required>
                <button type="submit">💾 저장</button>
            </form>
            <p class="tt-links"><a href="/">✨ 다시 추천받기</a> · <a href="/schedules">저장된 시간표 보기</a></p>
        </div>
    """
    return html_content.replace("</head>", f"<style>{_BAR_STYLE}</style></head>").replace("</body>", bar + "</body>")


def _with_back_link(html_content: str) -> str:
    link = '<div class="tt-bar"><p class="tt-links"><a href="/schedules">← 목록으로</a></p></div>'
    return html_content.replace("</head>", f"<style>{_BAR_STYLE}</style></head>").replace("</body>", link + "</body>")


def _parse_preferences(form) -> UserPreferences:
    professors = [p.strip() for p in form.get("preferred_professors", "").split(",") if p.strip()]
    excluded = [c.strip() for c in form.get("excluded_courses", "").split(",") if c.strip()]
    days = [int(d) for d in form.getlist("preferred_days")]
    return UserPreferences(
        min_credits=int(form.get("min_credits", TOTAL_CREDITS_MIN)),
        max_credits=int(form.get("max_credits", TOTAL_CREDITS_MAX)),
        preferred_days=days,
        preferred_professors=professors,
        excluded_courses=excluded,
        preferred_difficulty=float(form.get("preferred_difficulty", 0.5)),
        preferred_rating=float(form.get("preferred_rating", 3.0)),
    )


def _load_base_recommender() -> ScheduleRecommender:
    """앱 시작 시 한 번만 courses.csv를 읽고(가능하면) 모델을 학습해둔다."""
    recommender, _trained = build_recommender_from_csv(str(PROJECT_ROOT / "courses.csv"))
    return recommender


def create_app() -> Flask:
    app = Flask(__name__)
    # 세션 쿠키 서명용 키. 여러 워커/프로세스로 배포할 때는 워커마다 다른 랜덤
    # 키가 생기면 세션이 깨지므로, 배포 환경에서는 반드시 FLASK_SECRET_KEY
    # 환경변수로 고정값을 넣어야 한다 (README 참고).
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
    # 업로드 CSV 크기 제한 (남용 방지). 과목 몇백 개짜리 CSV도 수백 KB 수준이라
    # 2MB면 충분히 넉넉하다.
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

    base_recommender = _load_base_recommender()
    # 세션이 업로드한 커스텀 카탈로그(학습된 모델 포함)를 세션ID로 캐싱.
    # 매 요청마다 CSV를 다시 파싱/학습하면 느리므로, 한 번 만든 뒤 재사용한다.
    #
    # 프로세스 메모리 캐시라 gunicorn을 워커 여러 개로 띄우면 요청이 다른
    # 워커로 갈 때 이 캐시가 비어 있을 수 있다 - 그래서 캐시 미스일 때는
    # 디스크(data_dir/uploaded_courses.csv, upload()가 저장해둔 파일)를 다시
    # 확인해 재구성한다. 디스크는 워커들이 공유하므로 이 경로면 몇 초 안 걸리는
    # 재학습 한 번으로 정합성이 보장된다 (완전히 사라지는 건 재배포/재시작 때뿐 -
    # 저장 시간표(webdata/)도 같은 제약이라 새로울 것 없다).
    uploaded_recommenders: Dict[str, ScheduleRecommender] = {}
    app.uploaded_recommenders = uploaded_recommenders  # 테스트에서 캐시 상태를 직접 들여다보기 위한 참조

    def _uploaded_csv_path(session_id: str) -> Path:
        return WEB_DATA_DIR / session_id / "uploaded_courses.csv"

    def _ensure_session_id() -> str:
        session_id = session.get("session_id")
        if not session_id:
            session_id = uuid.uuid4().hex
            session["session_id"] = session_id
        return session_id

    def _has_custom_catalog(session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        return session_id in uploaded_recommenders or _uploaded_csv_path(session_id).exists()

    def _session_recommender() -> ScheduleRecommender:
        session_id = _ensure_session_id()
        data_dir = WEB_DATA_DIR / session_id
        custom = uploaded_recommenders.get(session_id)
        if custom is None:
            csv_path = _uploaded_csv_path(session_id)
            if csv_path.exists():
                custom, _trained = build_recommender_from_csv(str(csv_path), data_dir=data_dir)
                uploaded_recommenders[session_id] = custom
        if custom is not None:
            return custom.clone_with_data_dir(data_dir)
        return base_recommender.clone_with_data_dir(data_dir)

    @app.get("/")
    def index():
        session_id = session.get("session_id")
        return _preference_form(custom_active=_has_custom_catalog(session_id))

    @app.post("/recommend")
    def recommend():
        try:
            preferences = _parse_preferences(request.form)
        except (ValueError, TypeError) as e:
            return _error_page(f"입력값이 올바르지 않습니다: {e}"), 400

        recommender = _session_recommender()
        recommender.set_user_preferences(preferences)
        recommendations = recommender.generate_recommendations()

        if not recommendations:
            return _error_page("조건을 만족하는 시간표를 찾을 수 없습니다. 조건을 완화해서 다시 시도하세요.")

        best_schedule, score = recommendations[0]
        # 저장(/save) 요청이 왔을 때 다시 찾을 수 있도록 과목 코드만 세션에 남긴다.
        session["last_schedule_codes"] = [c.code for c in best_schedule]

        html_content = generate_html(best_schedule, score)
        return _with_save_bar(html_content)

    @app.post("/save")
    def save():
        name = request.form.get("name", "").strip()
        codes = session.get("last_schedule_codes")
        if not name or not codes:
            return _error_page("저장할 시간표가 없습니다. 먼저 추천을 받아주세요.")

        recommender = _session_recommender()
        schedule: List[Course] = [c for c in recommender.courses if c.code in codes]
        try:
            recommender.save_schedule(name, schedule)
        except ValueError as e:
            return _error_page(str(e)), 400

        body = f"""
            <div class="center-icon">✅</div>
            <h1 style="text-align:center;">저장 완료</h1>
            <p class="subtitle" style="text-align:center;">"{name}" 이름으로 저장되었습니다.</p>
            <a href="/schedules" class="btn">💾 저장된 시간표 보기</a>
        """
        return _page("저장 완료", body)

    @app.get("/schedules")
    def list_schedules():
        recommender = _session_recommender()
        recommender.schedules_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(p.stem for p in recommender.schedules_dir.glob("*.json"))
        if not names:
            items = '<div class="empty-state">아직 저장된 시간표가 없습니다.</div>'
        else:
            items = "<ul class='schedule-list'>" + "".join(
                f'<li><a href="/schedules/{n}">{n}</a></li>' for n in names
            ) + "</ul>"
        body = f"""
            <div class="eyebrow">AI Timetable</div>
            <h1>💾 저장된 시간표</h1>
            <p class="subtitle">이 브라우저 세션에서 저장한 시간표 목록입니다.</p>
            {items}
            <a href="/" class="btn">✨ 새로 추천받기</a>
        """
        return _page("저장된 시간표", body)

    @app.get("/schedules/<name>")
    def view_schedule(name: str):
        recommender = _session_recommender()
        try:
            schedule = recommender.load_schedule(name)
        except (FileNotFoundError, ValueError) as e:
            return _error_page(f"시간표를 불러올 수 없습니다: {e}"), 404
        html_content = generate_html(schedule)
        return _with_back_link(html_content)

    @app.get("/upload")
    def upload_form():
        return _upload_page()

    @app.post("/upload")
    def upload():
        file = request.files.get("csv_file")
        if file is None or file.filename == "":
            return _upload_page("CSV 파일을 선택해주세요."), 400
        if not file.filename.lower().endswith(".csv"):
            return _upload_page("CSV 파일(.csv)만 업로드할 수 있습니다."), 400

        session_id = _ensure_session_id()
        data_dir = WEB_DATA_DIR / session_id
        data_dir.mkdir(parents=True, exist_ok=True)
        # 원본 파일명은 쓰지 않고 고정된 이름으로 저장 - 경로 조작 위험을
        # 애초에 차단하고, 이후 로직도 항상 이 경로 하나만 알면 된다.
        csv_path = _uploaded_csv_path(session_id)
        file.save(csv_path)

        try:
            recommender, trained = build_recommender_from_csv(str(csv_path), data_dir=data_dir)
        except Exception as e:
            # 업로드 파일은 신뢰할 수 없는 외부 입력이라(깨진 인코딩, CSV가 아닌
            # 파일 등 pandas가 어떤 예외를 던질지 예측 불가), 파싱 단계의 모든
            # 예외를 사용자에게 보여줄 오류로 취급한다.
            csv_path.unlink(missing_ok=True)
            return _upload_page(f"CSV를 읽을 수 없습니다: {e}"), 400

        if not recommender.courses:
            csv_path.unlink(missing_ok=True)
            return _upload_page(
                "올바른 과목 데이터를 찾지 못했습니다. 필수 컬럼과 형식(요일 0~4, "
                "시간 09:00~23:00 등)을 확인해주세요. 자세한 사유는 timetable.log에 남습니다."
            ), 400

        uploaded_recommenders[session_id] = recommender

        model_note = "AI 모델이 학습되었습니다." if trained else "score 값이 없어 규칙 기반으로 추천합니다."
        body = f"""
            <div class="center-icon">✅</div>
            <h1 style="text-align:center;">업로드 완료</h1>
            <p class="subtitle" style="text-align:center;">{len(recommender.courses)}개 과목을 불러왔습니다. {model_note}</p>
            <a href="/" class="btn">✨ 이 데이터로 추천받기</a>
        """
        return _page("업로드 완료", body)

    @app.post("/reset-data")
    def reset_data():
        session_id = session.get("session_id")
        if session_id:
            uploaded_recommenders.pop(session_id, None)
            _uploaded_csv_path(session_id).unlink(missing_ok=True)
        return redirect("/")

    @app.errorhandler(413)
    def too_large(_e):
        return _upload_page("파일이 너무 큽니다 (최대 2MB)."), 413

    return app


def main():
    """개발 서버 실행 진입점 (python app.py 또는 `timetable-web` 콘솔 스크립트)"""
    app = create_app()
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
        threaded=True,
    )


if __name__ == "__main__":
    main()
