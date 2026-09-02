"""
시간표 HTML 시각화

추천된 시간표를 다크 테마 HTML 페이지로 렌더링합니다.
"""
import hashlib
from typing import List, Optional

from .models import Course

DAYS = ["월", "화", "수", "목", "금"]
HOURS = list(range(9, 18))

# 과목 색상은 과목 코드의 해시값으로 팔레트에서 결정합니다.
# (과목명을 하드코딩한 매핑 테이블은 다른 CSV/JSON으로 교체하면 전부 기본색으로
#  깨지므로, 데이터가 바뀌어도 항상 과목마다 고유하고 일관된 색을 갖도록 함)
COLOR_PALETTE = [
    "#e76f51", "#90be6d", "#e9c46a", "#577590",
    "#f94144", "#43aa8b", "#f3722c", "#a05195",
    "#277da1", "#f9c74f", "#4d908e", "#ef476f",
]


def calculate_total_credits(schedule: List[Course]) -> int:
    """총 학점 계산"""
    return sum(course.credits for course in schedule)


def _course_color(course: Course) -> str:
    """과목 코드 기반 결정적(deterministic) 색상 배정"""
    digest = hashlib.md5(course.code.encode("utf-8")).hexdigest()
    return COLOR_PALETTE[int(digest, 16) % len(COLOR_PALETTE)]


def generate_html(schedule: List[Course], score: Optional[float] = None) -> str:
    """HTML 생성 (점수 표시 포함)"""
    timetable = {hour: {day: [] for day in range(5)} for hour in HOURS}

    for course in schedule:
        color = _course_color(course)
        for slot in course.time_slots:
            start_hour = slot.start_time.hour
            duration = (slot.end_time.hour - slot.start_time.hour) + \
                      (slot.end_time.minute - slot.start_time.minute) / 60

            for offset in range(int(duration)):
                timetable[start_hour + offset][slot.day].append({
                    "subject": course.name,
                    "location": course.classroom,
                    "is_first": offset == 0,
                    "rowspan": int(duration) if offset == 0 else 0,
                    "professor": course.professor,
                    "credits": course.credits,
                    "color": color,
                })

    html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>AI 추천 시간표</title>
    <style>
        body {{
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .timetable {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        .timetable th, .timetable td {{
            border: 1px solid #333;
            text-align: center;
            height: 80px;
            position: relative;
        }}
        .timetable th {{
            background-color: #1e1e1e;
            font-weight: bold;
        }}
        .class-block {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            margin: 4px;
            padding: 5px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            display: flex;
            flex-direction: column;
            justify-content: center;
            color: white;
        }}
        .info-panel {{
            background-color: #1e1e1e;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
        }}
        .info-panel h2 {{
            margin-top: 0;
        }}
        .info-panel p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="info-panel">
        <h2>시간표 정보</h2>
        <p>총 학점: {calculate_total_credits(schedule)}</p>
        <p>총 과목 수: {len(schedule)}</p>
        {f'<p>선호도 점수: {score:.2f}</p>' if score is not None else ''}
    </div>
    <table class="timetable">
        <thead>
            <tr>
                <th>시간/요일</th>
"""

    for day in DAYS:
        html_content += f"                <th>{day}</th>\n"

    html_content += "            </tr>\n        </thead>\n        <tbody>\n"

    for hour in HOURS:
        html_content += f"            <tr>\n                <td>{hour:02d}:00</td>\n"
        for day in range(5):
            cells = timetable[hour][day]
            if not cells:
                html_content += "                <td></td>\n"
            else:
                cell = cells[0]
                if cell["is_first"]:
                    html_content += f"""                <td rowspan="{cell['rowspan']}">
                    <div class="class-block" style="background-color:{cell['color']};">
                        {cell['subject']}<br>
                        {cell['professor']}<br>
                        {cell['location']}<br>
                        {cell['credits']}학점
                    </div>
                </td>\n"""
        html_content += "            </tr>\n"

    html_content += "        </tbody>\n    </table>\n</body>\n</html>"
    return html_content
