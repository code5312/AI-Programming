from typing import List, Dict, Any
from ..models.course import Course
import os
from datetime import time

def generate_html_timetable(schedule: List[Dict[str, Any]], output_file: str = "timetable.html") -> None:
    """시간표를 HTML 형식으로 생성"""
    days = ["월", "화", "수", "목", "금"]
    time_slots = [
        (time(9, 0), time(10, 30)),
        (time(10, 30), time(12, 0)),
        (time(12, 0), time(13, 30)),
        (time(13, 30), time(15, 0)),
        (time(15, 0), time(16, 30)),
        (time(16, 30), time(18, 0))
    ]

    # HTML 템플릿 시작
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>시간표</title>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
            }
            .course {
                background-color: #e6f3ff;
                padding: 5px;
                margin: 2px;
                border-radius: 3px;
            }
            .course-info {
                font-size: 0.9em;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>시간표</h1>
        <table>
            <tr>
                <th>시간</th>
    """

    # 요일 헤더 추가
    for day in days:
        html += f"<th>{day}요일</th>"
    html += "</tr>"

    # 시간대별로 셀 생성
    for start_time, end_time in time_slots:
        html += "<tr>"
        html += f"<td>{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}</td>"
        
        for day in range(5):  # 0-4: 월-금
            cell_content = ""
            for course in schedule:
                for slot in course['time_slots']:
                    if slot['day'] == day:
                        slot_start = time(*map(int, slot['start_time'].split(':')))
                        slot_end = time(*map(int, slot['end_time'].split(':')))
                        if slot_start <= start_time and slot_end >= end_time:
                            cell_content += f"""
                            <div class="course">
                                <div>{course['name']}</div>
                                <div class="course-info">
                                    {course['professor']}<br>
                                    {course['classroom']}<br>
                                    {course['credits']}학점
                                </div>
                            </div>
                            """
            html += f"<td>{cell_content}</td>"
        html += "</tr>"

    # HTML 템플릿 종료
    html += """
        </table>
    </body>
    </html>
    """

    # 파일 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

def generate_html(schedule: List[Dict[str, Any]], output_file: str = "timetable.html") -> None:
    """generate_html_timetable의 별칭으로 사용"""
    generate_html_timetable(schedule, output_file) 