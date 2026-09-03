"""HTML 시각화 테스트 (30분 단위 슬롯 배치 회귀 테스트 포함)"""
import unittest
from datetime import time

from timetable_app.html_report import (
    _course_color,
    _slot_index,
    _slot_label,
    calculate_total_credits,
    generate_html,
)
from timetable_app.models import Course, TimeSlot


def _course(code, day, start, end, credits=3):
    return Course(
        code=code, name=f"과목{code}", professor="P", credits=credits,
        time_slots=[TimeSlot(day=day, start_time=time(*start), end_time=time(*end))],
        classroom="R1", capacity=30,
    )


class SlotIndexTests(unittest.TestCase):
    def test_on_the_hour(self):
        self.assertEqual(_slot_index(time(9, 0)), 0)
        self.assertEqual(_slot_label(_slot_index(time(9, 0))), "09:00")

    def test_half_hour(self):
        self.assertEqual(_slot_index(time(9, 30)), 1)
        self.assertEqual(_slot_label(_slot_index(time(9, 30))), "09:30")

    def test_end_of_day(self):
        self.assertEqual(_slot_label(_slot_index(time(18, 0))), "18:00")


class GenerateHtmlTests(unittest.TestCase):
    def test_half_hour_start_class_placed_and_sized_correctly(self):
        """09:30 시작 수업이 잘린 줄이 아니라 정확한 줄/길이로 렌더링돼야 한다 (회귀 테스트)."""
        course = _course("X1", day=0, start=(9, 30), end=(11, 0))  # 1.5시간
        html = generate_html([course])
        idx = html.find(">09:30<")
        self.assertNotEqual(idx, -1)
        snippet = html[idx:idx + 200]
        self.assertIn('rowspan="3"', snippet)  # 09:30, 10:00, 10:30 -> 11:00에 끝남

        # 09:00 줄에는 이 수업이 나타나면 안 됨 (예전 버그: 분을 버리고 09:00 줄에 배치)
        idx_9 = html.find(">09:00<")
        snippet_9 = html[idx_9:idx_9 + 200]
        self.assertNotIn(course.name, snippet_9)

    def test_on_the_hour_class_unaffected(self):
        course = _course("X2", day=0, start=(14, 0), end=(16, 0))  # 2시간, 정시
        html = generate_html([course])
        idx = html.find(">14:00<")
        snippet = html[idx:idx + 200]
        self.assertIn('rowspan="4"', snippet)  # 30분 그리드 기준 4칸 = 2시간

    def test_total_credits(self):
        courses = [_course("A", 0, (9, 0), (10, 0), credits=3), _course("B", 1, (9, 0), (10, 0), credits=2)]
        self.assertEqual(calculate_total_credits(courses), 5)

    def test_color_is_deterministic(self):
        c1 = _course("SAME", 0, (9, 0), (10, 0))
        c2 = _course("SAME", 1, (10, 0), (11, 0))
        self.assertEqual(_course_color(c1), _course_color(c2))

    def test_no_crash_on_empty_schedule(self):
        html = generate_html([])
        self.assertIn("<table", html)


if __name__ == "__main__":
    unittest.main()
