"""
가천대 강의시간표 임포터 테스트.

실제 학교 서버에 네트워크 요청을 보내지 않고(CI에서 외부 서버를 두드리면 안
되므로) 순수 파싱/변환 로직만 검증한다. fetch_department_courses()는
unittest.mock으로 대체한다.
"""
import unittest
from datetime import time
from unittest.mock import patch

from timetable_app.importers.gachon import (
    GachonCourse,
    _parse_credits,
    _parse_time_field,
    fetch_courses,
    to_csv_rows,
)


class ParseTimeFieldTests(unittest.TestCase):
    def test_single_isolated_period(self):
        """연속되지 않는 단독 교시는 50분짜리 슬롯 하나."""
        slots = _parse_time_field("목9")
        self.assertEqual(slots, [(3, time(17, 0), time(17, 50))])

    def test_consecutive_periods_on_same_day_are_merged(self):
        """같은 요일의 연속 교시(8,9)는 쉬는시간 없이 이어지는 한 블록으로 병합."""
        slots = _parse_time_field("수8 ,수9 ,목9")
        self.assertEqual(slots, [
            (2, time(16, 0), time(17, 50)),
            (3, time(17, 0), time(17, 50)),
        ])

    def test_non_consecutive_periods_on_same_day_stay_separate(self):
        """같은 요일이라도 사이가 뜨면(중간에 쉬는 교시) 별도 슬롯이어야 한다."""
        slots = _parse_time_field("월1 ,월2 ,월5")
        self.assertEqual(slots, [
            (0, time(9, 0), time(10, 50)),
            (0, time(13, 0), time(13, 50)),
        ])

    def test_empty_string_returns_no_slots(self):
        self.assertEqual(_parse_time_field(""), [])


class ParseCreditsTests(unittest.TestCase):
    def test_extracts_leading_number(self):
        self.assertEqual(_parse_credits("3(3/0)"), 3)
        self.assertEqual(_parse_credits("1(0/2)"), 1)

    def test_invalid_format_returns_none(self):
        self.assertIsNone(_parse_credits(""))
        self.assertIsNone(_parse_credits(None))


class FetchCoursesDedupeTests(unittest.TestCase):
    """서로 다른 학과 코드로 같은 학수번호가 중복 조회돼도 한 번만 남아야 한다."""

    def test_same_course_from_two_departments_is_merged_once(self):
        shared_record = {
            "HAKSU_NO": "07251002", "SUBJECT_NM_KOR": "데이터통신", "PROFNM": "박기성",
            "SISU": "3(3/0)", "LOC_NM": "제3생활관-B130", "APP_PEOPLE": 60,
            "TIME": "수8 ,수9 ,목9", "PRINT_DPT": "컴퓨터공학부(스마트보안전공)/스마트보안학과2",
        }
        with patch(
            "timetable_app.importers.gachon.fetch_department_courses",
            return_value=[shared_record],
        ) as mocked:
            courses = fetch_courses(["CS3120", "CS2170"], year=2026, term="20")

        self.assertEqual(mocked.call_count, 2)  # 두 학과 다 조회는 하되
        self.assertEqual(len(courses), 1)  # 결과는 한 번만 남음
        self.assertEqual(courses[0].code, "07251002")

    def test_records_missing_credits_or_time_are_skipped(self):
        bad_records = [
            {"HAKSU_NO": "A1", "SISU": "", "TIME": "월1"},  # 학점 파싱 실패
            {"HAKSU_NO": "A2", "SISU": "3(3/0)", "TIME": ""},  # 시간 파싱 실패
        ]
        with patch(
            "timetable_app.importers.gachon.fetch_department_courses",
            return_value=bad_records,
        ):
            courses = fetch_courses(["CS3120"], year=2026, term="20")
        self.assertEqual(courses, [])


class ToCsvRowsTests(unittest.TestCase):
    def test_one_row_per_slot_shares_other_columns(self):
        course = GachonCourse(
            code="X1", name="테스트과목", professor="P", credits=3,
            classroom="R1", capacity=30, department="테스트학과",
            slots=[(0, time(9, 0), time(10, 50)), (2, time(9, 0), time(9, 50))],
        )
        rows = to_csv_rows([course])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["code"] == "X1" and r["credits"] == 3 for r in rows))
        self.assertEqual(rows[0]["day"], 0)
        self.assertEqual(rows[0]["start_time"], "09:00")
        self.assertEqual(rows[0]["end_time"], "10:50")


if __name__ == "__main__":
    unittest.main()
