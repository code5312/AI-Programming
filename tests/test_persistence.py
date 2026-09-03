"""courses.csv 파싱 및 저장된 시간표(schedules/*.json) 입출력 테스트"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from timetable_app import persistence
from timetable_app.models import Course, TimeSlot
from datetime import time

VALID_CSV = """code,name,professor,credits,classroom,capacity,current_enrolled,difficulty,rating,day,start_time,end_time,prerequisites,score
14395001,인공지능프로그래밍,박기성,3,AI공학관 407호,60,0,0.7,4.5,2,14:00,16:00,,0.85
14395001,인공지능프로그래밍,박기성,3,AI공학관 407호,60,0,0.7,4.5,3,15:00,16:00,,0.85
03671002,소프트웨어공학,김태규,3,AI관-407,60,0,0.5,3.0,0,09:00,11:00,,0.80
"""


def _write_csv(content: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


class LoadCoursesFromCsvTests(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            persistence.load_courses_from_csv("definitely_does_not_exist.csv")

    def test_missing_required_column_raises(self):
        path = _write_csv("code,name\nX,Y\n")
        try:
            with self.assertRaises(ValueError):
                persistence.load_courses_from_csv(path)
        finally:
            Path(path).unlink()

    def test_leading_zero_code_preserved(self):
        path = _write_csv(VALID_CSV)
        try:
            courses, _ = persistence.load_courses_from_csv(path)
            codes = [c.code for c in courses]
            self.assertIn("03671002", codes)  # 앞자리 0이 정수 변환으로 날아가지 않아야 함
        finally:
            Path(path).unlink()

    def test_multi_row_course_merges_time_slots(self):
        path = _write_csv(VALID_CSV)
        try:
            courses, _ = persistence.load_courses_from_csv(path)
            course = next(c for c in courses if c.code == "14395001")
            self.assertEqual(len(course.time_slots), 2)
        finally:
            Path(path).unlink()

    def test_score_column_builds_training_data(self):
        path = _write_csv(VALID_CSV)
        try:
            _, training_data = persistence.load_courses_from_csv(path)
            self.assertIsNotNone(training_data)
            self.assertEqual(set(training_data["code"]), {"14395001", "03671002"})
        finally:
            Path(path).unlink()

    def test_no_score_column_returns_none_training_data(self):
        csv_no_score = VALID_CSV.replace(",score", "").replace(",0.85\n", "\n").replace(",0.80\n", "\n")
        # score 값 컬럼을 아예 제거 (헤더 포함) - 마지막 컬럼이 prerequisites로 끝나도록 재작성
        csv_no_score = (
            "code,name,professor,credits,classroom,capacity,current_enrolled,difficulty,rating,day,start_time,end_time,prerequisites\n"
            "14395001,인공지능프로그래밍,박기성,3,AI공학관 407호,60,0,0.7,4.5,2,14:00,16:00,\n"
        )
        path = _write_csv(csv_no_score)
        try:
            _, training_data = persistence.load_courses_from_csv(path)
            self.assertIsNone(training_data)
        finally:
            Path(path).unlink()

    def test_inconsistent_duplicate_rows_still_loads_using_first_row(self):
        csv_content = (
            "code,name,professor,credits,classroom,capacity,day,start_time,end_time,rating\n"
            "X1,과목A,교수A,3,R1,30,0,09:00,10:00,4.5\n"
            "X1,과목A,교수A,3,R1,30,2,10:00,11:00,1.0\n"
        )
        path = _write_csv(csv_content)
        try:
            with self.assertLogs(level="WARNING") as log_ctx:
                courses, _ = persistence.load_courses_from_csv(path)
            self.assertEqual(len(courses), 1)
            self.assertEqual(courses[0].rating, 4.5)  # 첫 번째 행 값 사용
            self.assertTrue(any("rating" in msg for msg in log_ctx.output))
        finally:
            Path(path).unlink()

    def test_invalid_course_row_is_skipped_not_fatal(self):
        # credits=9는 models.py의 COURSE_CREDITS_MAX(3)를 벗어남 -> 로그만 남기고 건너뜀
        csv_content = (
            "code,name,professor,credits,classroom,capacity,day,start_time,end_time\n"
            "BAD,잘못된과목,교수,9,R1,30,0,09:00,10:00\n"
            "GOOD,정상과목,교수,3,R1,30,0,09:00,10:00\n"
        )
        path = _write_csv(csv_content)
        try:
            courses, _ = persistence.load_courses_from_csv(path)
            codes = [c.code for c in courses]
            self.assertNotIn("BAD", codes)
            self.assertIn("GOOD", codes)
        finally:
            Path(path).unlink()


class ScheduleSaveLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        patcher = patch.object(persistence, "SCHEDULES_DIR", Path(self.tmpdir.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmpdir.cleanup)

    def test_save_and_load_roundtrip(self):
        schedule = [
            Course(
                code="C1", name="과목1", professor="교수1", credits=3,
                time_slots=[TimeSlot(day=0, start_time=time(9, 0), end_time=time(10, 0))],
                classroom="R1", capacity=30,
            )
        ]
        persistence.save_schedule_json("my_schedule", schedule)
        loaded = persistence.load_schedule_json("my_schedule")
        self.assertEqual(loaded[0].code, "C1")

    def test_path_traversal_blocked(self):
        with self.assertRaises(ValueError):
            persistence.save_schedule_json("../../evil", [])

    def test_nested_subdirectory_name_is_allowed(self):
        # schedules/ 안에 머무르는 하위 경로("sub/evil")는 탈출이 아니므로 허용되어야 함
        path = persistence._safe_schedule_path("sub/evil")
        self.assertIn(Path(self.tmpdir.name).resolve(), path.parents)


if __name__ == "__main__":
    unittest.main()
