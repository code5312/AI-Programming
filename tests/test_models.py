"""TimeSlot / Course / UserPreferences 검증 규칙 테스트"""
import unittest
from datetime import time

from timetable_app.models import (
    Course,
    CourseError,
    TimeSlot,
    TimeSlotError,
    UserPreferences,
)


class TimeSlotTests(unittest.TestCase):
    def test_valid_slot(self):
        slot = TimeSlot(day=0, start_time=time(9, 0), end_time=time(10, 0))
        self.assertEqual(slot.day, 0)

    def test_day_out_of_range(self):
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=5, start_time=time(9, 0), end_time=time(10, 0))

    def test_start_after_end(self):
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=0, start_time=time(10, 0), end_time=time(9, 0))

    def test_start_equals_end(self):
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=0, start_time=time(9, 0), end_time=time(9, 0))

    def test_before_business_hours(self):
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=0, start_time=time(8, 0), end_time=time(9, 0))

    def test_after_business_hours(self):
        # CLASS_TIME_MAX(23:00)를 넘어가는 경우 (50분 교시제 학교의 야간수업
        # 14교시가 22:00~22:50까지 있어 야간수업 자체는 허용 범위 안에 있음)
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=0, start_time=time(22, 30), end_time=time(23, 50))

    def test_duration_too_short(self):
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=0, start_time=time(9, 0), end_time=time(9, 30))

    def test_duration_too_long(self):
        with self.assertRaises(TimeSlotError):
            TimeSlot(day=0, start_time=time(9, 0), end_time=time(14, 0))

    def test_half_hour_start_is_allowed(self):
        # 09:30 시작처럼 정시가 아닌 슬롯도 유효해야 함 (html_report 렌더링 버그의 전제 조건)
        slot = TimeSlot(day=0, start_time=time(9, 30), end_time=time(11, 0))
        self.assertEqual(slot.start_time, time(9, 30))


def _make_course(**overrides):
    defaults = dict(
        code="C001",
        name="테스트과목",
        professor="교수",
        credits=3,
        time_slots=[TimeSlot(day=0, start_time=time(9, 0), end_time=time(10, 0))],
        classroom="R1",
        capacity=30,
    )
    defaults.update(overrides)
    return Course(**defaults)


class CourseTests(unittest.TestCase):
    def test_valid_course_defaults(self):
        course = _make_course()
        self.assertEqual(course.difficulty, 0.5)
        self.assertEqual(course.rating, 3.0)
        self.assertEqual(course.prerequisites, [])

    def test_credits_out_of_range(self):
        with self.assertRaises(CourseError):
            _make_course(credits=4)

    def test_capacity_not_positive(self):
        with self.assertRaises(CourseError):
            _make_course(capacity=0)

    def test_current_enrolled_exceeds_capacity(self):
        with self.assertRaises(CourseError):
            _make_course(current_enrolled=100, capacity=30)

    def test_difficulty_out_of_range(self):
        with self.assertRaises(CourseError):
            _make_course(difficulty=1.5)

    def test_rating_out_of_range(self):
        with self.assertRaises(CourseError):
            _make_course(rating=5.5)

    def test_missing_required_field(self):
        with self.assertRaises(CourseError):
            _make_course(name="")

    def test_no_time_slots(self):
        with self.assertRaises(CourseError):
            _make_course(time_slots=[])

    def test_to_dict_from_dict_roundtrip(self):
        course = _make_course()
        restored = Course.from_dict(course.to_dict())
        self.assertEqual(restored.code, course.code)
        self.assertEqual(len(restored.time_slots), len(course.time_slots))
        self.assertEqual(restored.time_slots[0].start_time, course.time_slots[0].start_time)


class UserPreferencesTests(unittest.TestCase):
    def test_valid_preferences(self):
        prefs = UserPreferences(
            min_credits=9, max_credits=18,
            preferred_days=[0, 2], preferred_professors=["김교수"],
            excluded_courses=[],
        )
        self.assertEqual(prefs.min_credits, 9)

    def test_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            UserPreferences(
                min_credits=18, max_credits=9,
                preferred_days=[], preferred_professors=[], excluded_courses=[],
            )

    def test_max_over_total_limit(self):
        with self.assertRaises(ValueError):
            UserPreferences(
                min_credits=1, max_credits=22,
                preferred_days=[], preferred_professors=[], excluded_courses=[],
            )

    def test_invalid_preferred_day(self):
        with self.assertRaises(ValueError):
            UserPreferences(
                min_credits=1, max_credits=21,
                preferred_days=[5], preferred_professors=[], excluded_courses=[],
            )


if __name__ == "__main__":
    unittest.main()
