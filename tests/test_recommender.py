"""
ScheduleRecommender 테스트

특히 이번 세션에서 실제로 고쳤던 버그들의 회귀 테스트를 포함한다:
- 카탈로그 적재 시 시간 충돌 과목을 거부하던 버그 (add_course)
- 모델이 없을 때 존재하지 않는 부모 클래스를 호출하던 버그 (generate_recommendations)
- training_data.csv를 행 순서로 매칭하던 버그 (train_model)
"""
import tempfile
import unittest
from datetime import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from timetable_app import PROJECT_ROOT
from timetable_app.models import Course, CourseError, TimeSlot, UserPreferences
from timetable_app.recommender import ScheduleRecommender


def _course(code, day, start, end, professor="P", credits=3, **kw):
    return Course(
        code=code, name=f"과목{code}", professor=professor, credits=credits,
        time_slots=[TimeSlot(day=day, start_time=time(*start), end_time=time(*end))],
        classroom="R1", capacity=30, **kw,
    )


class AddCourseTests(unittest.TestCase):
    def test_time_conflicting_courses_are_both_kept(self):
        """카탈로그 적재 단계에서는 시간 충돌 과목도 둘 다 들어가야 한다 (회귀 테스트)."""
        r = ScheduleRecommender()
        c1 = _course("A", day=0, start=(9, 0), end=(11, 0))
        c2 = _course("B", day=0, start=(10, 0), end=(12, 0))  # c1과 시간 겹침
        r.add_course(c1)
        r.add_course(c2)
        self.assertEqual(len(r.courses), 2)

    def test_duplicate_code_rejected(self):
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0)))
        with self.assertRaises(CourseError):
            r.add_course(_course("A", day=1, start=(9, 0), end=(10, 0)))


class CheckTimeConflictTests(unittest.TestCase):
    def setUp(self):
        self.r = ScheduleRecommender()

    def test_overlapping_same_day(self):
        c1 = _course("A", day=0, start=(9, 0), end=(11, 0))
        c2 = _course("B", day=0, start=(10, 0), end=(12, 0))
        self.assertTrue(self.r.check_time_conflict(c1, c2))

    def test_touching_boundaries_not_conflict(self):
        c1 = _course("A", day=0, start=(9, 0), end=(10, 0))
        c2 = _course("B", day=0, start=(10, 0), end=(11, 0))
        self.assertFalse(self.r.check_time_conflict(c1, c2))

    def test_different_days_not_conflict(self):
        c1 = _course("A", day=0, start=(9, 0), end=(11, 0))
        c2 = _course("B", day=1, start=(9, 0), end=(11, 0))
        self.assertFalse(self.r.check_time_conflict(c1, c2))


class TrainModelTests(unittest.TestCase):
    def test_matches_by_code_not_row_order(self):
        """training_data 행 순서가 과목 등록 순서와 달라도 code로 정확히 매칭돼야 한다 (회귀 테스트)."""
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0)))
        r.add_course(_course("B", day=1, start=(9, 0), end=(10, 0)))
        # CSV에는 B가 먼저, A가 나중 (등록 순서와 반대) + 매칭 안 되는 코드 하나 섞음
        training_data = pd.DataFrame([
            {"code": "B", "score": 0.9},
            {"code": "UNKNOWN", "score": 0.1},
            {"code": "A", "score": 0.5},
        ])
        with self.assertLogs(level="WARNING"):
            r.train_model(training_data)
        self.assertIsNotNone(r.model)

    def test_too_few_matches_raises(self):
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0)))
        training_data = pd.DataFrame([{"code": "A", "score": 0.5}])
        with self.assertRaises(ValueError):
            r.train_model(training_data)


class GenerateRecommendationsTests(unittest.TestCase):
    def test_requires_preferences(self):
        r = ScheduleRecommender()
        with self.assertRaises(ValueError):
            r.generate_recommendations()

    def test_no_model_falls_back_to_rule_based_without_crashing(self):
        """모델이 없을 때 존재하지 않는 부모 클래스의 super()를 부르던 버그의 회귀 테스트."""
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0)))
        r.add_course(_course("B", day=1, start=(9, 0), end=(10, 0)))
        r.set_user_preferences(UserPreferences(
            min_credits=3, max_credits=6,
            preferred_days=[], preferred_professors=[], excluded_courses=[],
        ))
        recs = r.generate_recommendations()  # model=None이어도 예외 없이 동작해야 함
        self.assertGreater(len(recs), 0)

    def test_respects_credit_range_and_no_conflicts(self):
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0), credits=3))
        r.add_course(_course("B", day=0, start=(9, 0), end=(10, 0), credits=3))  # A와 충돌
        r.add_course(_course("C", day=1, start=(9, 0), end=(10, 0), credits=3))
        r.set_user_preferences(UserPreferences(
            min_credits=6, max_credits=6,
            preferred_days=[], preferred_professors=[], excluded_courses=[],
        ))
        recs = r.generate_recommendations()
        for schedule, _score in recs:
            credits = sum(c.credits for c in schedule)
            self.assertEqual(credits, 6)
            codes = {c.code for c in schedule}
            self.assertFalse({"A", "B"} <= codes)  # A, B는 동시에 못 들어감

    def test_excluded_courses_never_appear(self):
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0)))
        r.add_course(_course("B", day=1, start=(9, 0), end=(10, 0)))
        r.set_user_preferences(UserPreferences(
            min_credits=1, max_credits=3,
            preferred_days=[], preferred_professors=[], excluded_courses=["A"],
        ))
        recs = r.generate_recommendations()
        for schedule, _score in recs:
            self.assertNotIn("A", {c.code for c in schedule})

    def test_empty_catalog_returns_empty(self):
        r = ScheduleRecommender()
        r.set_user_preferences(UserPreferences(
            min_credits=1, max_credits=21,
            preferred_days=[], preferred_professors=[], excluded_courses=[],
        ))
        self.assertEqual(r.generate_recommendations(), [])

    def test_large_catalog_routes_to_greedy_search(self):
        """후보가 EXACT_SEARCH_MAX_CANDIDATES를 넘으면 그리디 탐색으로 가야 한다."""
        r = ScheduleRecommender()
        threshold = ScheduleRecommender.EXACT_SEARCH_MAX_CANDIDATES
        for i in range(threshold + 3):
            r.add_course(_course(f"C{i}", day=i % 5, start=(9, 0), end=(10, 0), credits=1))
        r.set_user_preferences(UserPreferences(
            min_credits=1, max_credits=3,
            preferred_days=[], preferred_professors=[], excluded_courses=[],
        ))
        with patch.object(r, "_greedy_search", wraps=r._greedy_search) as spy_greedy, \
             patch.object(r, "_exact_search", wraps=r._exact_search) as spy_exact:
            r.generate_recommendations()
            spy_greedy.assert_called_once()
            spy_exact.assert_not_called()

    def test_small_catalog_routes_to_exact_search(self):
        r = ScheduleRecommender()
        r.add_course(_course("A", day=0, start=(9, 0), end=(10, 0)))
        r.set_user_preferences(UserPreferences(
            min_credits=1, max_credits=3,
            preferred_days=[], preferred_professors=[], excluded_courses=[],
        ))
        with patch.object(r, "_greedy_search", wraps=r._greedy_search) as spy_greedy, \
             patch.object(r, "_exact_search", wraps=r._exact_search) as spy_exact:
            r.generate_recommendations()
            spy_exact.assert_called_once()
            spy_greedy.assert_not_called()


class DataDirIsolationTests(unittest.TestCase):
    """
    ScheduleRecommender(data_dir=...) 격리 테스트.

    나중에 웹 등 여러 사용자가 동시에 쓰는 환경에서 사용자/세션별 data_dir을
    넘기게 되는데, 그때 한 사용자의 저장된 시간표가 다른 사용자에게 보이거나
    덮어써지면 안 된다는 것을 보장하는 회귀 테스트.
    """

    def test_default_data_dir_is_project_root(self):
        r = ScheduleRecommender()
        self.assertEqual(r.data_dir, PROJECT_ROOT)

    def test_two_recommenders_do_not_share_schedules(self):
        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            r_a = ScheduleRecommender(data_dir=Path(dir_a))
            r_b = ScheduleRecommender(data_dir=Path(dir_b))
            schedule = [_course("A", day=0, start=(9, 0), end=(10, 0))]

            r_a.save_schedule("my_schedule", schedule)

            self.assertTrue((Path(dir_a) / "schedules" / "my_schedule.json").exists())
            self.assertFalse((Path(dir_b) / "schedules" / "my_schedule.json").exists())
            with self.assertRaises(FileNotFoundError):
                r_b.load_schedule("my_schedule")


if __name__ == "__main__":
    unittest.main()
