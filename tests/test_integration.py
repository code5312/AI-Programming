"""
실제 courses.csv를 사용한 end-to-end 통합 테스트.

프로젝트 루트의 courses.csv가 깨지면(스키마 오류, 시간 충돌 데이터 등) 여기서
바로 드러나야 한다는 취지의 회귀 테스트.
"""
import unittest

from timetable_app import PROJECT_ROOT
from timetable_app.html_report import generate_html
from timetable_app.models import UserPreferences
from timetable_app.recommender import ScheduleRecommender


class RealCoursesCsvTests(unittest.TestCase):
    def test_full_pipeline_runs_without_error(self):
        csv_path = PROJECT_ROOT / "courses.csv"
        if not csv_path.exists():
            self.skipTest("courses.csv가 없어 통합 테스트를 건너뜁니다.")

        r = ScheduleRecommender()
        training_data = r.load_courses_from_csv(str(csv_path))
        self.assertGreater(len(r.courses), 0)

        if training_data is not None and not training_data.empty:
            r.train_model(training_data)

        r.set_user_preferences(UserPreferences(
            min_credits=6, max_credits=12,
            preferred_days=[2, 3], preferred_professors=[],
            excluded_courses=[],
        ))
        recs = r.generate_recommendations()
        self.assertGreater(len(recs), 0)

        best_schedule, score = recs[0]
        # 상위 추천이 학점 범위를 지키고 서로 시간 충돌이 없어야 함
        credits = sum(c.credits for c in best_schedule)
        self.assertTrue(6 <= credits <= 12)
        for i in range(len(best_schedule)):
            for j in range(i + 1, len(best_schedule)):
                self.assertFalse(r.check_time_conflict(best_schedule[i], best_schedule[j]))

        html = generate_html(best_schedule, score)
        self.assertIn("<table", html)


if __name__ == "__main__":
    unittest.main()
