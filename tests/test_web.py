"""
Flask 웹 레이어 테스트.

핵심 검증 포인트는 서로 다른 브라우저 세션(쿠키)이 저장 시간표를 공유하지
않는다는 것 - clone_with_data_dir() 기반 세션 격리(2번 작업)가 웹 계층에서도
실제로 동작하는지 확인하는 통합 테스트.
"""
import shutil
import unittest

from timetable_app.web import WEB_DATA_DIR, create_app


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.testing = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WEB_DATA_DIR, ignore_errors=True)

    def _client(self):
        return self.app.test_client()

    def test_index_shows_form(self):
        resp = self._client().get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<form", resp.data)

    def test_recommend_returns_timetable(self):
        client = self._client()
        resp = client.post("/recommend", data={
            "min_credits": "6", "max_credits": "12",
            "preferred_professors": "", "excluded_courses": "",
            "preferred_difficulty": "0.5", "preferred_rating": "3.0",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<table", resp.data)

    def test_recommend_with_invalid_credit_range_shows_error(self):
        client = self._client()
        resp = client.post("/recommend", data={
            "min_credits": "20", "max_credits": "5",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("오류".encode(), resp.data)

    def test_save_and_view_roundtrip(self):
        client = self._client()
        client.post("/recommend", data={"min_credits": "6", "max_credits": "12"})
        resp = client.post("/save", data={"name": "my_schedule"}, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)

        resp = client.get("/schedules/my_schedule")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"<table", resp.data)

    def test_sessions_do_not_share_saved_schedules(self):
        """서로 다른 두 세션(쿠키)이 저장한 시간표는 서로 안 보여야 한다."""
        client_a = self._client()
        client_b = self._client()

        client_a.post("/recommend", data={"min_credits": "6", "max_credits": "12"})
        client_a.post("/save", data={"name": "shared_name"})

        client_b.post("/recommend", data={"min_credits": "6", "max_credits": "12"})
        resp = client_b.get("/schedules/shared_name")
        self.assertEqual(resp.status_code, 404)

        resp = client_a.get("/schedules/shared_name")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
