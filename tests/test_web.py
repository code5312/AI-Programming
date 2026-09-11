"""
Flask 웹 레이어 테스트.

핵심 검증 포인트는 서로 다른 브라우저 세션(쿠키)이 저장 시간표를 공유하지
않는다는 것 - clone_with_data_dir() 기반 세션 격리(2번 작업)가 웹 계층에서도
실제로 동작하는지 확인하는 통합 테스트. CSV 업로드 테스트도 마찬가지로 한
세션의 업로드가 다른 세션에 새어나가지 않는지를 핵심으로 본다.
"""
import io
import shutil
import unittest

from timetable_app.web import WEB_DATA_DIR, create_app

_CUSTOM_CSV = (
    "code,name,professor,credits,classroom,capacity,day,start_time,end_time\n"
    "X001,UploadedCourseOne,ProfKim,3,R101,30,0,09:00,10:50\n"
    "X002,UploadedCourseTwo,ProfLee,3,R102,30,1,09:00,10:50\n"
).encode("utf-8")

_INVALID_CSV = b"code,name\nX001,Foo\n"  # 필수 컬럼(professor 등) 누락

_XSS_PAYLOAD = "<script>alert(1)</script>"
_XSS_CSV = (
    "code,name,professor,credits,classroom,capacity,day,start_time,end_time\n"
    f"X001,{_XSS_PAYLOAD},ProfKim,3,R101,30,0,09:00,10:50\n"
).encode("utf-8")


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

    def test_unknown_route_returns_themed_404(self):
        resp = self._client().get("/this-page-does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("페이지를 찾을 수 없습니다".encode(), resp.data)

    def test_favicon_and_og_tags_present(self):
        resp = self._client().get("/")
        self.assertIn(b'rel="icon"', resp.data)
        self.assertIn(b'property="og:title"', resp.data)

    def test_favicon_present_on_recommend_page(self):
        """generate_html() 기반 페이지에도 파비콘이 주입되는지 확인."""
        client = self._client()
        resp = client.post("/recommend", data={"min_credits": "6", "max_credits": "12"})
        self.assertIn(b'rel="icon"', resp.data)

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


class XssTests(unittest.TestCase):
    """
    폼/업로드/저장 이름처럼 사용자가 직접 넣는 값이 그대로 HTML에 echo되면
    스크립트 삽입(XSS)으로 이어진다. 이스케이프 처리가 실제로 되고 있는지
    회귀 테스트로 고정해둔다.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.testing = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WEB_DATA_DIR, ignore_errors=True)

    def _client(self):
        return self.app.test_client()

    def test_invalid_input_error_message_is_escaped(self):
        """int() 변환 실패 메시지에 그대로 담기는 사용자 입력값이 이스케이프돼야 한다."""
        resp = self._client().post("/recommend", data={
            "min_credits": _XSS_PAYLOAD, "max_credits": "12",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn(_XSS_PAYLOAD.encode(), resp.data)
        self.assertIn(b"&lt;script&gt;", resp.data)

    def test_saved_schedule_name_is_escaped(self):
        # <, > 등은 Windows 파일시스템이 파일명으로 거부해서(테스트 환경에
        # 따라 저장 성공 여부가 달라짐) 필터명은 플랫폼 어디서나 유효하되
        # HTML 이스케이프는 꼭 필요한 &/' 를 쓴다.
        name = "Tom & Jerry's course"
        client = self._client()
        client.post("/recommend", data={"min_credits": "6", "max_credits": "12"})

        resp = client.post("/save", data={"name": name})
        self.assertNotIn(b"Tom & Jerry's", resp.data)
        self.assertIn(b"Tom &amp; Jerry&#x27;s", resp.data)

        resp = client.get("/schedules")
        self.assertNotIn(b"Tom & Jerry's", resp.data)
        self.assertIn(b"Tom &amp; Jerry&#x27;s", resp.data)

    def test_uploaded_course_name_is_escaped_in_timetable_html(self):
        client = self._client()
        client.post("/upload", data={"csv_file": (io.BytesIO(_XSS_CSV), "evil.csv")},
                     content_type="multipart/form-data")

        resp = client.post("/recommend", data={"min_credits": "1", "max_credits": "6"})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(_XSS_PAYLOAD.encode(), resp.data)
        self.assertIn(b"&lt;script&gt;", resp.data)


class UploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.testing = True

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WEB_DATA_DIR, ignore_errors=True)

    def _client(self):
        return self.app.test_client()

    def _upload(self, client, content: bytes, filename: str = "my_courses.csv"):
        return client.post(
            "/upload",
            data={"csv_file": (io.BytesIO(content), filename)},
            content_type="multipart/form-data",
        )

    def test_valid_csv_upload_succeeds_and_switches_catalog(self):
        client = self._client()
        resp = self._upload(client, _CUSTOM_CSV)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("2개 과목".encode(), resp.data)

        # 업로드 후 폼 페이지에 "기본 데이터로 돌아가기" 배너가 보여야 함
        resp = client.get("/")
        self.assertIn("기본 데이터로 돌아가기".encode(), resp.data)

        # 추천 결과에 업로드한 과목명이 실제로 등장해야 함 (기본 카탈로그가
        # 아니라 업로드한 카탈로그로 추천했다는 증거)
        resp = client.post("/recommend", data={"min_credits": "1", "max_credits": "6"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"UploadedCourse", resp.data)

    def test_missing_required_columns_rejected(self):
        client = self._client()
        resp = self._upload(client, _INVALID_CSV)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("CSV를 읽을 수 없습니다".encode(), resp.data)

    def test_non_csv_extension_rejected(self):
        client = self._client()
        resp = self._upload(client, b"not a csv", filename="courses.txt")
        self.assertEqual(resp.status_code, 400)

    def test_no_file_rejected(self):
        client = self._client()
        resp = client.post("/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_reset_data_reverts_to_default_catalog(self):
        client = self._client()
        self._upload(client, _CUSTOM_CSV)
        client.post("/reset-data")

        resp = client.get("/")
        self.assertNotIn("기본 데이터로 돌아가기".encode(), resp.data)

        resp = client.post("/recommend", data={"min_credits": "6", "max_credits": "12"})
        self.assertNotIn(b"UploadedCourse", resp.data)

    def test_uploaded_catalog_survives_memory_cache_miss(self):
        """
        gunicorn 멀티 워커 흉내: 업로드를 처리한 뒤 메모리 캐시가 비어도(다른
        워커로 요청이 갔다고 가정) 디스크에 저장된 CSV로 다시 살아나야 한다.
        """
        client = self._client()
        self._upload(client, _CUSTOM_CSV)

        self.app.uploaded_recommenders.clear()  # 이 워커의 메모리 캐시만 비움 (디스크 파일은 그대로)

        resp = client.get("/")
        self.assertIn("기본 데이터로 돌아가기".encode(), resp.data)

        resp = client.post("/recommend", data={"min_credits": "1", "max_credits": "6"})
        self.assertIn(b"UploadedCourse", resp.data)

    def test_uploaded_catalog_does_not_leak_to_other_sessions(self):
        client_a = self._client()
        client_b = self._client()

        self._upload(client_a, _CUSTOM_CSV)

        resp_a = client_a.post("/recommend", data={"min_credits": "1", "max_credits": "6"})
        self.assertIn(b"UploadedCourse", resp_a.data)

        resp_b = client_b.post("/recommend", data={"min_credits": "6", "max_credits": "12"})
        self.assertNotIn(b"UploadedCourse", resp_b.data)


if __name__ == "__main__":
    unittest.main()
