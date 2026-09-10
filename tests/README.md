# tests

표준 라이브러리 `unittest` 기반 테스트 스위트. 추가 설치 없이 실행 가능하며, push/PR마다 GitHub Actions([`.github/workflows/tests.yml`](../.github/workflows/tests.yml))에서 자동으로 돈다.

```bash
python -m unittest discover -s tests -t .
```

## 파일 구성

| 파일 | 검증 대상 |
|---|---|
| `test_models.py` | `TimeSlot`/`Course`/`UserPreferences`의 유효성 검증 규칙 (요일·시간·학점 범위 등) |
| `test_persistence.py` | `courses.csv` 파싱, 저장된 시간표(JSON) 저장/불러오기, 경로 조작 방지 |
| `test_recommender.py` | 카탈로그 관리, 모델 학습, 시간표 탐색(백트래킹/그리디 전환), 세션별 `data_dir` 격리 (`features.py`는 여기서 추천 결과를 통해 간접 검증됨) |
| `test_html_report.py` | 30분 단위 그리드 배치, 색상 결정성 등 HTML 렌더링 |
| `test_web.py` | Flask 라우트 동작, 특히 **서로 다른 브라우저 세션(쿠키)이 저장 시간표/업로드한 카탈로그를 공유하지 않는지**, 그리고 업로드 카탈로그가 **인메모리 캐시 미스 후에도(멀티 워커 흉내) 디스크에서 복원되는지** |
| `test_gachon_importer.py` | 가천대 임포터의 교시 병합/학점 파싱/학과 간 중복 제거 (네트워크 요청 없음) |
| `test_integration.py` | 실제 `courses.csv`로 카탈로그 로드 → 추천 생성 → HTML 렌더링까지 end-to-end 확인 |

## 원칙

- **외부 네트워크 요청 금지**: `test_gachon_importer.py`는 `unittest.mock.patch`로 `fetch_department_courses`를 대체해 실제 학교 서버를 두드리지 않는다. 새 임포터를 테스트할 때도 이 패턴을 따를 것.
- **회귀 테스트 우선**: 실제로 발견/수정했던 버그는 대부분 회귀 테스트로 남겨져 있다 (각 테스트의 docstring에 어떤 버그였는지 적혀 있음).
