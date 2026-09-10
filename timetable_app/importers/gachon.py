"""
가천대학교 정보시스템(info.gachon.ac.kr) 강의시간표 조회 API를 courses.csv
형식으로 변환하는 임포터.

info.gachon.ac.kr/Ssu1000q/mainSearch.do는 브라우저 개발자도구(Network 탭)로
역추적한 내부 엔드포인트로, 공식 문서화된 API가 아니다. 학과 코드로 그 학기
개설 과목 목록을 로그인 세션 없이 JSON으로 반환하는 것을 확인했다(캡처된
요청 쿠키에 인증 토큰 없이 구글 애널리틱스 쿠키만 있었음 - 즉 "개설강좌 조회"는
로그인이 필요 없는 공개 정보). 학교 쪽 개편으로 이 URL/파라미터가 언제든 바뀔
수 있으므로, 이 파일 하나만 고치면 나머지 앱(models/recommender/persistence)은
영향받지 않도록 격리했다 - 이 모듈의 결과물은 결국 기존 courses.csv와 동일한
긴 형식(long format) 행이라, persistence.load_courses_from_csv를 그대로 재사용한다.

사용 예:
    python -m timetable_app.importers.gachon --dept CS3120 CS2170 \\
        --year 2026 --term 20 --out courses.csv
"""
import argparse
import csv
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

MAIN_SEARCH_URL = "https://info.gachon.ac.kr/Ssu1000q/mainSearch.do"

_DAY_CODE = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
_TIME_TOKEN_RE = re.compile(r"([월화수목금토일])\s*(\d+)")

_HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://info.gachon.ac.kr/ssu/showTimetable.do",
    "Origin": "https://info.gachon.ac.kr",
    "User-Agent": "Mozilla/5.0",
}

# 응답에 없는 값(현재 수강인원/난이도/평점/선수과목/score)은 기존 courses.csv와
# 동일하게 비워두면 persistence.load_courses_from_csv가 알아서 기본값을 채운다.
CSV_COLUMNS = [
    "code", "name", "professor", "credits", "classroom", "capacity",
    "day", "start_time", "end_time", "department",
]


@dataclass
class GachonCourse:
    code: str
    name: str
    professor: str
    credits: int
    classroom: str
    capacity: int
    department: str
    slots: List[Tuple[int, time, time]]  # (day, start, end)


def _period_to_range(period: int) -> Tuple[time, time]:
    """가천대 교시표: N교시는 (8+N)시 정각에 시작해 50분간 진행 (1교시 09:00~09:50 등)."""
    hour = 8 + period
    return time(hour, 0), time(hour, 50)


def _parse_time_field(raw: str) -> List[Tuple[int, time, time]]:
    """
    '수8 ,수9 ,목9' 같은 문자열을 요일별로 모아, 같은 요일에 연속한 교시를
    하나의 시간 슬롯으로 합친 (요일, 시작, 종료) 목록으로 변환.

    가천대 API는 수업 시간을 50분 단위 교시로 쪼개 내려주는데, 같은 요일의
    연속 교시(예: 수8,수9)는 쉬는시간 없이 이어지는 하나의 수업 블록
    (16:00~17:50)이다. 교시 단위 그대로 각각 별도 슬롯으로 두면 안 되므로
    (예: 3교시 연속 수업이 50분짜리 3개로 쪼개짐), 연속 구간을 찾아 병합한다.
    """
    by_day: Dict[int, List[int]] = {}
    for day_char, period_str in _TIME_TOKEN_RE.findall(raw or ""):
        by_day.setdefault(_DAY_CODE[day_char], []).append(int(period_str))

    slots: List[Tuple[int, time, time]] = []
    for day, periods in by_day.items():
        periods.sort()
        run_start = run_end = periods[0]
        for period in periods[1:]:
            if period == run_end + 1:
                run_end = period
                continue
            start, _ = _period_to_range(run_start)
            _, end = _period_to_range(run_end)
            slots.append((day, start, end))
            run_start = run_end = period
        start, _ = _period_to_range(run_start)
        _, end = _period_to_range(run_end)
        slots.append((day, start, end))
    return slots


def _parse_credits(sisu: str) -> Optional[int]:
    """'3(3/0)' 형식(학점(이론/실습))에서 학점 숫자만 추출."""
    match = re.match(r"\s*(\d+)", sisu or "")
    return int(match.group(1)) if match else None


def fetch_department_courses(
    dept_cd: str,
    year: int,
    term: str,
    univ_cd: str = "CS0000",
    isu_cd: str = "002",
) -> List[dict]:
    """
    학과 코드 하나에 대해 그 학기 개설 과목 원본 레코드(dict) 목록을 가져온다.

    요청 파라미터 이름(`@d1#필드명`, `tp=dm` 등)은 이 시스템이 쓰는 WebSquare
    프레임워크의 데이터셋 제출 규약을 그대로 따른 것 - 브라우저가 실제로 보내는
    요청을 그대로 재현한 것이라 임의로 바꾸면 안 된다.
    """
    body = (
        f"%40d1%23groupType=20&%40d1%23searchYear={year}&%40d1%23searchTerm={term}"
        f"&%40d1%23searchUnivCD={univ_cd}&%40d1%23searchDeptCD={dept_cd}"
        f"&%40d1%23searchIsuCD={isu_cd}&%40d1%23searchGrade=&%40d1%23searchSubjectNm="
        f"&%40d%23=%40d1%23&%40d1%23=SendData&%40d1%23tp=dm&"
    ).encode("ascii")

    req = urllib.request.Request(MAIN_SEARCH_URL, data=body, headers=_HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("MainData", [])


def fetch_courses(
    dept_codes: Iterable[str],
    year: int,
    term: str,
    univ_cd: str = "CS0000",
) -> List[GachonCourse]:
    """
    여러 학과 코드에 대해 과목을 가져와 학수번호(HAKSU_NO) 기준으로 중복
    제거해 합친다.

    서로 다른 학과가 같은 융합 커리큘럼을 공유하면(예: 스마트보안학과와
    컴퓨터공학부(스마트보안전공)) 같은 과목이 여러 학과 코드로 동시에
    조회되는데, 학수번호가 곧 분반 단위 고유 식별자이므로 이걸로 합쳐야
    같은 과목이 courses.csv에 중복으로 들어가지 않는다.
    """
    by_code: Dict[str, GachonCourse] = {}
    for dept_cd in dept_codes:
        for raw in fetch_department_courses(dept_cd, year=year, term=term, univ_cd=univ_cd):
            code = raw.get("HAKSU_NO")
            if not code or code in by_code:
                continue
            credits = _parse_credits(raw.get("SISU", ""))
            slots = _parse_time_field(raw.get("TIME", ""))
            if credits is None or not slots:
                continue  # 학점/시간을 못 읽은 레코드는 건너뜀 (형식이 다른 특강 등으로 추정)
            by_code[code] = GachonCourse(
                code=code,
                name=(raw.get("SUBJECT_NM_KOR") or "").strip(),
                professor=raw.get("PROFNM") or "",
                credits=credits,
                classroom=raw.get("LOC_NM") or "",
                capacity=int(raw.get("APP_PEOPLE") or 0),
                department=raw.get("PRINT_DPT") or "",
                slots=slots,
            )
    return list(by_code.values())


def to_csv_rows(courses: Iterable[GachonCourse]) -> List[dict]:
    """
    긴 형식(long format) CSV 행으로 변환 - 한 과목의 시간 슬롯마다 한 행,
    나머지 컬럼은 반복 (persistence.load_courses_from_csv가 기대하는 형식과 동일).
    """
    rows = []
    for course in courses:
        for day, start, end in course.slots:
            rows.append({
                "code": course.code,
                "name": course.name,
                "professor": course.professor,
                "credits": course.credits,
                "classroom": course.classroom,
                "capacity": course.capacity,
                "day": day,
                "start_time": start.strftime("%H:%M"),
                "end_time": end.strftime("%H:%M"),
                "department": course.department,
            })
    return rows


def write_courses_csv(rows: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="가천대 강의시간표 -> courses.csv 변환")
    parser.add_argument("--dept", nargs="+", required=True, help="학과 코드 (예: CS3120 CS2170)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--term", required=True, help="학기 코드 (예: 20)")
    parser.add_argument("--univ", default="CS0000", help="단과대학 코드 (기본: CS0000)")
    parser.add_argument("--out", default="courses.csv", type=Path)
    args = parser.parse_args()

    courses = fetch_courses(args.dept, year=args.year, term=args.term, univ_cd=args.univ)
    rows = to_csv_rows(courses)
    write_courses_csv(rows, args.out)
    print(f"{len(courses)}개 과목, {len(rows)}개 시간 슬롯 -> {args.out}")


if __name__ == "__main__":
    main()
