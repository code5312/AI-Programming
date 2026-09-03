"""
과목/시간표 데이터 입출력

- CSV(courses.csv): 과목 카탈로그 + 학습 데이터를 한 파일로 담는 입력 경로
- 저장된 시간표(schedules/*.json) 읽기/쓰기
"""
import json
import logging
import os
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import PROJECT_ROOT
from .models import Course, CourseError, TimeSlot, TimeSlotError

SCHEDULES_DIR = PROJECT_ROOT / "schedules"

CSV_REQUIRED_COLUMNS = [
    'code', 'name', 'professor', 'credits', 'classroom', 'capacity',
    'day', 'start_time', 'end_time',
]

# 같은 code로 묶인 여러 행 중 시간(day/start_time/end_time)을 제외한 나머지는
# 전부 같은 과목의 속성이므로 값이 같아야 한다. 다르면 첫 번째 행 값만 쓰지만
# (아래 로직), 사람이 엑셀에서 손으로 채우다 실수로 값을 다르게 적어도 조용히
# 넘어가지 않도록 여기서 경고를 남긴다.
_CONSISTENCY_COLUMNS = [
    'name', 'professor', 'credits', 'classroom', 'capacity',
    'current_enrolled', 'difficulty', 'rating', 'prerequisites', 'score',
]


def _warn_on_inconsistent_group(code: str, group: pd.DataFrame) -> None:
    if len(group) <= 1:
        return
    for column in _CONSISTENCY_COLUMNS:
        if column not in group.columns:
            continue
        values = group[column].dropna().unique().tolist()
        if len(values) > 1:
            logging.warning(
                f"과목 코드 '{code}'의 여러 행에서 '{column}' 값이 서로 다릅니다 {values} "
                "- 첫 번째 행의 값만 사용됩니다."
            )


def _parse_hhmm(value: Any) -> time:
    hour, minute = map(int, str(value).strip().split(':'))
    return time(hour, minute)


def load_courses_from_csv(csv_file: str) -> Tuple[List[Course], Optional[pd.DataFrame]]:
    """
    긴 형식(long format) CSV 한 파일에서 과목 카탈로그와 (있다면) 학습 데이터를 함께 로드.

    한 과목이 여러 시간에 수업하면 같은 code로 행을 여러 번 반복해서 적습니다
    (엑셀에서 행을 복사해 day/start_time/end_time만 바꾸면 됨).

    필수 컬럼: code, name, professor, credits, classroom, capacity, day, start_time, end_time
    선택 컬럼: current_enrolled, difficulty, rating, prerequisites(세미콜론 구분 문자열), score

    score 컬럼이 있고 값이 채워진 과목만 모아 학습용 DataFrame(code, score)을 만들어
    반환합니다. score 컬럼 자체가 없거나 값이 있는 과목이 하나도 없으면 None을 반환하며,
    이 경우 호출부는 모델 학습 없이(규칙 기반으로) 진행하면 됩니다.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_file}")

    df = pd.read_csv(csv_file, dtype={'code': str})

    missing_columns = [col for col in CSV_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {', '.join(missing_columns)}")

    courses: List[Course] = []
    score_rows: List[Dict[str, Any]] = []

    for code, group in df.groupby('code', sort=False):
        _warn_on_inconsistent_group(code, group)
        first = group.iloc[0]
        try:
            time_slots = []
            for _, row in group.iterrows():
                try:
                    time_slots.append(TimeSlot(
                        day=int(row['day']),
                        start_time=_parse_hhmm(row['start_time']),
                        end_time=_parse_hhmm(row['end_time']),
                    ))
                except Exception as e:
                    raise TimeSlotError(f"시간 슬롯 변환 실패: {str(e)}")

            prerequisites: List[str] = []
            if 'prerequisites' in group.columns and pd.notna(first.get('prerequisites')):
                prerequisites = [
                    p.strip() for p in str(first['prerequisites']).split(';') if p.strip()
                ]

            course = Course(
                code=str(code),
                name=str(first['name']),
                professor=str(first['professor']),
                credits=int(first['credits']),
                time_slots=time_slots,
                classroom=str(first['classroom']),
                capacity=int(first['capacity']),
                current_enrolled=(
                    int(first['current_enrolled'])
                    if 'current_enrolled' in group.columns and pd.notna(first.get('current_enrolled'))
                    else 0
                ),
                difficulty=(
                    float(first['difficulty'])
                    if 'difficulty' in group.columns and pd.notna(first.get('difficulty'))
                    else 0.5
                ),
                rating=(
                    float(first['rating'])
                    if 'rating' in group.columns and pd.notna(first.get('rating'))
                    else 3.0
                ),
                prerequisites=prerequisites,
            )
            courses.append(course)

            if 'score' in group.columns and pd.notna(first.get('score')):
                score_rows.append({'code': course.code, 'score': float(first['score'])})

        except (CourseError, TimeSlotError) as e:
            logging.error(f"과목 데이터 오류 (code={code}): {str(e)}")
        except Exception as e:
            logging.error(f"예상치 못한 오류 발생 (code={code}): {str(e)}")

    training_data = pd.DataFrame(score_rows) if score_rows else None
    return courses, training_data


def _safe_schedule_path(name: str) -> Path:
    """schedules 디렉터리 밖을 가리키는 이름(경로 조작)을 차단"""
    schedules_dir = SCHEDULES_DIR.resolve()
    target = (schedules_dir / f"{name}.json").resolve()
    if target != schedules_dir and schedules_dir not in target.parents:
        raise ValueError(f"올바르지 않은 시간표 이름입니다: {name}")
    return target


def save_schedule_json(name: str, schedule: List[Course]) -> None:
    """시간표를 schedules/{name}.json으로 저장"""
    path = _safe_schedule_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([course.to_dict() for course in schedule], f, ensure_ascii=False, indent=2)


def load_schedule_json(name: str) -> List[Course]:
    """저장된 시간표 불러오기"""
    path = _safe_schedule_path(name)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [Course.from_dict(course_data) for course_data in data]
