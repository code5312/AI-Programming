"""
데이터 모델

TimeSlot, Course, UserPreferences 데이터클래스와 관련 예외, 그리고
이들이 공유하는 시간표 도메인 규칙(수업 가능 시간대, 학점 범위 등)을 정의합니다.

다른 커리큘럼의 CSV/JSON으로 데이터를 교체할 때 유효성 검사 기준을 바꿔야 한다면
아래 상수만 조정하면 됩니다 (검증 로직 안에 흩어진 매직 넘버를 피하기 위함).
"""
from dataclasses import dataclass
from datetime import time
from typing import Any, Dict, List, Optional


class TimeSlotError(Exception):
    """시간 슬롯 관련 예외"""


class CourseError(Exception):
    """과목 관련 예외"""


# ---- 시간표 도메인 규칙 ----
DAY_MIN, DAY_MAX = 0, 4  # 0: 월요일 ~ 4: 금요일
DAY_NAMES = ["월", "화", "수", "목", "금"]
CLASS_TIME_MIN, CLASS_TIME_MAX = time(9, 0), time(23, 0)
# 50분 단위 교시제 학교(예: 가천대)는 한 교시(50분)짜리 단독 수업 블록이
# 실제로 존재한다(3학점 과목을 2교시+1교시로 나눠 배정하는 등) - 1시간
# 미만이라고 무시하면 실제 학사 데이터의 상당수가 거부된다.
CLASS_DURATION_MIN_HOURS, CLASS_DURATION_MAX_HOURS = 0.8, 4
COURSE_CREDITS_MIN, COURSE_CREDITS_MAX = 1, 3
TOTAL_CREDITS_MIN, TOTAL_CREDITS_MAX = 1, 21
DIFFICULTY_MIN, DIFFICULTY_MAX = 0.0, 1.0
RATING_MIN, RATING_MAX = 0.0, 5.0


@dataclass
class TimeSlot:
    """
    수업 시간 슬롯을 나타내는 클래스

    Attributes:
        day (int): 요일 (0: 월요일, 1: 화요일, ..., 4: 금요일)
        start_time (time): 시작 시간
        end_time (time): 종료 시간
    """
    day: int
    start_time: time
    end_time: time

    def __post_init__(self):
        """시간 슬롯의 유효성을 검증"""
        if not DAY_MIN <= self.day <= DAY_MAX:
            raise TimeSlotError(f"요일은 {DAY_MIN}(월)부터 {DAY_MAX}(금)까지여야 합니다.")

        if self.start_time >= self.end_time:
            raise TimeSlotError("시작 시간은 종료 시간보다 빨라야 합니다.")

        if not (CLASS_TIME_MIN <= self.start_time <= CLASS_TIME_MAX and
                CLASS_TIME_MIN <= self.end_time <= CLASS_TIME_MAX):
            raise TimeSlotError(
                f"수업 시간은 {CLASS_TIME_MIN.strftime('%H:%M')}부터 "
                f"{CLASS_TIME_MAX.strftime('%H:%M')} 사이여야 합니다."
            )

        duration = (self.end_time.hour - self.start_time.hour) + \
                  (self.end_time.minute - self.start_time.minute) / 60
        if not CLASS_DURATION_MIN_HOURS <= duration <= CLASS_DURATION_MAX_HOURS:
            raise TimeSlotError(
                f"수업 시간은 {CLASS_DURATION_MIN_HOURS}시간에서 "
                f"{CLASS_DURATION_MAX_HOURS}시간 사이여야 합니다."
            )


@dataclass
class Course:
    """
    과목 정보를 나타내는 클래스

    Attributes:
        code (str): 과목 코드
        name (str): 과목명
        professor (str): 교수명
        credits (int): 학점
        time_slots (List[TimeSlot]): 수업 시간 슬롯 목록
        classroom (str): 강의실
        capacity (int): 수용 인원
        current_enrolled (int): 현재 수강 인원
        difficulty (float): 난이도 (0-1)
        rating (float): 평점 (0-5)
        prerequisites (List[str]): 선수과목 코드 목록
    """
    code: str
    name: str
    professor: str
    credits: int
    time_slots: List[TimeSlot]
    classroom: str
    capacity: int
    current_enrolled: int = 0
    difficulty: float = 0.5
    rating: float = 3.0
    prerequisites: Optional[List[str]] = None

    def __post_init__(self):
        """과목 정보의 유효성을 검증"""
        self._validate_types()
        self._validate_required_fields()
        self._validate_ranges()
        self._validate_prerequisites()

    def _validate_types(self):
        """각 필드의 타입을 검증"""
        if not isinstance(self.credits, int):
            raise CourseError("학점은 정수여야 합니다.")
        if not isinstance(self.capacity, int):
            raise CourseError("수용 인원은 정수여야 합니다.")
        if not isinstance(self.current_enrolled, int):
            raise CourseError("현재 수강 인원은 정수여야 합니다.")
        if not isinstance(self.difficulty, (int, float)):
            raise CourseError("난이도는 숫자여야 합니다.")
        if not isinstance(self.rating, (int, float)):
            raise CourseError("평점은 숫자여야 합니다.")
        if not isinstance(self.time_slots, list):
            raise CourseError("시간 슬롯은 리스트여야 합니다.")
        if not all(isinstance(slot, TimeSlot) for slot in self.time_slots):
            raise CourseError("시간 슬롯은 TimeSlot 객체여야 합니다.")

    def _validate_required_fields(self):
        """필수 필드의 존재 여부를 검증"""
        if not all([self.code, self.name, self.professor, self.classroom]):
            raise CourseError("과목 코드, 이름, 교수, 강의실은 필수 입력사항입니다.")
        if not self.time_slots:
            raise CourseError("수업 시간이 지정되어야 합니다.")

    def _validate_ranges(self):
        """각 필드의 값 범위를 검증"""
        if not COURSE_CREDITS_MIN <= self.credits <= COURSE_CREDITS_MAX:
            raise CourseError(f"학점은 {COURSE_CREDITS_MIN}~{COURSE_CREDITS_MAX} 사이여야 합니다.")
        if self.capacity <= 0:
            raise CourseError("수용 인원은 0보다 커야 합니다.")
        if not 0 <= self.current_enrolled <= self.capacity:
            raise CourseError("현재 수강 인원은 0 이상이고 수용 인원 이하여야 합니다.")
        if not DIFFICULTY_MIN <= self.difficulty <= DIFFICULTY_MAX:
            raise CourseError(f"난이도는 {DIFFICULTY_MIN}부터 {DIFFICULTY_MAX} 사이여야 합니다.")
        if not RATING_MIN <= self.rating <= RATING_MAX:
            raise CourseError(f"평점은 {RATING_MIN}부터 {RATING_MAX} 사이여야 합니다.")

    def _validate_prerequisites(self):
        """선수과목 정보를 검증"""
        if self.prerequisites is None:
            self.prerequisites = []
        elif not isinstance(self.prerequisites, list):
            raise CourseError("선수과목은 리스트여야 합니다.")
        elif not all(isinstance(code, str) for code in self.prerequisites):
            raise CourseError("선수과목 코드는 문자열이어야 합니다.")

    def to_dict(self) -> Dict[str, Any]:
        """Course 객체를 딕셔너리로 변환"""
        return {
            "code": self.code,
            "name": self.name,
            "professor": self.professor,
            "credits": self.credits,
            "time_slots": [{
                "day": slot.day,
                "start_time": slot.start_time.strftime('%H:%M'),
                "end_time": slot.end_time.strftime('%H:%M')
            } for slot in self.time_slots],
            "classroom": self.classroom,
            "capacity": self.capacity,
            "current_enrolled": self.current_enrolled,
            "difficulty": self.difficulty,
            "rating": self.rating,
            "prerequisites": self.prerequisites
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Course':
        """딕셔너리에서 Course 객체 생성"""
        try:
            time_slots = []
            for slot in data['time_slots']:
                start_hour, start_min = map(int, slot['start_time'].split(':'))
                end_hour, end_min = map(int, slot['end_time'].split(':'))
                time_slots.append(TimeSlot(
                    day=slot['day'],
                    start_time=time(start_hour, start_min),
                    end_time=time(end_hour, end_min)
                ))

            return cls(
                code=data['code'],
                name=data['name'],
                professor=data['professor'],
                credits=data['credits'],
                time_slots=time_slots,
                classroom=data['classroom'],
                capacity=data['capacity'],
                current_enrolled=data.get('current_enrolled', 0),
                difficulty=data.get('difficulty', 0.5),
                rating=data.get('rating', 3.0),
                prerequisites=data.get('prerequisites', [])
            )
        except KeyError as e:
            raise CourseError(f"필수 필드가 누락되었습니다: {str(e)}")
        except ValueError as e:
            raise CourseError(f"데이터 형식이 올바르지 않습니다: {str(e)}")


@dataclass
class UserPreferences:
    """
    사용자의 시간표 선호도를 나타내는 클래스

    Attributes:
        min_credits (int): 최소 학점
        max_credits (int): 최대 학점
        preferred_days (List[int]): 선호 요일 목록
        preferred_professors (List[str]): 선호 교수 목록
        excluded_courses (List[str]): 제외할 과목 코드 목록
        preferred_difficulty (float): 선호 난이도 (0-1)
        preferred_rating (float): 선호 평점 (0-5)
    """
    min_credits: int
    max_credits: int
    preferred_days: List[int]
    preferred_professors: List[str]
    excluded_courses: List[str]
    preferred_difficulty: float = 0.5
    preferred_rating: float = 3.0

    def __post_init__(self):
        """선호도 정보의 유효성을 검증"""
        if not TOTAL_CREDITS_MIN <= self.min_credits <= self.max_credits <= TOTAL_CREDITS_MAX:
            raise ValueError(
                f"학점 범위가 올바르지 않습니다. "
                f"(최소 {TOTAL_CREDITS_MIN}학점, 최대 {TOTAL_CREDITS_MAX}학점)"
            )

        if not all(DAY_MIN <= day <= DAY_MAX for day in self.preferred_days):
            raise ValueError(f"선호 요일은 {DAY_MIN}(월)부터 {DAY_MAX}(금)까지여야 합니다.")
