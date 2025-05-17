from dataclasses import dataclass
from typing import List, Dict, Any
from .time_slot import TimeSlot
from datetime import time

class CourseError(Exception):
    """과목 관련 에러"""
    pass

@dataclass
class Course:
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
    prerequisites: List[str] = None
    course_type: str = "전공필수"
    year: int = 1

    def __post_init__(self):
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

        if not all([self.code, self.name, self.professor, self.classroom]):
            raise CourseError("과목 코드, 이름, 교수, 강의실은 필수 입력사항입니다.")
        
        if not 1 <= self.credits <= 3:
            raise CourseError("학점은 1~3 사이여야 합니다.")
        
        if not self.time_slots:
            raise CourseError("수업 시간이 지정되어야 합니다.")
        
        if self.capacity <= 0:
            raise CourseError("수용 인원은 0보다 커야 합니다.")
        if not 0 <= self.current_enrolled <= self.capacity:
            raise CourseError("현재 수강 인원은 0 이상이고 수용 인원 이하여야 합니다.")
        
        if not 0 <= self.difficulty <= 1:
            raise CourseError("난이도는 0부터 1 사이여야 합니다.")
        if not 0 <= self.rating <= 5:
            raise CourseError("평점은 0부터 5 사이여야 합니다.")
        
        if self.prerequisites is None:
            self.prerequisites = []
        elif not isinstance(self.prerequisites, list):
            raise CourseError("선수과목은 리스트여야 합니다.")
        elif not all(isinstance(code, str) for code in self.prerequisites):
            raise CourseError("선수과목 코드는 문자열이어야 합니다.")

        if not isinstance(self.course_type, str):
            raise CourseError("과목 유형은 문자열이어야 합니다.")

        if not isinstance(self.year, int) or not 1 <= self.year <= 4:
            raise CourseError("학년은 1에서 4 사이의 정수여야 합니다.")

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
            "prerequisites": self.prerequisites,
            "course_type": self.course_type,
            "year": self.year
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
                prerequisites=data.get('prerequisites', []),
                course_type=data.get('course_type', '전공필수'),
                year=data.get('year', 1)
            )
        except KeyError as e:
            raise CourseError(f"필수 필드가 누락되었습니다: {str(e)}")
        except ValueError as e:
            raise CourseError(f"데이터 형식이 올바르지 않습니다: {str(e)}") 