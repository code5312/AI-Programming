from dataclasses import dataclass
from typing import List
from .time_slot import TimeSlot
from datetime import time

@dataclass
class PreferredTimeRange:
    day: int
    start_time: time
    end_time: time

@dataclass
class UserPreferences:
    min_credits: int
    max_credits: int
    preferred_days: List[int]
    preferred_times: List[PreferredTimeRange]
    preferred_professors: List[str]
    excluded_courses: List[str]
    preferred_difficulty: float = 0.5
    preferred_rating: float = 3.0
    must_take_courses: List[str] = None
    avoid_courses: List[str] = None
    preferred_classrooms: List[str] = None

    def __post_init__(self):
        if self.must_take_courses is None:
            self.must_take_courses = []
        if self.avoid_courses is None:
            self.avoid_courses = []
        if self.preferred_classrooms is None:
            self.preferred_classrooms = []
        if not 1 <= self.min_credits <= self.max_credits <= 21:
            raise ValueError("학점 범위가 올바르지 않습니다.")
        if not all(0 <= day <= 4 for day in self.preferred_days):
            raise ValueError("선호 요일은 0(월)부터 4(금)까지여야 합니다.") 