"""
모델 클래스들을 포함하는 패키지
"""

from .time_slot import TimeSlot, TimeSlotError
from .course import Course, CourseError
from .user_preferences import UserPreferences
from .schedule_recommender import ScheduleRecommender

__all__ = [
    'TimeSlot',
    'TimeSlotError',
    'Course',
    'CourseError',
    'UserPreferences',
    'ScheduleRecommender'
] 