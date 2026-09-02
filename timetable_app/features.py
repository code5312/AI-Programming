"""
특성(feature) 추출

과목 정보를 AI 모델 입력용 수치 특성으로 변환합니다. 학습(preprocess_features)과
예측(predict_schedule_score) 양쪽에서 동일한 extract_all_features를 사용하므로,
두 경로의 특성 집합이 서로 어긋날 일이 없습니다.
"""
from typing import Any, Dict, List, Optional

from .models import Course, UserPreferences


def extract_basic_features(course: Course) -> Dict[str, Any]:
    """과목의 기본 특성 추출"""
    return {
        'credits': course.credits,
        'capacity': course.capacity,
        'current_enrolled': course.current_enrolled,
        'enrollment_ratio': course.current_enrolled / course.capacity,
        'total_hours': sum(
            (slot.end_time.hour - slot.start_time.hour) +
            (slot.end_time.minute - slot.start_time.minute) / 60
            for slot in course.time_slots
        ),
        'difficulty': course.difficulty,
        'rating': course.rating,
        'prerequisite_count': len(course.prerequisites)
    }


def extract_time_features(course: Course) -> Dict[str, Any]:
    """시간 관련 특성 추출"""
    time_features = {
        'morning_classes': 0,  # 9-12시
        'afternoon_classes': 0,  # 12-17시
        'evening_classes': 0,  # 17시 이후
        'total_days': len(set(slot.day for slot in course.time_slots)),
        'avg_duration': 0.0
    }

    durations = []
    for slot in course.time_slots:
        duration = (slot.end_time.hour - slot.start_time.hour) + \
                  (slot.end_time.minute - slot.start_time.minute) / 60
        durations.append(duration)

        if slot.start_time.hour < 12:
            time_features['morning_classes'] += 1
        elif slot.start_time.hour < 17:
            time_features['afternoon_classes'] += 1
        else:
            time_features['evening_classes'] += 1

    time_features['avg_duration'] = sum(durations) / len(durations)
    return time_features


def extract_professor_features(course: Course, all_courses: List[Course]) -> Dict[str, Any]:
    """교수 관련 특성 추출"""
    professor_courses = [c for c in all_courses if c.professor == course.professor]
    return {
        'professor_course_count': len(professor_courses),
        'professor_total_students': sum(c.current_enrolled for c in professor_courses),
        'professor_avg_rating': (
            sum(c.rating for c in professor_courses) / len(professor_courses)
            if professor_courses else 0
        )
    }


def extract_preference_features(course: Course, preferences: Optional[UserPreferences]) -> Dict[str, Any]:
    """학생 선호도 관련 특성 추출"""
    if not preferences:
        return {}

    return {
        # int로 캐스팅: bool 컬럼은 pandas/sklearn의 수치형 판별(select_dtypes)에서
        # 빠지기 때문에 다른 특성과 함께 정규화되지 않는 문제를 피하기 위함
        'matches_preferred_professor': int(course.professor in preferences.preferred_professors),
        'matches_preferred_days': int(any(
            slot.day in preferences.preferred_days
            for slot in course.time_slots
        )),
        'difficulty_match': 1 - abs(course.difficulty - preferences.preferred_difficulty),
        'rating_match': 1 - abs(course.rating - preferences.preferred_rating) / 5
    }


def extract_all_features(
    course: Course,
    all_courses: List[Course],
    preferences: Optional[UserPreferences],
) -> Dict[str, Any]:
    """한 과목의 전체 특성 벡터 생성 (학습/예측 공통 사용)"""
    return {
        **extract_basic_features(course),
        **extract_time_features(course),
        **extract_professor_features(course, all_courses),
        **extract_preference_features(course, preferences),
    }
