"""
시간표 추천 시스템

이 모듈은 학생의 선호도를 기반으로 최적의 시간표를 추천하는 시스템을 구현합니다.
AI 모델을 사용하여 과목의 특성과 학생의 선호도를 고려한 시간표를 생성합니다.

주요 기능:
- 과목 정보 관리 및 검증
- 학생 선호도 설정
- AI 기반 시간표 추천
- 시간표 시각화
"""

import json
import os
import webbrowser
from datetime import time, datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timetable.log'),
        logging.StreamHandler()
    ]
)

class TimeSlotError(Exception):
    """시간 슬롯 관련 예외"""
    pass

class CourseError(Exception):
    """과목 관련 예외"""
    pass

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
        # 요일 검증
        if not 0 <= self.day <= 4:
            raise TimeSlotError("요일은 0(월)부터 4(금)까지여야 합니다.")
        
        # 시간 순서 검증
        if self.start_time >= self.end_time:
            raise TimeSlotError("시작 시간은 종료 시간보다 빨라야 합니다.")
        
        # 수업 시간 범위 검증 (9시-18시)
        if not (time(9, 0) <= self.start_time <= time(18, 0) and 
                time(9, 0) <= self.end_time <= time(18, 0)):
            raise TimeSlotError("수업 시간은 9시부터 18시 사이여야 합니다.")
        
        # 수업 시간 길이 검증 (1-4시간)
        duration = (self.end_time.hour - self.start_time.hour) + \
                  (self.end_time.minute - self.start_time.minute) / 60
        if not 1 <= duration <= 4:
            raise TimeSlotError("수업 시간은 1시간에서 4시간 사이여야 합니다.")

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
    prerequisites: List[str] = None

    def __post_init__(self):
        """과목 정보의 유효성을 검증"""
        # 타입 검증
        self._validate_types()
        
        # 필수 필드 검증
        self._validate_required_fields()
        
        # 값 범위 검증
        self._validate_ranges()
        
        # 선수과목 초기화 및 검증
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
        if not 1 <= self.credits <= 3:
            raise CourseError("학점은 1~3 사이여야 합니다.")
        if self.capacity <= 0:
            raise CourseError("수용 인원은 0보다 커야 합니다.")
        if not 0 <= self.current_enrolled <= self.capacity:
            raise CourseError("현재 수강 인원은 0 이상이고 수용 인원 이하여야 합니다.")
        if not 0 <= self.difficulty <= 1:
            raise CourseError("난이도는 0부터 1 사이여야 합니다.")
        if not 0 <= self.rating <= 5:
            raise CourseError("평점은 0부터 5 사이여야 합니다.")

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
            # 시간 슬롯 변환
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
        # 학점 범위 검증
        if not 1 <= self.min_credits <= self.max_credits <= 21:
            raise ValueError("학점 범위가 올바르지 않습니다. (최소 1학점, 최대 21학점)")
        
        # 선호 요일 검증
        if not all(0 <= day <= 4 for day in self.preferred_days):
            raise ValueError("선호 요일은 0(월)부터 4(금)까지여야 합니다.")

class ScheduleRecommender:
    """
    시간표 추천 시스템의 핵심 클래스
    
    주요 기능:
    - 과목 정보 관리
    - 사용자 선호도 설정
    - AI 기반 시간표 추천
    - 시간표 시각화
    """
    
    def __init__(self):
        """ScheduleRecommender 초기화"""
        self.courses: List[Course] = []
        self.user_preferences: Optional[UserPreferences] = None
        self.model = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.saved_schedules: Dict[str, List[Course]] = {}
        self.training_data = None
        self.feature_names = None

    def extract_basic_features(self, course: Course) -> Dict[str, Any]:
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

    def extract_time_features(self, course: Course) -> Dict[str, Any]:
        """시간 관련 특성 추출"""
        time_features = {
            'morning_classes': 0,  # 9-12시
            'afternoon_classes': 0,  # 12-17시
            'evening_classes': 0,  # 17시 이후
            'total_days': len(set(slot.day for slot in course.time_slots)),
            'avg_duration': 0
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

    def extract_professor_features(self, course: Course) -> Dict[str, Any]:
        """교수 관련 특성 추출"""
        professor_courses = [c for c in self.courses if c.professor == course.professor]
        return {
            'professor_course_count': len(professor_courses),
            'professor_total_students': sum(c.current_enrolled for c in professor_courses),
            'professor_avg_rating': sum(c.rating for c in professor_courses) / len(professor_courses) if professor_courses else 0
        }

    def extract_preference_features(self, course: Course) -> Dict[str, Any]:
        """학생 선호도 관련 특성 추출"""
        if not self.user_preferences:
            return {}
            
        return {
            'matches_preferred_professor': course.professor in self.user_preferences.preferred_professors,
            'matches_preferred_days': any(
                slot.day in self.user_preferences.preferred_days 
                for slot in course.time_slots
            ),
            'difficulty_match': 1 - abs(course.difficulty - self.user_preferences.preferred_difficulty),
            'rating_match': 1 - abs(course.rating - self.user_preferences.preferred_rating) / 5
        }

    def preprocess_features(self) -> np.ndarray:
        """모든 특성을 전처리하고 정규화"""
        all_features = []
        
        for course in self.courses:
            # 모든 특성 추출 및 결합
            features = {
                **self.extract_basic_features(course),
                **self.extract_time_features(course),
                **self.extract_professor_features(course),
                **self.extract_preference_features(course)
            }
            all_features.append(features)
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_features)
        
        # 특성 이름 저장
        self.feature_names = df.columns.tolist()
        
        # 수치형 특성 정규화
        numeric_features = df.select_dtypes(include=[np.number]).columns
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        
        # 범주형 특성 인코딩
        categorical_features = df.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = OneHotEncoder(sparse=False)
                encoded = self.encoders[feature].fit_transform(df[[feature]])
            else:
                encoded = self.encoders[feature].transform(df[[feature]])
            
            # 인코딩된 특성 추가
            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{feature}_{i}" for i in range(encoded.shape[1])]
            )
            df = pd.concat([df, encoded_df], axis=1)
            df = df.drop(feature, axis=1)
        
        # 특성 순서 정렬하여 일관성 유지
        if hasattr(self, 'feature_names'):
            df = df.reindex(columns=self.feature_names)
        
        return df.values

    def train_model(self, training_data: pd.DataFrame):
        """AI 모델 훈련"""
        self.training_data = training_data
        
        # 특성과 레이블 분리
        X = self.preprocess_features()
        y = training_data['score'].values
        
        # 특성 개수 로깅
        logging.info(f"특성 개수: {X.shape[1]}")
        logging.info(f"특성 이름: {self.feature_names}")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 모델 훈련
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # 모델 평가
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        logging.info(f"모델 훈련 완료 - 훈련 점수: {train_score:.3f}, 테스트 점수: {test_score:.3f}")
        
        # 특성 중요도 분석
        self.analyze_feature_importance()

    def analyze_feature_importance(self):
        """특성 중요도 분석 및 시각화"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 특성 중요도 계산
        importances = self.model.feature_importances_
        
        # 중요도 시각화
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances, y=self.feature_names)
        plt.title("특성 중요도")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

    def predict_schedule_score(self, schedule: List[Course]) -> float:
        """시간표 점수 예측"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 시간표의 특성 추출
        schedule_features = []
        for course in schedule:
            features = {
                **self.extract_basic_features(course),
                **self.extract_time_features(course),
                **self.extract_professor_features(course),
                **self.extract_preference_features(course)
            }
            schedule_features.append(features)
        
        # DataFrame으로 변환
        df = pd.DataFrame(schedule_features)
        
        # 학습된 특성 순서와 동일하게 정렬
        if hasattr(self, 'feature_names'):
            # 누락된 특성은 0으로 채움
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            # 특성 순서 정렬
            df = df.reindex(columns=self.feature_names)
        
        # 수치형 특성 정규화
        numeric_features = df.select_dtypes(include=[np.number]).columns
        df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        # 범주형 특성 인코딩
        categorical_features = df.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            if feature in self.encoders:
                encoded = self.encoders[feature].transform(df[[feature]])
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{feature}_{i}" for i in range(encoded.shape[1])]
                )
                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(feature, axis=1)
        
        # 특성 개수 로깅
        logging.info(f"예측 시 특성 개수: {df.shape[1]}")
        logging.info(f"예측 시 특성 이름: {df.columns.tolist()}")
        
        # 점수 예측
        return float(self.model.predict(df.values).mean())

    def generate_recommendations(self) -> List[Tuple[List[Course], float]]:
        """AI 기반 추천 시간표 생성"""
        if not self.user_preferences:
            raise ValueError("사용자 선호도가 설정되지 않았습니다.")
        
        if self.model is None:
            # 모델이 없는 경우 기존 규칙 기반 방식 사용
            return super().generate_recommendations()
        
        valid_schedules = []
        
        # 선호하는 교수의 과목 먼저 추가
        preferred_courses = [
            course for course in self.courses
            if course.professor in self.user_preferences.preferred_professors
            and course.code not in self.user_preferences.excluded_courses
        ]
        
        # 나머지 과목 추가
        other_courses = [
            course for course in self.courses
            if course.professor not in self.user_preferences.preferred_professors
            and course.code not in self.user_preferences.excluded_courses
        ]
        
        # 모든 과목을 하나의 리스트로 합침
        all_courses = preferred_courses + other_courses
        
        def backtrack(index: int, schedule: List[Course], credits: int):
            """백트래킹으로 가능한 모든 시간표 생성"""
            if credits >= self.user_preferences.min_credits:
                # AI 모델로 점수 예측
                score = self.predict_schedule_score(schedule)
                valid_schedules.append((schedule.copy(), score))
                if credits >= self.user_preferences.max_credits:
                    return
            
            if index >= len(all_courses):
                return
            
            current_course = all_courses[index]
            
            # 시간 충돌 확인
            can_add = True
            for course in schedule:
                if self.check_time_conflict(current_course, course):
                    can_add = False
                    break
            
            # 과목 추가 가능한 경우
            if can_add and credits + current_course.credits <= self.user_preferences.max_credits:
                schedule.append(current_course)
                backtrack(index + 1, schedule, credits + current_course.credits)
                schedule.pop()
            
            # 현재 과목을 건너뛰는 경우
            backtrack(index + 1, schedule, credits)
        
        # 백트래킹으로 가능한 모든 시간표 생성
        backtrack(0, [], 0)
        
        # 점수 기준으로 정렬
        valid_schedules.sort(key=lambda x: x[1], reverse=True)
        
        return valid_schedules[:5]  # 상위 5개 시간표만 반환

    def check_time_conflict(self, course1: Course, course2: Course) -> bool:
        """두 과목의 시간이 충돌하는지 확인"""
        for slot1 in course1.time_slots:
            for slot2 in course2.time_slots:
                if slot1.day == slot2.day:
                    # 시간이 겹치는지 확인
                    # 시작 시간과 종료 시간이 같을 때는 충돌로 판단하지 않음
                    if (slot1.start_time < slot2.end_time and 
                        slot1.end_time > slot2.start_time):
                        return True
        return False

    def load_courses_from_json(self, json_file: str) -> None:
        """JSON 파일에서 과목 데이터를 로드하고 검증"""
        try:
            # 파일 존재 여부 확인
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_file}")
            
            # 파일 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"JSON 파일 형식이 올바르지 않습니다: {str(e)}", e.doc, e.pos)
            
            # 데이터 구조 검증
            if not isinstance(data, list):
                raise ValueError("JSON 데이터는 과목 리스트여야 합니다.")
            
            # 각 과목 데이터 검증 및 변환
            for course_data in data:
                try:
                    # 필수 필드 존재 여부 확인
                    required_fields = ['code', 'name', 'professor', 'credits', 'time_slots', 'classroom', 'capacity']
                    missing_fields = [field for field in required_fields if field not in course_data]
                    if missing_fields:
                        raise CourseError(f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}")
                    
                    # 시간 슬롯 변환 및 검증
                    time_slots = []
                    for slot in course_data['time_slots']:
                        try:
                            # 시간 슬롯 필드 검증
                            if not all(key in slot for key in ['day', 'start_time', 'end_time']):
                                raise TimeSlotError("시간 슬롯에 필수 필드가 누락되었습니다.")
                            
                            # 시간 형식 검증
                            try:
                                start_hour, start_min = map(int, slot['start_time'].split(':'))
                                end_hour, end_min = map(int, slot['end_time'].split(':'))
                            except ValueError:
                                raise TimeSlotError("시간 형식이 올바르지 않습니다 (HH:MM)")
                            
                            time_slots.append(TimeSlot(
                                day=slot['day'],
                                start_time=time(start_hour, start_min),
                                end_time=time(end_hour, end_min)
                            ))
                        except Exception as e:
                            raise TimeSlotError(f"시간 슬롯 변환 실패: {str(e)}")
                    
                    # 선택적 필드 기본값 설정
                    difficulty = course_data.get('difficulty', 0.5)
                    rating = course_data.get('rating', 3.0)
                    prerequisites = course_data.get('prerequisites', [])
                    current_enrolled = course_data.get('current_enrolled', 0)
                    
                    # Course 객체 생성
                    course = Course(
                        code=course_data['code'],
                        name=course_data['name'],
                        professor=course_data['professor'],
                        credits=course_data['credits'],
                        time_slots=time_slots,
                        classroom=course_data['classroom'],
                        capacity=course_data['capacity'],
                        current_enrolled=current_enrolled,
                        difficulty=difficulty,
                        rating=rating,
                        prerequisites=prerequisites
                    )
                    
                    # 과목 추가
                    self.add_course(course)
                    logging.info(f"과목 추가 성공: {course.code} - {course.name}")
                    
                except (CourseError, TimeSlotError) as e:
                    logging.error(f"과목 데이터 오류 ({course_data.get('code', 'unknown')}): {str(e)}")
                    continue
                except Exception as e:
                    logging.error(f"예상치 못한 오류 발생 ({course_data.get('code', 'unknown')}): {str(e)}")
                    continue
                    
        except FileNotFoundError as e:
            logging.error(str(e))
            raise
        except json.JSONDecodeError as e:
            logging.error(str(e))
            raise
        except Exception as e:
            logging.error(f"과목 데이터 로드 중 예상치 못한 오류 발생: {str(e)}")
            raise

    def add_course(self, course: Course) -> None:
        """과목 추가 및 시간 충돌 검사"""
        for existing_course in self.courses:
            if self.check_time_conflict(course, existing_course):
                raise TimeSlotError(f"시간 충돌: {course.name}과 {existing_course.name}")
        self.courses.append(course)

    def save_schedule(self, name: str, schedule: List[Course]) -> None:
        """시간표 저장"""
        self.saved_schedules[name] = schedule
        try:
            with open(f"schedules/{name}.json", 'w', encoding='utf-8') as f:
                json.dump([course.to_dict() for course in schedule], f, ensure_ascii=False, indent=2)
            logging.info(f"시간표가 저장되었습니다: {name}")
        except Exception as e:
            logging.error(f"시간표 저장 실패: {str(e)}")
            raise

    def load_schedule(self, name: str) -> List[Course]:
        """저장된 시간표 불러오기"""
        try:
            with open(f"schedules/{name}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [Course.from_dict(course_data) for course_data in data]
        except Exception as e:
            logging.error(f"시간표 불러오기 실패: {str(e)}")
            raise

    def calculate_total_credits(self, schedule: List[Course]) -> int:
        """총 학점 계산"""
        return sum(course.credits for course in schedule)

    def get_user_preferences(self) -> UserPreferences:
        """사용자로부터 선호도 입력 받기"""
        print("\n=== 시간표 선호도 설정 ===")
        
        # 학점 범위
        while True:
            try:
                min_credits = int(input("최소 학점 (1-21, 기본값: 1): ").strip() or "1")
                max_credits = int(input("최대 학점 (1-21, 기본값: 21): ").strip() or "21")
                if 1 <= min_credits <= max_credits <= 21:
                    break
                print("올바른 학점 범위를 입력하세요.")
            except ValueError:
                print("숫자를 입력하세요.")

        # 선호 요일
        print("\n선호하는 요일을 선택하세요 (0: 월, 1: 화, 2: 수, 3: 목, 4: 금)")
        print("선호하는 요일이 없다면 '없음'을 입력하세요.")
        preferred_days = []
        while True:
            try:
                day_input = input("선호하는 요일 번호를 입력하세요 (0-4, 없음: n): ").strip().lower()
                if day_input == 'n':
                    break
                day = int(day_input)
                if 0 <= day <= 4:
                    if day not in preferred_days:
                        preferred_days.append(day)
                        print(f"선택된 요일: {['월', '화', '수', '목', '금'][day]}")
                    else:
                        print("이미 선택된 요일입니다.")
                else:
                    print("0부터 4 사이의 숫자를 입력하세요.")
            except ValueError:
                print("올바른 숫자를 입력하세요.")
            
            if input("다른 요일도 선택하시겠습니까? (y/n): ").strip().lower() != 'y':
                break

        # 선호 교수
        print("\n선호하는 교수를 입력하세요 (쉼표로 구분, 없음: n)")
        while True:
            prof_input = input("교수명: ").strip()
            if prof_input.lower() == 'n':
                preferred_professors = []
                break
            preferred_professors = [p.strip() for p in prof_input.split(',') if p.strip()]
            if preferred_professors:
                print(f"선택된 교수: {', '.join(preferred_professors)}")
                break
            print("올바른 교수명을 입력하세요.")

        # 제외 과목
        print("\n제외하고 싶은 과목 코드를 입력하세요 (쉼표로 구분, 없음: n)")
        while True:
            course_input = input("과목 코드: ").strip()
            if course_input.lower() == 'n':
                excluded_courses = []
                break
            excluded_courses = [c.strip() for c in course_input.split(',') if c.strip()]
            if excluded_courses:
                print(f"제외할 과목: {', '.join(excluded_courses)}")
                break
            print("올바른 과목 코드를 입력하세요.")

        # 난이도 선호
        while True:
            try:
                diff_input = input("선호하는 난이도 (0-1, 기본값: 0.5): ").strip() or "0.5"
                preferred_difficulty = float(diff_input)
                if 0 <= preferred_difficulty <= 1:
                    break
                print("0부터 1 사이의 숫자를 입력하세요.")
            except ValueError:
                print("올바른 숫자를 입력하세요.")

        # 평점 선호
        while True:
            try:
                rating_input = input("선호하는 평점 (0-5, 기본값: 3.0): ").strip() or "3.0"
                preferred_rating = float(rating_input)
                if 0 <= preferred_rating <= 5:
                    break
                print("0부터 5 사이의 숫자를 입력하세요.")
            except ValueError:
                print("올바른 숫자를 입력하세요.")

        print("\n=== 입력된 선호도 정보 ===")
        print(f"학점 범위: {min_credits}~{max_credits}")
        print(f"선호 요일: {[['월', '화', '수', '목', '금'][d] for d in preferred_days] if preferred_days else '없음'}")
        print(f"선호 교수: {', '.join(preferred_professors) if preferred_professors else '없음'}")
        print(f"제외 과목: {', '.join(excluded_courses) if excluded_courses else '없음'}")
        print(f"선호 난이도: {preferred_difficulty}")
        print(f"선호 평점: {preferred_rating}")

        return UserPreferences(
            min_credits=min_credits,
            max_credits=max_credits,
            preferred_days=preferred_days,
            preferred_professors=preferred_professors,
            excluded_courses=excluded_courses,
            preferred_difficulty=preferred_difficulty,
            preferred_rating=preferred_rating
        )

    def set_user_preferences(self, preferences: UserPreferences) -> None:
        """사용자 선호도 설정"""
        self.user_preferences = preferences

    def generate_html(self, schedule: List[Course], score: float = None) -> str:
        """HTML 생성 (점수 표시 추가)"""
        # 시간표 정보 패널 개선
        # 색상 구분 개선
        # 모바일 대응 추가
        days = ["월", "화", "수", "목", "금"]
        hours = list(range(9, 18))
        
        subject_colors = {
            "인공지능프로그래밍": "#e76f51",  # 산호색
            "인공지능개론": "#90be6d",      # 연한 녹색
            "웹프로그래밍": "#e9c46a",      # 골드
            "소프트웨어공학": "#577590",     # 네이비
            "제어시스템보안": "#f94144",     # 빨간색
            "시스템보안": "#43aa8b",        # 민트
            "현대암호학": "#f3722c"         # 주황색
        }

        timetable = {hour: {day: [] for day in range(5)} for hour in hours}

        for course in schedule:
            for slot in course.time_slots:
                start_hour = slot.start_time.hour
                duration = (slot.end_time.hour - slot.start_time.hour) + \
                          (slot.end_time.minute - slot.start_time.minute) / 60
                
                for offset in range(int(duration)):
                    timetable[start_hour + offset][slot.day].append({
                        "subject": course.name,
                        "location": course.classroom,
                        "is_first": offset == 0,
                        "rowspan": int(duration) if offset == 0 else 0,
                        "professor": course.professor,
                        "credits": course.credits
                    })

        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>AI 추천 시간표</title>
    <style>
        body {{
            background-color: #121212;
            color: white;
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .timetable {{
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }}
        .timetable th, .timetable td {{
            border: 1px solid #333;
            text-align: center;
            height: 80px;
            position: relative;
        }}
        .timetable th {{
            background-color: #1e1e1e;
            font-weight: bold;
        }}
        .class-block {{
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            margin: 4px;
            padding: 5px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            display: flex;
            flex-direction: column;
            justify-content: center;
            color: white;
        }}
        .info-panel {{
            background-color: #1e1e1e;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
        }}
        .info-panel h2 {{
            margin-top: 0;
        }}
        .info-panel p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="info-panel">
        <h2>시간표 정보</h2>
        <p>총 학점: {self.calculate_total_credits(schedule)}</p>
        <p>총 과목 수: {len(schedule)}</p>
        {f'<p>선호도 점수: {score:.2f}</p>' if score is not None else ''}
    </div>
    <table class="timetable">
        <thead>
            <tr>
                <th>시간/요일</th>
"""

        for day in days:
            html_content += f"                <th>{day}</th>\n"

        html_content += "            </tr>\n        </thead>\n        <tbody>\n"

        for hour in hours:
            html_content += f"            <tr>\n                <td>{hour:02d}:00</td>\n"
            for day in range(5):
                cells = timetable[hour][day]
                if not cells:
                    html_content += "                <td></td>\n"
                else:
                    cell = cells[0]
                    if cell["is_first"]:
                        subject = cell["subject"]
                        color = subject_colors.get(subject, "#4caf50")
                        rowspan = cell["rowspan"]
                        location = cell["location"]
                        professor = cell["professor"]
                        credits = cell["credits"]
                        html_content += f"""                <td rowspan="{rowspan}">
                    <div class="class-block" style="background-color:{color};">
                        {subject}<br>
                        {professor}<br>
                        {location}<br>
                        {credits}학점
                    </div>
                </td>\n"""
            html_content += "            </tr>\n"

        html_content += "        </tbody>\n    </table>\n</body>\n</html>"
        return html_content

def main():
    """메인 함수"""
    try:
        # schedules 디렉토리 생성
        Path("schedules").mkdir(exist_ok=True)
        
        # 추천 시스템 초기화
        recommender = ScheduleRecommender()
        
        # JSON 파일에서 과목 데이터 로드
        recommender.load_courses_from_json("C:\\Users\\ASUS\\.cursor\\timetable.json")
        
        # 학습 데이터 로드 (CSV 파일)
        if os.path.exists("training_data.csv"):
            training_data = pd.read_csv("training_data.csv")
            recommender.train_model(training_data)
            print("✅ AI 모델이 훈련되었습니다.")
        
        # 사용자 선호도 설정
        preferences = recommender.get_user_preferences()
        recommender.set_user_preferences(preferences)
        
        # 추천 시간표 생성
        recommendations = recommender.generate_recommendations()
        
        if recommendations:
            # 최고 점수의 시간표 선택
            best_schedule, score = recommendations[0]
            
            # HTML 생성
            html_content = recommender.generate_html(best_schedule, score)
            
            # 파일 저장
            with open("recommended_timetable.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"\n✅ 추천 시간표가 생성되었습니다. (선호도 점수: {score:.2f})")
            
            # 시간표 저장 여부 확인
            if input("\n이 시간표를 저장하시겠습니까? (y/n): ").lower() == 'y':
                name = input("저장할 이름을 입력하세요: ")
                recommender.save_schedule(name, best_schedule)
            
            # 웹 브라우저에서 열기
            file_path = os.path.abspath("recommended_timetable.html")
            webbrowser.open(f"file://{file_path}")
        else:
            print("❌ 조건을 만족하는 시간표를 찾을 수 없습니다.")
    
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 