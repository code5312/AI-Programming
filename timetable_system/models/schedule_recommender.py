import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import time
from typing import List, Dict, Tuple, Any, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from .course import Course, CourseError
from .user_preferences import UserPreferences
from .time_slot import TimeSlot, TimeSlotError
from timetable_system.utils.html_generator import generate_html

class ScheduleRecommenderError(Exception):
    """시간표 추천 관련 에러"""
    pass

class CourseType:
    """과목 유형 정의"""
    MAJOR_REQUIRED = "전공필수"
    MAJOR_ELECTIVE = "전공선택"

class ScheduleRecommender:
    def __init__(self, courses: List[Course], preferences: UserPreferences):
        self.courses = courses
        self.preferences = preferences
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.saved_schedules = {}  # 저장된 시간표를 관리하는 딕셔너리 추가
        
        # 과목 유형별 가중치 설정
        self.course_type_weights = {
            CourseType.MAJOR_REQUIRED: 1.5,  # 전공필수 과목 가중치
            CourseType.MAJOR_ELECTIVE: 1.2,  # 전공선택 과목 가중치
        }
        
        # 학년별 가중치 설정
        self.year_weights = {
            1: 2.0,  # 1학년 과목
            2: 2.0,  # 2학년 과목
            3: 2.0,  # 3학년 과목
            4: 2.0   # 4학년 과목
        }
        
        self._validate_inputs()

    def _validate_inputs(self):
        """입력 데이터 검증"""
        if not self.courses:
            raise ScheduleRecommenderError("과목 목록이 비어있습니다.")
        if not isinstance(self.preferences, UserPreferences):
            raise ScheduleRecommenderError("유효하지 않은 사용자 선호도입니다.")

    def _check_time_conflicts(self, schedule: List[Course]) -> bool:
        """시간표의 시간 충돌 여부 확인"""
        for i, course1 in enumerate(schedule):
            for course2 in schedule[i+1:]:
                for slot1 in course1.time_slots:
                    for slot2 in course2.time_slots:
                        if slot1.overlaps_with(slot2):
                            return True
        return False

    def _extract_features(self, schedule: List[Course]) -> Dict[str, Any]:
        """시간표에서 특징 추출 (개선된 버전)"""
        features = {
            'total_credits': sum(course.credits for course in schedule),
            'avg_difficulty': np.mean([course.difficulty for course in schedule]),
            'avg_rating': np.mean([course.rating for course in schedule]),
            'num_courses': len(schedule),
            'has_time_conflicts': int(self._check_time_conflicts(schedule)),
            'preferred_days_ratio': sum(1 for course in schedule 
                                     for slot in course.time_slots 
                                     if slot.day in self.preferences.preferred_days) / 
                                 sum(len(course.time_slots) for course in schedule),
            'preferred_time_ratio': sum(1 for course in schedule 
                                     for slot in course.time_slots 
                                     if (self.preferences.preferred_start_time <= slot.start_time and 
                                         slot.end_time <= self.preferences.preferred_end_time)) /
                                 sum(len(course.time_slots) for course in schedule),
            'must_take_ratio': sum(1 for course in schedule 
                                if course.code in self.preferences.must_take_courses) / len(schedule),
            'avoid_ratio': sum(1 for course in schedule 
                            if course.code in self.preferences.avoid_courses) / len(schedule),
            # 과목 유형별 비율 추가
            'major_required_ratio': sum(1 for course in schedule 
                                     if course.course_type == CourseType.MAJOR_REQUIRED) / len(schedule),
            'major_elective_ratio': sum(1 for course in schedule 
                                     if course.course_type == CourseType.MAJOR_ELECTIVE) / len(schedule),
            
            # 학년별 비율 추가
            'year1_ratio': sum(1 for course in schedule if course.year == 1) / len(schedule),
            'year2_ratio': sum(1 for course in schedule if course.year == 2) / len(schedule),
            'year3_ratio': sum(1 for course in schedule if course.year == 3) / len(schedule),
            'year4_ratio': sum(1 for course in schedule if course.year == 4) / len(schedule)
        }
        return features

    def train_model(self, training_data: pd.DataFrame):
        """지도학습 모델 훈련 (개선된 버전)"""
        try:
            # 특징과 타겟 분리
            X = training_data.drop('score', axis=1)
            y = training_data['score']

            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 범주형 변수 인코딩
            for column in X.select_dtypes(include=['object']).columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                # 훈련 데이터의 모든 고유한 값들을 저장
                unique_values = X_train[column].unique()
                self.label_encoders[column].fit(unique_values)
                
                # 훈련 데이터 변환
                X_train[column] = self.label_encoders[column].transform(X_train[column])
                
                # 테스트 데이터 변환 (새로운 레이블은 -1로 처리)
                try:
                    X_test[column] = self.label_encoders[column].transform(X_test[column])
                except ValueError:
                    # 새로운 레이블이 있는 경우 -1로 처리
                    X_test[column] = X_test[column].apply(
                        lambda x: -1 if x not in self.label_encoders[column].classes_ 
                        else self.label_encoders[column].transform([x])[0]
                    )

            # 수치형 변수 정규화
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X_train[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
            X_test[numeric_columns] = self.scaler.transform(X_test[numeric_columns])

            # 모델 훈련 (GradientBoostingRegressor 사용)
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            # 모델 평가
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"모델 훈련 완료 - MSE: {mse:.4f}, R2: {r2:.4f}")
            
            # 특성 중요도 시각화
            self._plot_feature_importance(X.columns)
            
            # 모델 저장
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, 'models/schedule_recommender.joblib')
            joblib.dump(self.label_encoders, 'models/label_encoders.joblib')
            joblib.dump(self.scaler, 'models/scaler.joblib')
            
        except Exception as e:
            raise ScheduleRecommenderError(f"모델 훈련 중 오류 발생: {str(e)}")

    def _plot_feature_importance(self, feature_names):
        """특성 중요도 시각화"""
        try:
            importance = self.model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance')
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig('models/feature_importance.png')
            plt.close()
        except Exception as e:
            logging.warning(f"특성 중요도 시각화 중 오류 발생: {str(e)}")

    def _calculate_schedule_score(self, schedule: List[Course]) -> float:
        """시간표의 점수 계산 (개선된 버전)"""
        if not schedule:
            return 0.0

        # 기본 규칙 기반 점수 계산
        total_score = 0.0
        total_credits = sum(course.credits for course in schedule)

        # 학점 범위 검사
        if not (self.preferences.min_credits <= total_credits <= self.preferences.max_credits):
            return 0.0

        # AI 모델이 있는 경우 모델 기반 점수 계산
        if self.model is not None:
            try:
                features = self._extract_features(schedule)
                feature_df = pd.DataFrame([features])
                
                # 범주형 변수 인코딩
                for column in feature_df.select_dtypes(include=['object']).columns:
                    if column in self.label_encoders:
                        try:
                            feature_df[column] = self.label_encoders[column].transform(feature_df[column])
                        except ValueError:
                            # 새로운 레이블이 있는 경우 -1로 처리
                            feature_df[column] = feature_df[column].apply(
                                lambda x: -1 if x not in self.label_encoders[column].classes_ 
                                else self.label_encoders[column].transform([x])[0]
                            )
                
                # 수치형 변수 정규화
                numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
                feature_df[numeric_columns] = self.scaler.transform(feature_df[numeric_columns])
                
                # 모델 예측
                model_score = self.model.predict(feature_df)[0]
                
                # 과목 유형별 가중치 적용
                type_weight = 1.0
                for course in schedule:
                    type_weight *= self.course_type_weights.get(course.course_type, 1.0)
                
                # 학년별 가중치 적용
                year_weight = 1.0
                for course in schedule:
                    year_weight *= self.year_weights.get(course.year, 1.0)
                
                # 최종 점수 계산
                final_score = model_score * type_weight * year_weight
                return final_score
                
            except Exception as e:
                logging.warning(f"모델 예측 중 오류 발생, 규칙 기반 점수 사용: {str(e)}")

        # 규칙 기반 점수 계산 (AI 모델이 없거나 오류 발생 시)
        for course in schedule:
            # 기본 점수
            course_score = 1.0
            
            # 과목 유형별 가중치
            course_score *= self.course_type_weights.get(course.course_type, 1.0)
            
            # 학년별 가중치
            course_score *= self.year_weights.get(course.year, 1.0)
            
            # 필수 과목 검사
            if course.code in self.preferences.must_take_courses:
                course_score *= 2.0
            
            # 피하고 싶은 과목 검사
            if course.code in self.preferences.avoid_courses:
                course_score *= 0.5
            
            # 교수 선호도
            if course.professor in self.preferences.preferred_professors:
                course_score *= 1.2
            
            # 강의실 선호도
            if course.classroom in self.preferences.preferred_classrooms:
                course_score *= 1.1
            
            total_score += course_score

        return total_score / len(schedule) if schedule else 0.0

    def load_model(self):
        """저장된 모델 로드"""
        try:
            model_path = 'models/schedule_recommender.joblib'
            encoders_path = 'models/label_encoders.joblib'
            
            if os.path.exists(model_path) and os.path.exists(encoders_path):
                self.model = joblib.load(model_path)
                self.label_encoders = joblib.load(encoders_path)
                return True
            return False
        except Exception as e:
            raise ScheduleRecommenderError(f"모델 로드 중 오류 발생: {str(e)}")

    def _generate_schedules(self, 
                          current_schedule: List[Course],
                          remaining_courses: List[Course],
                          max_schedules: int = 10) -> List[List[Course]]:
        """가능한 모든 시간표 생성"""
        if not remaining_courses or len(current_schedule) >= 8:  # 최대 8개 과목
            if not self._check_time_conflicts(current_schedule):
                return [current_schedule]
            return []

        schedules = []
        for i, course in enumerate(remaining_courses):
            new_schedule = current_schedule + [course]
            if not self._check_time_conflicts(new_schedule):
                new_remaining = remaining_courses[:i] + remaining_courses[i+1:]
                schedules.extend(self._generate_schedules(new_schedule, new_remaining, max_schedules))
                if len(schedules) >= max_schedules:
                    break

        return schedules

    def recommend_schedules(self, max_schedules: int = 10) -> List[Dict[str, Any]]:
        """시간표 추천"""
        try:
            # AI 모델 로드 시도
            if self.model is None:
                self.load_model()
            
            # 가능한 모든 시간표 생성
            all_schedules = self._generate_schedules([], self.courses, max_schedules)
            
            # 각 시간표의 점수 계산
            scored_schedules = [(schedule, self._calculate_schedule_score(schedule)) 
                              for schedule in all_schedules]
            
            # 점수 기준으로 정렬
            scored_schedules.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 시간표 반환
            return [{
                "schedule": [course.to_dict() for course in schedule],
                "score": score,
                "total_credits": sum(course.credits for course in schedule)
            } for schedule, score in scored_schedules[:max_schedules]]
            
        except Exception as e:
            raise ScheduleRecommenderError(f"시간표 추천 중 오류 발생: {str(e)}")

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
            'avg_duration': 0,
            'time_gaps': []
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
        if not self.preferences:
            return {}
            
        return {
            'matches_preferred_professor': course.professor in self.preferences.preferred_professors,
            'matches_preferred_days': any(
                slot.day in self.preferences.preferred_days 
                for slot in course.time_slots
            ),
            'matches_preferred_times': any(
                any(
                    slot.start_time >= pref_time.start_time and 
                    slot.end_time <= pref_time.end_time
                    for pref_time in self.preferences.preferred_times
                )
                for slot in course.time_slots
            ),
            'difficulty_match': 1 - abs(course.difficulty - self.preferences.preferred_difficulty),
            'rating_match': 1 - abs(course.rating - self.preferences.preferred_rating) / 5
        }

    def preprocess_features(self) -> np.ndarray:
        """모든 특성을 전처리하고 정규화"""
        all_features = []
        
        for course in self.courses:
            # 기본 특성
            basic_features = self.extract_basic_features(course)
            
            # 시간 특성
            time_features = self.extract_time_features(course)
            
            # 교수 특성
            professor_features = self.extract_professor_features(course)
            
            # 선호도 특성
            preference_features = self.extract_preference_features(course)
            
            # 모든 특성 결합
            combined_features = {
                **basic_features,
                **time_features,
                **professor_features,
                **preference_features
            }
            
            all_features.append(combined_features)
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_features)
        self.feature_names = df.columns.tolist()
        
        # 수치형 특성 정규화
        numeric_features = df.select_dtypes(include=[np.number]).columns
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        
        # 범주형 특성 인코딩
        categorical_features = df.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
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
        
        return df.values

    def preprocess_features_for_schedule(self, schedule: List[Course]) -> np.ndarray:
        """입력된 schedule(과목 리스트)에 대해 특성 추출 및 전처리"""
        all_features = []
        for course in schedule:
            basic_features = self.extract_basic_features(course)
            time_features = self.extract_time_features(course)
            professor_features = self.extract_professor_features(course)
            preference_features = self.extract_preference_features(course)
            combined_features = {
                **basic_features,
                **time_features,
                **professor_features,
                **preference_features
            }
            all_features.append(combined_features)
        df = pd.DataFrame(all_features)
        # 수치형 특성 정규화 (스케일러는 전체 학습 데이터 기준으로 fit되어 있어야 함)
        numeric_features = df.select_dtypes(include=[np.number]).columns
        if hasattr(self.scaler, 'mean_'):
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        # 범주형 특성 인코딩 (인코더는 전체 학습 데이터 기준으로 fit되어 있어야 함)
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
        return df.values

    def predict_schedule_score(self, schedule: List[Course]) -> float:
        """시간표 점수 예측 (schedule 단위 전처리 사용)"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        X = self.preprocess_features_for_schedule(schedule)
        return float(self.model.predict(X).mean())

    def generate_recommendations(self, top_n: int = 3) -> List[Tuple[List[Course], float]]:
        """AI 기반 추천 시간표 생성 (중복 캐시, top_n 반환)"""
        if not self.preferences:
            raise ValueError("사용자 선호도가 설정되지 않았습니다.")
        if self.model is None:
            raise RuntimeError("AI 모델이 훈련되지 않았습니다. training_data.csv 파일이 필요합니다.")
        
        valid_schedules = []
        seen = set()
        
        # 선호도 기반 과목 필터링
        preferred_courses = [
            course for course in self.courses
            if course.professor in self.preferences.preferred_professors
            and course.code not in self.preferences.excluded_courses
        ]
        other_courses = [
            course for course in self.courses
            if course.professor not in self.preferences.preferred_professors
            and course.code not in self.preferences.excluded_courses
        ]
        all_courses = preferred_courses + other_courses
        
        def backtrack(index: int, schedule: List[Course], credits: int):
            # 중복 체크
            key = frozenset(course.code for course in schedule)
            if key in seen:
                return
            seen.add(key)
            
            # 최소 학점 조건 체크
            if credits >= self.preferences.min_credits:
                try:
                    score = self.predict_schedule_score(schedule)
                    valid_schedules.append((schedule.copy(), score))
                except Exception as e:
                    logging.warning(f"시간표 점수 계산 중 오류 발생: {str(e)}")
                    return
                
                # 최대 학점 조건 체크
                if credits >= self.preferences.max_credits:
                    return
            
            # 모든 과목을 검사했으면 종료
            if index >= len(all_courses):
                return
            
            current_course = all_courses[index]
            
            # 시간 충돌 체크
            can_add = True
            for course in schedule:
                if self.check_time_conflict(current_course, course):
                    can_add = False
                    break
            
            # 과목 추가 가능 여부 체크
            if can_add and credits + current_course.credits <= self.preferences.max_credits:
                schedule.append(current_course)
                backtrack(index + 1, schedule, credits + current_course.credits)
                schedule.pop()
            
            # 현재 과목을 추가하지 않고 다음 과목 검사
            backtrack(index + 1, schedule, credits)
        
        # 백트래킹 시작
        backtrack(0, [], 0)
        
        # 점수 기준으로 정렬하고 상위 N개 반환
        valid_schedules.sort(key=lambda x: x[1], reverse=True)
        return valid_schedules[:top_n]

    def check_time_conflict(self, course1: Course, course2: Course) -> bool:
        """두 과목의 시간이 충돌하는지 확인 (끝나는 시간과 시작 시간이 같으면 허용)"""
        for slot1 in course1.time_slots:
            for slot2 in course2.time_slots:
                if slot1.day == slot2.day:
                    # slot1: [start1, end1), slot2: [start2, end2)
                    # 겹치는 경우: start1 < end2 and end1 > start2
                    # 단, end1 == start2 또는 end2 == start1은 허용
                    if (slot1.start_time < slot2.end_time and slot1.end_time > slot2.start_time):
                        if slot1.end_time == slot2.start_time or slot2.end_time == slot1.start_time:
                            continue  # 끝나는 시간과 시작 시간이 같으면 허용
                        return True
        return False

    def load_courses_from_json(self, json_file: str) -> None:
        """JSON 파일에서 과목 데이터를 로드하고 검증"""
        try:
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"JSON 파일 형식이 올바르지 않습니다: {str(e)}", e.doc, e.pos)
            
            if not isinstance(data, list):
                raise ValueError("JSON 데이터는 과목 리스트여야 합니다.")
            
            # 기존 과목 목록 초기화
            self.courses = []
            
            for course_data in data:
                try:
                    # 필수 필드 검사
                    required_fields = ['code', 'name', 'professor', 'credits', 'time_slots', 'classroom', 'capacity']
                    missing_fields = [field for field in required_fields if field not in course_data]
                    if missing_fields:
                        raise CourseError(f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}")
                    
                    # 시간 슬롯 변환
                    time_slots = []
                    for slot in course_data['time_slots']:
                        try:
                            if not all(key in slot for key in ['day', 'start_time', 'end_time']):
                                raise TimeSlotError("시간 슬롯에 필수 필드가 누락되었습니다.")
                            
                            # 시간 형식 변환
                            try:
                                start_hour, start_min = map(int, slot['start_time'].split(':'))
                                end_hour, end_min = map(int, slot['end_time'].split(':'))
                            except ValueError:
                                raise TimeSlotError(f"시간 형식이 올바르지 않습니다 (HH:MM): {slot}")
                            
                            # TimeSlot 객체 생성
                            ts = TimeSlot(
                                day=slot['day'],
                                start_time=time(start_hour, start_min),
                                end_time=time(end_hour, end_min)
                            )
                            time_slots.append(ts)
                            
                        except Exception as e:
                            raise TimeSlotError(f"시간 슬롯 변환 실패: {slot} - {str(e)}")
                    
                    # Course 객체 생성
                    course = Course(
                        code=course_data['code'],
                        name=course_data['name'],
                        professor=course_data['professor'],
                        credits=course_data['credits'],
                        time_slots=time_slots,
                        classroom=course_data['classroom'],
                        capacity=course_data['capacity'],
                        current_enrolled=course_data.get('current_enrolled', 0),
                        difficulty=course_data.get('difficulty', 0.5),
                        rating=course_data.get('rating', 3.0),
                        prerequisites=course_data.get('prerequisites', [])
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
                min_credits = int(input("최소 학점 (1-21): "))
                max_credits = int(input("최대 학점 (1-21): "))
                if 1 <= min_credits <= max_credits <= 21:
                    break
                print("올바른 학점 범위를 입력하세요.")
            except ValueError:
                print("숫자를 입력하세요.")

        # 선호 요일
        print("\n선호하는 요일을 선택하세요 (0: 월, 1: 화, 2: 수, 3: 목, 4: 금)")
        preferred_days = []
        for day in range(5):
            if input(f"{['월', '화', '수', '목', '금'][day]}요일 선호? (y/n): ").lower() == 'y':
                preferred_days.append(day)

        # 선호 시간대
        preferred_times = []
        print("\n선호하는 시간대를 입력하세요 (24시간 형식)")
        while True:
            try:
                start = input("시작 시간 (HH:MM): ")
                end = input("종료 시간 (HH:MM): ")
                start_hour, start_min = map(int, start.split(':'))
                end_hour, end_min = map(int, end.split(':'))
                preferred_times.append(TimeSlot(0, time(start_hour, start_min), time(end_hour, end_min)))
                if input("더 입력하시겠습니까? (y/n): ").lower() != 'y':
                    break
            except ValueError:
                print("올바른 시간 형식을 입력하세요 (HH:MM)")

        # 선호 교수
        print("\n선호하는 교수를 입력하세요 (쉼표로 구분)")
        preferred_professors = [p.strip() for p in input("교수명: ").split(',') if p.strip()]

        # 제외 과목
        print("\n제외하고 싶은 과목 코드를 입력하세요 (쉼표로 구분)")
        excluded_courses = [c.strip() for c in input("과목 코드: ").split(',') if c.strip()]

        return UserPreferences(
            min_credits=min_credits,
            max_credits=max_credits,
            preferred_days=preferred_days,
            preferred_times=preferred_times,
            preferred_professors=preferred_professors,
            excluded_courses=excluded_courses
        )

    def set_user_preferences(self, preferences: UserPreferences) -> None:
        """사용자 선호도 설정"""
        self.preferences = preferences 