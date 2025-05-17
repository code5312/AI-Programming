import os
import json
import logging
from datetime import time
import pandas as pd
from timetable_system.models.user_preferences import PreferredTimeRange

from timetable_system.models import (
    Course, UserPreferences, ScheduleRecommender
)
from timetable_system.utils.html_generator import generate_html_timetable

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        filename='timetable.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_courses(json_file: str) -> list:
    """JSON 파일에서 과목 데이터 로드"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [Course.from_dict(course_data) for course_data in data]
    except Exception as e:
        logging.error(f"과목 데이터 로드 실패: {str(e)}")
        raise

def main():
    """메인 함수"""
    setup_logging()
    
    try:
        # 과목 데이터 로드
        courses = load_courses('timetable.json')
        logging.info(f"과목 데이터 로드 완료: {len(courses)}개 과목")
        
        # 사용자 선호도 설정
        preferences = UserPreferences(
            min_credits=15,
            max_credits=21,
            preferred_days=[0, 1, 2, 3, 4],  # 월-금
            preferred_times=[
                PreferredTimeRange(0, time(9, 0), time(18, 0))
            ],
            preferred_professors=[],
            excluded_courses=[],
            preferred_difficulty=0.5,
            preferred_rating=3.0
        )
        
        # 시간표 추천 시스템 초기화
        recommender = ScheduleRecommender(courses, preferences)
        
        # AI 모델 훈련 (training_data.csv가 있는 경우)
        if os.path.exists('training_data.csv'):
            training_data = pd.read_csv('training_data.csv')
            recommender.train_model(training_data)
            logging.info("AI 모델 훈련 완료")
        else:
            logging.warning("training_data.csv 파일이 없어 AI 모델을 훈련하지 않습니다.")
        
        # 시간표 추천
        recommended_schedules = recommender.recommend_schedules(max_schedules=3)
        
        # 추천된 시간표를 HTML로 생성
        for i, schedule_data in enumerate(recommended_schedules):
            schedule = schedule_data['schedule']
            score = schedule_data['score']
            output_file = f'schedules/timetable_{i+1}.html'
            generate_html_timetable(schedule, output_file)
            logging.info(f"시간표 {i+1} 생성 완료 (점수: {score:.2f})")
        
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 