"""
대화형 CLI: 사용자 선호도 입력 및 실행 진입점
"""
import logging
import webbrowser

from . import PROJECT_ROOT
from .html_report import generate_html
from .models import UserPreferences
from .recommender import ScheduleRecommender


def get_user_preferences() -> UserPreferences:
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


def main():
    """메인 함수"""
    try:
        recommender = ScheduleRecommender()

        # 과목 데이터 + 학습 데이터를 courses.csv 한 파일에서 로드.
        # 실행 시점의 현재 작업 디렉터리가 아니라 timetable.py가 있는 프로젝트
        # 루트를 기준으로 찾으므로, 어디서 실행하든(IDE 실행 버튼, 다른 폴더에서
        # 커맨드 실행 등) 같은 courses.csv를 찾는다. 이 파일을 다른 학기/커리큘럼
        # 데이터로 통째로 교체하면 그대로 반영됨.
        training_data = recommender.load_courses_from_csv(str(PROJECT_ROOT / "courses.csv"))

        if training_data is not None and not training_data.empty:
            recommender.train_model(training_data)
            print("✅ AI 모델이 훈련되었습니다.")
        else:
            print("ℹ️ courses.csv에 score 값이 없어 AI 모델 없이 규칙 기반으로 진행합니다.")

        preferences = get_user_preferences()
        recommender.set_user_preferences(preferences)

        recommendations = recommender.generate_recommendations()

        if recommendations:
            best_schedule, score = recommendations[0]

            html_content = generate_html(best_schedule, score)

            output_path = PROJECT_ROOT / "recommended_timetable.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"\n✅ 추천 시간표가 생성되었습니다. (선호도 점수: {score:.2f})")

            if input("\n이 시간표를 저장하시겠습니까? (y/n): ").lower() == 'y':
                name = input("저장할 이름을 입력하세요: ")
                recommender.save_schedule(name, best_schedule)

            webbrowser.open(f"file://{output_path}")
        else:
            print("❌ 조건을 만족하는 시간표를 찾을 수 없습니다.")

    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")
