"""
대화형 CLI: 사용자 선호도 입력 및 실행 진입점
"""
import logging
import webbrowser

from . import PROJECT_ROOT
from .html_report import generate_html
from .models import (
    DAY_MAX,
    DAY_MIN,
    DAY_NAMES,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    RATING_MAX,
    RATING_MIN,
    TOTAL_CREDITS_MAX,
    TOTAL_CREDITS_MIN,
    UserPreferences,
)
from .recommender import build_recommender_from_csv


def get_user_preferences() -> UserPreferences:
    """
    사용자로부터 선호도 입력 받기.

    입력 범위는 전부 models.py의 도메인 상수(TOTAL_CREDITS_*, DAY_*, DIFFICULTY_*,
    RATING_*)를 그대로 참조하므로, 다른 커리큘럼에 맞춰 그 상수들을 바꾸면 여기
    프롬프트 문구와 검증 범위도 자동으로 같이 바뀐다 (따로 손볼 필요 없음).
    """
    print("\n=== 시간표 선호도 설정 ===")

    # 학점 범위
    while True:
        try:
            min_credits = int(
                input(f"최소 학점 ({TOTAL_CREDITS_MIN}-{TOTAL_CREDITS_MAX}, 기본값: {TOTAL_CREDITS_MIN}): ").strip()
                or str(TOTAL_CREDITS_MIN)
            )
            max_credits = int(
                input(f"최대 학점 ({TOTAL_CREDITS_MIN}-{TOTAL_CREDITS_MAX}, 기본값: {TOTAL_CREDITS_MAX}): ").strip()
                or str(TOTAL_CREDITS_MAX)
            )
            if TOTAL_CREDITS_MIN <= min_credits <= max_credits <= TOTAL_CREDITS_MAX:
                break
            print("올바른 학점 범위를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")

    # 선호 요일
    day_options = ", ".join(f"{i}: {name}" for i, name in enumerate(DAY_NAMES))
    print(f"\n선호하는 요일을 선택하세요 ({day_options})")
    print("선호하는 요일이 없다면 '없음'을 입력하세요.")
    preferred_days = []
    while True:
        try:
            day_input = input(f"선호하는 요일 번호를 입력하세요 ({DAY_MIN}-{DAY_MAX}, 없음: n): ").strip().lower()
            if day_input == 'n':
                break
            day = int(day_input)
            if DAY_MIN <= day <= DAY_MAX:
                if day not in preferred_days:
                    preferred_days.append(day)
                    print(f"선택된 요일: {DAY_NAMES[day]}")
                else:
                    print("이미 선택된 요일입니다.")
            else:
                print(f"{DAY_MIN}부터 {DAY_MAX} 사이의 숫자를 입력하세요.")
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
    default_difficulty = 0.5
    while True:
        try:
            diff_input = (
                input(f"선호하는 난이도 ({DIFFICULTY_MIN}-{DIFFICULTY_MAX}, 기본값: {default_difficulty}): ").strip()
                or str(default_difficulty)
            )
            preferred_difficulty = float(diff_input)
            if DIFFICULTY_MIN <= preferred_difficulty <= DIFFICULTY_MAX:
                break
            print(f"{DIFFICULTY_MIN}부터 {DIFFICULTY_MAX} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    # 평점 선호
    default_rating = 3.0
    while True:
        try:
            rating_input = (
                input(f"선호하는 평점 ({RATING_MIN}-{RATING_MAX}, 기본값: {default_rating}): ").strip()
                or str(default_rating)
            )
            preferred_rating = float(rating_input)
            if RATING_MIN <= preferred_rating <= RATING_MAX:
                break
            print(f"{RATING_MIN}부터 {RATING_MAX} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")

    print("\n=== 입력된 선호도 정보 ===")
    print(f"학점 범위: {min_credits}~{max_credits}")
    print(f"선호 요일: {[DAY_NAMES[d] for d in preferred_days] if preferred_days else '없음'}")
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
        # 과목 데이터 + 학습 데이터를 courses.csv 한 파일에서 로드.
        # 실행 시점의 현재 작업 디렉터리가 아니라 timetable.py가 있는 프로젝트
        # 루트를 기준으로 찾으므로, 어디서 실행하든(IDE 실행 버튼, 다른 폴더에서
        # 커맨드 실행 등) 같은 courses.csv를 찾는다. 이 파일을 다른 학기/커리큘럼
        # 데이터로 통째로 교체하면 그대로 반영됨 (가천대 강의시간표를 자동으로
        # 긁어와 교체하고 싶다면 timetable_app/importers/gachon.py 참고).
        recommender, trained = build_recommender_from_csv(str(PROJECT_ROOT / "courses.csv"))

        if trained:
            print("✅ AI 모델이 훈련되었습니다.")
        else:
            print("ℹ️ courses.csv에 score 값이 없어 AI 모델 없이 규칙 기반으로 진행합니다.")

        preferences = get_user_preferences()
        recommender.set_user_preferences(preferences)

        recommendations = recommender.generate_recommendations()

        if recommendations:
            best_schedule, score = recommendations[0]

            html_content = generate_html(best_schedule, score)

            output_path = recommender.data_dir / "recommended_timetable.html"
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
