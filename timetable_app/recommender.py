"""
시간표 추천 시스템의 핵심 클래스

과목 카탈로그 관리, AI 모델 학습/예측, 백트래킹 기반 시간표 탐색을 담당합니다.
"""
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import PROJECT_ROOT, persistence
from .features import extract_all_features, extract_preference_features
from .models import Course, CourseError, UserPreferences

# matplotlib 기본 폰트(DejaVu Sans)는 한글 글리프가 없어 특성 중요도 그래프의
# 한글 제목/라벨이 네모(tofu)로 깨진다. OS별로 흔히 깔려 있는 한글 폰트를
# 설치 여부를 확인해 가장 먼저 발견되는 것으로 설정하고, 하나도 없으면(예: 폰트
# 미설치 리눅스 CI) 조용히 기본 폰트로 남겨둔다 - 그래프가 안 예쁠 뿐 실행은
# 막지 않아야 하므로 예외를 던지지 않는다.
_KOREAN_FONT_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]


def _configure_korean_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_korean_font()


class ScheduleRecommender:
    """
    시간표 추천 시스템의 핵심 클래스

    주요 기능:
    - 과목 정보 관리
    - 사용자 선호도 설정
    - AI 기반 시간표 추천
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        ScheduleRecommender 초기화.

        data_dir: 저장된 시간표(schedules/)와 특성 중요도 그래프(feature_importance.png)를
        쓸 디렉터리. 지정하지 않으면 기존과 동일하게 PROJECT_ROOT를 쓴다(CLI 단일 사용자
        동작 그대로). 웹 등 여러 사용자가 동시에 쓰는 환경에서는 사용자/세션별 디렉터리를
        넘겨서 산출물이 서로 덮어쓰지 않도록 격리할 수 있다.
        """
        self.data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT
        self.schedules_dir = self.data_dir / "schedules"
        self.feature_importance_path = self.data_dir / "feature_importance.png"

        self.courses: List[Course] = []
        self.user_preferences: Optional[UserPreferences] = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: Optional[List[str]] = None

    def clone_with_data_dir(self, data_dir: Path) -> "ScheduleRecommender":
        """
        학습된 모델/과목 카탈로그는 그대로 공유하고 data_dir(저장 시간표 경로)만
        새로 지정한 가벼운 복제본을 만든다.

        웹처럼 여러 사용자가 동시에 요청을 보내는 환경에서, 비용이 큰 CSV 파싱 +
        모델 학습을 요청마다 반복하지 않으면서도 요청/세션마다 독립된
        user_preferences와 schedules_dir을 갖게 해 동시 요청 간 상태가 섞이지
        않도록 하기 위함. 원본과 복제본이 courses/model/scaler를 같은 객체로
        공유하지만, 두 값 모두 추천 생성 과정에서 읽기만 하고 쓰지 않으므로
        동시에 여러 복제본에서 접근해도 안전하다.
        """
        clone = ScheduleRecommender(data_dir=data_dir)
        clone.courses = self.courses
        clone.model = self.model
        clone.scaler = self.scaler
        clone.feature_names = self.feature_names
        return clone

    # ---------------------------------------------------------------- #
    # 과목 카탈로그 관리
    # ---------------------------------------------------------------- #

    def add_course(self, course: Course) -> None:
        """
        과목을 카탈로그에 추가.

        시간 충돌 여부는 여기서 검사하지 않습니다 - 서로 다른 과목이 시간이
        겹치는 것은 실제 수강편람에서 지극히 정상이며(같은 시간대의 다른 선택
        과목 등), 어떤 조합을 고를지는 generate_recommendations()의 백트래킹
        탐색이 결정할 몫입니다. 카탈로그 적재 단계에서 충돌 과목을 미리
        걸러내면 데이터가 클수록 상당수의 정상 과목이 조용히 사라집니다.
        """
        if any(existing.code == course.code for existing in self.courses):
            raise CourseError(f"이미 등록된 과목 코드입니다: {course.code}")
        self.courses.append(course)

    def load_courses_from_csv(self, csv_file: str) -> Optional[pd.DataFrame]:
        """
        CSV 한 파일에서 과목 카탈로그를 로드.

        같은 파일에 score 컬럼이 채워진 과목이 있으면 학습용 DataFrame을 반환하고,
        없으면 None을 반환합니다 (호출부는 None이면 학습을 건너뛰고 규칙 기반으로
        진행하면 됩니다).
        """
        courses, training_data = persistence.load_courses_from_csv(csv_file)
        for course in courses:
            try:
                self.add_course(course)
                logging.info(f"과목 추가 성공: {course.code} - {course.name}")
            except CourseError as e:
                logging.error(str(e))
        return training_data

    def save_schedule(self, name: str, schedule: List[Course]) -> None:
        """시간표 저장"""
        try:
            persistence.save_schedule_json(name, schedule, schedules_dir=self.schedules_dir)
            logging.info(f"시간표가 저장되었습니다: {name}")
        except Exception as e:
            logging.error(f"시간표 저장 실패: {str(e)}")
            raise

    def load_schedule(self, name: str) -> List[Course]:
        """저장된 시간표 불러오기"""
        try:
            return persistence.load_schedule_json(name, schedules_dir=self.schedules_dir)
        except Exception as e:
            logging.error(f"시간표 불러오기 실패: {str(e)}")
            raise

    def set_user_preferences(self, preferences: UserPreferences) -> None:
        """사용자 선호도 설정"""
        self.user_preferences = preferences

    def check_time_conflict(self, course1: Course, course2: Course) -> bool:
        """두 과목의 시간이 충돌하는지 확인"""
        for slot1 in course1.time_slots:
            for slot2 in course2.time_slots:
                if slot1.day == slot2.day:
                    # 시작 시간과 종료 시간이 같을 때는 충돌로 판단하지 않음
                    if (slot1.start_time < slot2.end_time and
                        slot1.end_time > slot2.start_time):
                        return True
        return False

    # ---------------------------------------------------------------- #
    # AI 모델 학습
    # ---------------------------------------------------------------- #

    def _build_feature_frame(self, courses: List[Course], fit: bool) -> pd.DataFrame:
        """
        과목 리스트를 특성 DataFrame으로 변환 (학습/예측 공통 경로).

        fit=True면 특성 이름을 새로 확정하고 스케일러를 그 데이터에 맞춰
        학습(fit)합니다. fit=False(예측)면 학습 시점의 feature_names에
        맞춰 열을 정렬하고, 이미 학습된 스케일러로만 변환(transform)합니다.
        """
        rows = [
            extract_all_features(course, self.courses, self.user_preferences)
            for course in courses
        ]
        df = pd.DataFrame(rows)

        if fit:
            self.feature_names = df.columns.tolist()
            df[self.feature_names] = self.scaler.fit_transform(df[self.feature_names])
        else:
            if self.feature_names is None:
                raise ValueError("모델이 훈련되지 않았습니다.")
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            df = df.reindex(columns=self.feature_names)
            df[self.feature_names] = self.scaler.transform(df[self.feature_names])

        return df

    def preprocess_features(self, courses: Optional[List[Course]] = None) -> Any:
        """특성 행렬 생성 (courses 미지정 시 카탈로그 전체 사용)"""
        target_courses = self.courses if courses is None else courses
        return self._build_feature_frame(target_courses, fit=True).values

    def train_model(self, training_data: pd.DataFrame) -> None:
        """
        AI 모델 훈련.

        training_data의 각 행을 'code' 컬럼으로 self.courses와 매칭해 학습
        데이터를 구성합니다 (단순히 행 순서로 짝짓지 않음 - CSV와 JSON의
        과목 순서/개수가 어긋나면 엉뚱한 과목에 엉뚱한 점수가 붙는 것을
        방지하기 위함).
        """
        course_by_code = {course.code: course for course in self.courses}

        matched_courses: List[Course] = []
        scores: List[float] = []
        for _, row in training_data.iterrows():
            code = str(row['code'])
            course = course_by_code.get(code)
            if course is None:
                logging.warning(
                    f"training_data.csv의 과목 코드 '{code}'가 현재 과목 목록에 없어 "
                    "학습에서 제외합니다."
                )
                continue
            matched_courses.append(course)
            scores.append(row['score'])

        if len(matched_courses) < 2:
            raise ValueError(
                "학습에 사용할 수 있는 데이터가 부족합니다. training_data.csv의 "
                "code 컬럼이 timetable.json의 과목 코드와 일치하는지 확인하세요."
            )

        X = self.preprocess_features(matched_courses)
        y = np.array(scores, dtype=float)

        logging.info(f"학습에 사용된 과목 수: {len(matched_courses)}")
        logging.info(f"특성 개수: {X.shape[1]}")
        logging.info(f"특성 이름: {self.feature_names}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        logging.info(f"모델 훈련 완료 - 훈련 점수: {train_score:.3f}, 테스트 점수: {test_score:.3f}")

        self.analyze_feature_importance()

    def analyze_feature_importance(self) -> None:
        """특성 중요도 분석 및 시각화"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")

        importances = self.model.feature_importances_

        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances, y=self.feature_names)
        plt.title("특성 중요도")
        plt.tight_layout()
        self.feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(self.feature_importance_path))
        plt.close()

    # ---------------------------------------------------------------- #
    # 예측
    # ---------------------------------------------------------------- #

    def _predict_course_scores(self, courses: List[Course]) -> Dict[str, float]:
        """
        여러 과목의 예측 점수를 한 번의 배치 예측으로 계산.

        모델은 과목 단위 특성만으로 학습되어(과목 간 상호작용 특성이 없음)
        시간표 점수는 결국 소속 과목들의 예측 점수 평균이 됩니다. 따라서
        후보 과목 전체를 한 번만 예측해 두면, 이후 백트래킹 탐색에서
        조합마다 매번 특성 추출 → 정규화 → model.predict를 반복할 필요가
        없어집니다(과목 N개, 시간표 조합 최대 2^N개 기준으로 예측 호출을
        2^N번에서 1번으로 줄임).
        """
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        if not courses:
            return {}

        df = self._build_feature_frame(courses, fit=False)
        predictions = self.model.predict(df.values)
        return {course.code: float(pred) for course, pred in zip(courses, predictions)}

    def predict_schedule_score(self, schedule: List[Course]) -> float:
        """시간표(과목 조합) 점수 예측: 소속 과목 예측 점수의 평균"""
        if not schedule:
            return 0.0
        scores = self._predict_course_scores(schedule)
        return sum(scores.values()) / len(scores)

    def _rule_based_course_score(self, course: Course) -> float:
        """
        AI 모델이 없을 때 사용하는 과목별 규칙 기반 점수.

        선호 교수/요일 일치 여부와 난이도/평점 근접도를 0~1 사이로 평균 낸
        값으로, predict_schedule_score와 같은 척도(과목별 점수의 평균)를
        따르도록 맞췄습니다.
        """
        pref = extract_preference_features(course, self.user_preferences)
        return (
            pref.get('difficulty_match', 0.5)
            + pref.get('rating_match', 0.5)
            + (1.0 if pref.get('matches_preferred_professor') else 0.5)
            + (1.0 if pref.get('matches_preferred_days') else 0.5)
        ) / 4

    # ---------------------------------------------------------------- #
    # 시간표 탐색
    # ---------------------------------------------------------------- #

    # 후보 과목이 이 개수를 넘으면 전수 백트래킹(2^N) 대신 무작위 다중 시작
    # 그리디 탐색으로 전환합니다. N=22면 2^22(약 400만) 조합이라 아직은
    # 수 초 내로 감당되지만, 그보다 커지면 실행 시간이 감당할 수 없이
    # 커지기 때문입니다. (과목 충돌 그래프 위에서 학점 합 제약을 만족하는
    # 최고점 부분집합을 찾는 문제는 일반적으로 NP-hard이므로, 큰 카탈로그에서는
    # 정확한 최적해 대신 빠른 근사해를 반환합니다.)
    EXACT_SEARCH_MAX_CANDIDATES = 22
    GREEDY_SEARCH_TRIALS = 300

    def generate_recommendations(self) -> List[Tuple[List[Course], float]]:
        """AI 기반(또는 모델이 없을 경우 규칙 기반) 추천 시간표 생성"""
        if not self.user_preferences:
            raise ValueError("사용자 선호도가 설정되지 않았습니다.")

        excluded = set(self.user_preferences.excluded_courses)
        preferred_professors = set(self.user_preferences.preferred_professors)

        # 제외 과목을 뺀 뒤, 선호 교수의 과목이 먼저 탐색되도록 정렬
        candidates = [c for c in self.courses if c.code not in excluded]
        candidates.sort(key=lambda c: c.professor not in preferred_professors)

        if not candidates:
            return []

        if self.model is not None:
            course_scores = self._predict_course_scores(candidates)
        else:
            course_scores = {c.code: self._rule_based_course_score(c) for c in candidates}

        def score_fn(schedule: List[Course]) -> float:
            return sum(course_scores[c.code] for c in schedule) / len(schedule)

        min_credits = self.user_preferences.min_credits
        max_credits = self.user_preferences.max_credits

        if len(candidates) <= self.EXACT_SEARCH_MAX_CANDIDATES:
            valid_schedules = self._exact_search(candidates, score_fn, min_credits, max_credits)
        else:
            logging.info(
                f"후보 과목이 {len(candidates)}개로 많아 전수 탐색 대신 "
                "무작위 다중 시작 그리디 탐색을 사용합니다."
            )
            valid_schedules = self._greedy_search(
                candidates, course_scores, score_fn, min_credits, max_credits
            )

        valid_schedules.sort(key=lambda x: x[1], reverse=True)
        return valid_schedules[:5]  # 상위 5개 시간표만 반환

    def _exact_search(
        self,
        candidates: List[Course],
        score_fn,
        min_credits: int,
        max_credits: int,
    ) -> List[Tuple[List[Course], float]]:
        """백트래킹으로 시간 충돌 없는 모든 시간표 조합을 탐색 (소규모 카탈로그용, 최적해 보장)"""
        # 가지치기용: index부터 끝까지 남은 과목을 모두 더한 학점
        # (이걸 다 더해도 min_credits에 못 미치면 그 가지는 더 볼 필요가 없음)
        suffix_credits = [0] * (len(candidates) + 1)
        for i in range(len(candidates) - 1, -1, -1):
            suffix_credits[i] = suffix_credits[i + 1] + candidates[i].credits

        valid_schedules: List[Tuple[List[Course], float]] = []

        def backtrack(index: int, schedule: List[Course], credits: int) -> None:
            if credits >= min_credits:
                valid_schedules.append((schedule.copy(), score_fn(schedule)))
                if credits >= max_credits:
                    return

            if index >= len(candidates) or credits + suffix_credits[index] < min_credits:
                return

            current_course = candidates[index]

            can_add = (
                credits + current_course.credits <= max_credits
                and not any(self.check_time_conflict(current_course, c) for c in schedule)
            )

            if can_add:
                schedule.append(current_course)
                backtrack(index + 1, schedule, credits + current_course.credits)
                schedule.pop()

            backtrack(index + 1, schedule, credits)

        backtrack(0, [], 0)
        return valid_schedules

    def _greedy_search(
        self,
        candidates: List[Course],
        course_scores: Dict[str, float],
        score_fn,
        min_credits: int,
        max_credits: int,
    ) -> List[Tuple[List[Course], float]]:
        """
        무작위 다중 시작 그리디 탐색 (대규모 카탈로그용, 근사해).

        매 시행마다 과목 점수에 약간의 무작위 노이즈를 섞어 정렬한 뒤,
        앞에서부터 충돌 없이 학점 상한 이내로 그리디하게 담습니다.
        시행 횟수가 과목 수와 무관하게 고정되어 있어 카탈로그가 아무리
        커져도 실행 시간이 일정 수준으로 유지됩니다.
        """
        rng = random.Random(42)
        seen_combinations = set()
        results: List[Tuple[List[Course], float]] = []

        for _ in range(self.GREEDY_SEARCH_TRIALS):
            order = sorted(
                candidates,
                key=lambda c: course_scores[c.code] + rng.uniform(-0.15, 0.15),
                reverse=True,
            )

            schedule: List[Course] = []
            credits = 0
            for course in order:
                if credits + course.credits > max_credits:
                    continue
                if any(self.check_time_conflict(course, c) for c in schedule):
                    continue
                schedule.append(course)
                credits += course.credits
                if credits >= max_credits:
                    break

            if credits < min_credits:
                continue

            key = frozenset(c.code for c in schedule)
            if key in seen_combinations:
                continue
            seen_combinations.add(key)
            results.append((schedule, score_fn(schedule)))

        return results


def build_recommender_from_csv(
    csv_path: str,
    data_dir: Optional[Path] = None,
) -> Tuple[ScheduleRecommender, bool]:
    """
    courses.csv 하나로 ScheduleRecommender를 만들고, score 데이터가 있으면
    모델까지 학습해 둔다. CLI(cli.main)와 웹(web._load_base_recommender) 양쪽
    진입점이 실행 시작 시 똑같이 거치는 절차라 여기 하나로 모아 중복을 없앴다.

    반환값의 두 번째 값은 모델을 실제로 학습했는지 여부 - 호출부가 사용자에게
    "AI 모델 학습됨/규칙 기반" 여부를 알려줄 때 쓰면 된다.
    """
    recommender = ScheduleRecommender(data_dir=data_dir)
    training_data = recommender.load_courses_from_csv(csv_path)
    trained = training_data is not None and not training_data.empty
    if trained:
        recommender.train_model(training_data)
    return recommender, trained
