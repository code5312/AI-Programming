"""AI 기반 시간표 추천 시스템 패키지."""
import logging
import sys
from pathlib import Path

# timetable_app/ 폴더의 부모 = 프로젝트 루트. 실행 시점의 현재 작업 디렉터리(cwd)가
# 아니라 이 파일의 실제 위치를 기준으로 삼아, 어떤 위치에서 실행하든(더블클릭, IDE의
# 실행 버튼, 다른 폴더에서 커맨드 실행 등) 항상 같은 courses.csv/schedules/로그 파일을
# 찾도록 함.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Windows 콘솔은 로캘에 따라 기본 인코딩이 cp949 등으로 잡혀 있어, print()에 쓰는
# 이모지(✅/❌/ℹ️)가 UnicodeEncodeError를 일으킬 수 있다. 콘솔을 UTF-8로 강제해
# 실행 환경(터미널/IDE 실행 버튼 등)에 상관없이 항상 출력되도록 함.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(PROJECT_ROOT / 'timetable.log')),
        logging.StreamHandler()
    ]
)
