import logging

def setup_logger():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('timetable.log'),
            logging.StreamHandler()
        ]
    ) 