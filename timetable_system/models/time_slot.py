from datetime import time
from dataclasses import dataclass

class TimeSlotError(Exception):
    """시간 슬롯 관련 에러"""
    pass

@dataclass
class TimeSlot:
    day: int
    start_time: time
    end_time: time

    def __post_init__(self):
        if not 0 <= self.day <= 4:
            raise TimeSlotError("요일은 0(월)부터 4(금)까지여야 합니다.")
        if self.start_time >= self.end_time:
            raise TimeSlotError("시작 시간은 종료 시간보다 빨라야 합니다.")
        
        if not (time(9, 0) <= self.start_time <= time(18, 0) and 
                time(9, 0) <= self.end_time <= time(18, 0)):
            raise TimeSlotError("수업 시간은 9시부터 18시 사이여야 합니다.")
        
        duration = (self.end_time.hour - self.start_time.hour) + \
                  (self.end_time.minute - self.start_time.minute) / 60
        if not 1 <= duration <= 3:
            raise TimeSlotError("수업 시간은 1시간에서 3시간 사이여야 합니다.") 

    def overlaps_with(self, other: "TimeSlot") -> bool:
        if self.day != other.day:
            return False
        # [start, end) 구간이 겹치는지 확인
        return self.start_time < other.end_time and self.end_time > other.start_time 