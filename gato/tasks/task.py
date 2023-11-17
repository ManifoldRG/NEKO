from abc import ABC
from inspect import signature
from enum import Enum

class TaskTypeEnum(Enum):
    CONTROL = "control"
    TEXT = "text"
    # add more as we add more modalities

class Task(ABC):
    def __init__(self, task_type: str):
        if task_type not in TaskTypeEnum._value2member_map_:
            raise ValueError(f"'type' must be one of {', '.join([e.value for e in TaskTypeEnum])}")
        self.task_type = task_type

    def sample_batch(self):
        pass

    def evaluate(self, model):
        pass        
