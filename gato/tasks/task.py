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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        sig = signature(cls.__init__)
        type_param = sig.parameters.get('task_type')
        
        if not type_param:
            raise TypeError(f"{cls.__name__}.__init__ must have a 'task_type' parameter.")
        
        if type_param.annotation != str:
            raise TypeError(f"{cls.__name__}.__init__ 'task_type' parameter must be of type 'str'.")

    def sample_batch(self):
        pass

    def evaluate(self, model):
        pass        
