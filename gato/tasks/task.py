from abc import ABC

class Task(ABC):
    def sample_batch(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()
