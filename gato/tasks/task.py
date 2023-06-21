from abc import ABC


class Task(ABC):
    def __init__(self):
        pass

    def sample_batch(self, batch_size):
        pass

    def eval(self, model, n_iterations):
        pass        
    