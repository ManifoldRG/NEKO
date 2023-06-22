from abc import ABC


class Task(ABC):
    def __init__(self):
        pass

    def sample_batch(self, vanilla_batch_size, prompted_batch_size, device, max_tokens=1024):
        pass

    def eval(self, model, n_iterations):
        pass        
    