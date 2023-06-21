import gymnasium as gym
import minari

from gato.tasks.task import Task

supported_spaces = [
    gym.spaces.Box,
    gym.spaces.Discrete,
]

class ControlTask(Task):
    def __init__(self, env_name: str, env: gym.Env, dataset: minari.MinariDataset):
        super().__init__()
        self.env_name = env_name
        self.env = env
        self.dataset = dataset

        assert self.env.action_space in supported_spaces, f'Unsupported action space: {self.env.action_space}'
        assert self.env.observation_space in supported_spaces, f'Unsupported observation space: {self.env.observation_space}'
    
    def sample_batch(self, batch_size, device=None):
        pass