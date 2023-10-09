from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
import minari
import h5py
import numpy as np

from gato.tasks.control_task import ControlTask

class TaskDataset(Dataset):
    def __init__(
      self,
      device,
      seq_len: int,
      prompt_ep_proportion: float,
      tasks: List[ControlTask],
    ):
        super().__init__()
        self.device = device
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.seq_len = seq_len
        self.prompt_ep_proportion = prompt_ep_proportion
        assert self.n_tasks > 0, "No tasks provided"

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, idx):
        task = self.tasks[idx]

        use_prompt = np.random.rand() < self.prompt_ep_proportion
        prompt_type = None
        if use_prompt:
            prompt_type = np.random.choice(['end', 'uniform'])
            prompt_types = [prompt_type]
            prompt_props = [task.training_prompt_len_proportion]
        else:
            prompt_types = [None]
            prompt_props = [0]


        episode_dicts = task.sample_batch_configurable(
            batch_size = 1,
            prompt_proportions = prompt_props,
            prompt_types = prompt_types,
            device = 'cpu',
            max_tokens = self.seq_len,
        )[0]
        return episode_dicts
