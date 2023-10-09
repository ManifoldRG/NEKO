import gymnasium as gym
import numpy as np
import torch
import minari
import h5py
from minari.dataset.minari_dataset import EpisodeData

from gato.tasks.task import Task, TaskTypeEnum

supported_spaces = [
    gym.spaces.Box,
    gym.spaces.Discrete,
]

def tokens_per_space(space):
    if type(space) == gym.spaces.Box:
        return space.shape[0]
    elif type(space) == gym.spaces.Discrete:
        return 1
    else:
        raise NotImplementedError(f'Unsupported space: {space}')
    
   
class ControlMinimalTask:
    def __init__(
            self,
            task
        ):
        self.episode_indices = task.dataset._episode_indices
        self.tokens_per_timestep = task.tokens_per_timestep
        self.episode_lengths = task.episode_lengths
        self.prompt_types = task.prompt_types
        self.observation_space = task.env.observation_space
        self.action_space = task.env.action_space
        self.image_transform = task.image_transform #TODO, check
        self.action_tokens = task.action_tokens
        self.action_str = task.action_str
        self.obs_str = task.obs_str
        self.image_transform = task.image_transform
        self.training_prompt_len_proportion = task.training_prompt_len_proportion
        self.data_path = task.dataset.spec.data_path
        #self.get_episodes_sliced = task.get_episodes_sliced

    def sample_batch_configurable(
            self, batch_size: int, 
            device: str, 
            prompt_proportions: list, 
            prompt_types: list, 
            max_tokens: int = 1024, 
            ep_ids = None,
        ):
        # Samples a batch of episodes, where each episode has maximum of max_tokens tokens
        # This will return a list of dictionaries, where each dicionary contains variable length tensors,
        # This is in constrast to returning single tensors which contain all episodes with padding

        # Maximum number of timesteps we can fit in context
        num_timesteps = max_tokens // self.tokens_per_timestep

        # sample episode indices
        episode_indices = np.random.choice(self.episode_indices, size=batch_size, replace=False)
        timesteps_for_mains = []
        timesteps_for_prompts = []
        for i, episode_index in enumerate(episode_indices):
            num_timesteps_for_main = int(num_timesteps * (1 - prompt_proportions[i]))
            ep_len = self.episode_lengths[episode_index]
            if num_timesteps_for_main >= ep_len:
                # sample entire episode
                start = 0
                end = ep_len - 1
            else:
                # sample which timestep to start with, may want to change
                start = np.random.randint(0, ep_len - num_timesteps_for_main)
                end = start + num_timesteps_for_main - 1
            timesteps_for_mains.append((start, end))

            num_timesteps_for_prompt = num_timesteps - num_timesteps_for_main
            prompt_type = prompt_types[i]
            if num_timesteps_for_prompt > 0:
                assert prompt_type in self.prompt_types, 'Invalid prompt type'
                if num_timesteps_for_prompt >= ep_len:
                    # sample entire episode
                    prompt_start = 0
                    prompt_end = ep_len - 1
                elif prompt_type == 'start':
                    prompt_start = 0
                    prompt_end = num_timesteps_for_prompt - 1
                elif prompt_type == 'end':
                    prompt_end = ep_len - 1
                    prompt_start = prompt_end - num_timesteps_for_prompt + 1
                elif prompt_type == 'uniform':
                    prompt_start = np.random.randint(0, ep_len - num_timesteps_for_prompt)
                    prompt_end = prompt_start + num_timesteps_for_prompt - 1
                else:
                    raise NotImplementedError(f'Invalid prompt type: {prompt_type}')
            else:
                prompt_start = None
                prompt_end = None
            timesteps_for_prompts.append((prompt_start, prompt_end))
        # now we have indices to get from each episode
        episodes_data = self.get_episodes_sliced(episode_indices, timesteps_for_mains, timesteps_for_prompts)


        # Convert to dictionary for each episode
        episode_dicts = []

        for i in range(batch_size):
            actions = episodes_data['actions'][i]
            observations = episodes_data['observations'][i]

            # convert observations to tensors
            if type(self.observation_space) == gym.spaces.Box:
                observations = torch.tensor(observations, dtype=torch.float32, device=device)
            elif type(self.observation_space) == gym.spaces.Discrete:
                observations = torch.tensor(observations, dtype=torch.int32, device=device)
            
            # apply image transforms
            if self.image_transform is not None:
                observations = self.image_transform.transform(observations)
            
            # convert actions to tensors
            if type(self.action_space) == gym.spaces.Box:
                actions = torch.tensor(actions, dtype=torch.float32, device=device)
            elif type(self.action_space) == gym.spaces.Discrete:
                actions = torch.tensor(actions, dtype=torch.int32, device=device)
            
            # make sure actions are 2D
            actions = actions.reshape(actions.shape[0], self.action_tokens)
            episode_dict = {
                self.action_str: actions,
                self.obs_str: observations,
            }
            episode_dicts.append(episode_dict)
        return episode_dicts

    def get_episodes_sliced(self, episode_indices, timesteps_for_mains, timesteps_for_prompts):
        episodes_data = {
            'actions': [],
            'observations': [],
        }
        with h5py.File(self.data_path, 'r') as f:
            # iterate over episodes
            for i, episode_index in enumerate(episode_indices):
                ep_group = f[f"episode_{episode_index}"]
                main_start, main_end = timesteps_for_mains[i]
                prompt_start, prompt_end = timesteps_for_prompts[i]
                obs = ep_group["observations"][main_start:(main_end + 1),]
                actions = ep_group["actions"][main_start:(main_end + 1),]
                if prompt_start is not None:
                    prompt_obs = ep_group["observations"][prompt_start:(prompt_end + 1),]
                    prompt_actions = ep_group["actions"][prompt_start:(prompt_end + 1),]
                    obs = np.concatenate([prompt_obs, obs], axis=0)
                    actions = np.concatenate([prompt_actions, actions], axis=0)
                episodes_data['observations'].append(obs)
                episodes_data['actions'].append(actions)
        return episodes_data