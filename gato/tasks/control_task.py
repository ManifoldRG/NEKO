import gymnasium as gym
import numpy as np
import torch
import minari

from gato.tasks.task import Task

supported_spaces = [
    gym.spaces.Box,
    gym.spaces.Discrete,
]

def tokens_per_space(space):
    if type(space) == gym.spaces.Box:
        return space.shape[0]
    elif type(space) == gym.spaces.Discrete:
        return space.n
    else:
        raise NotImplementedError(f'Unsupported space: {space}')

class ControlTask(Task):
    def __init__(
            self, 
            env_name: str, 
            env: gym.Env, 
            dataset: minari.MinariDataset, 
            training_prompt_proportion=0.5, 
            share_prompt_episodes=True
        ):
        super().__init__()
        self.env_name = env_name
        self.env = env
        self.dataset = dataset
    
        assert type(self.env.action_space) in supported_spaces, f'Unsupported action space: {self.env.action_space}'
        assert type(self.env.observation_space) in supported_spaces, f'Unsupported observation space: {self.env.observation_space}'

        self.action_tokens = tokens_per_space(self.env.action_space)
        self.observation_tokens = tokens_per_space(self.env.observation_space)

        self.tokens_per_timestep =  self.action_tokens + self.observation_tokens + 1 # additional separator token
        
        # If sampled episode needs a prompt, this specifies what proportion of tokens should be from the prompt
        self.training_prompt_proportion = training_prompt_proportion 
        assert self.training_prompt_proportion >= 0 and self.training_prompt_proportion <= 1
        
        # Specifies if prompt should come from the same episode as the main chunk
        self.share_prompt_episodes = share_prompt_episodes
        # Ways of sampling prompts
        self.prompt_types = ['start', 'end','uniform'] 
    
    def sample_batch(self, vanilla_batch_size, prompted_batch_size, device, max_tokens=1024, prompt_type='end'):
        assert prompt_type in self.prompt_types

        vanilla_dicts = []
        prompted_dicts = []
        
        if vanilla_batch_size > 0:
            vanilla_dicts = self.sample_batch_configurable(
                self, 
                vanilla_batch_size, 
                device, 
                max_tokens=max_tokens, 
                prompt_proportion=0, 
                share_prompt_episodes=True
            )
        if prompted_batch_size > 0:

            prompted_dicts = self.sample_batch_configurable(
                self, 
                prompted_batch_size, 
                device, 
                max_tokens=max_tokens, 
                prompt_proportion=self.training_prompt_proportion, 
                share_prompt_episodes=self.share_prompt_episodes
            )

        episode_dicts = vanilla_dicts + prompted_dicts       
        return episode_dicts

    def sample_batch_configurable(self, batch_size, device, max_tokens=1024, prompt_proportion=0.5, prompt_type='end', share_prompt_episodes=True):
        # Samples a batch of episodes, where each episode has maximum of max_tokens tokens
        # This will return a list of dictionaries, where each dicionary contains variable length tensors,
        # This is in constrast to returning single tensors which contain all episodes with padding

        assert prompt_type in self.prompt_types
        #assert prompt_style in ['inclusive', 'exclusive'] 

        # Maximum number of timesteps we can fit in context
        num_timesteps = max_tokens // self.tokens_per_timestep

        # List of numpy arrays for each episode 
        episodes_data = {
            'actions': [],
            'observations': [],
        }

        all_episodes = self.dataset.sample_episodes(n_episodes=batch_size)
        if share_prompt_episodes:
            main_episodes = all_episodes
            prompt_episodes = all_episodes
        else:
            all_episodes = self.dataset.sample_episodes(n_episodes=batch_size)
            main_episodes = all_episodes
            # prompts come from different episodes
            prompt_episodes = all_episodes[1:] + all_episodes[:1]

        # If prompt_proportion is nonzero, then each episode has a proportion of its tokens replaced with a prompt 

        # sample "non-prompt" chunk from each episode
        timesteps_for_main = round(num_timesteps * (1 - prompt_proportion)) # max main size
        timesteps_for_prompt = num_timesteps - timesteps_for_main # max prompt size
        for episode in main_episodes:
            if timesteps_for_main >= len(episode):
                # sample entire episode
                start = 0
                end = len(episode) - 1
            else:
                # sample which timestep to start with
                start = np.random.randint(0, len(episode) - timesteps_for_main)
                end = start + timesteps_for_main
            observations = episode.observations[start:end,]
            actions = episode.actions[start:end,]

            episodes_data['observations'].append(observations)
            episodes_data['actions'].append(actions)

        # add prompt
        for episode in prompt_episodes:
            ep_len = len(episode)

            if timesteps_for_prompt >= ep_len:
                # sample entire episode
                prompt_start = 0
                prompt_end = ep_len - 1
            if prompt_type == 'start':
                prompt_start = 0
                prompt_end = timesteps_for_prompt - 1
            elif prompt_type == 'end':
                prompt_end = ep_len - 1
                prompt_start = prompt_end - timesteps_for_prompt + 1
            elif prompt_type == 'uniform':
                prompt_start = np.random.randint(0, ep_len - timesteps_for_prompt)
                prompt_end = prompt_start + timesteps_for_prompt - 1

        # Convert to dictionary for each episode
        episode_dicts = []

        for i in range(batch_size):
            actions = episodes_data['actions'][i]
            observations = episodes_data['observations'][i]

            # convert observations to tensors
            if type(self.env.observation_space) == gym.spaces.Box:
                observations = torch.tensor(observations, dtype=torch.float32, device=device)
                obs_str = 'continuous_obs'
            elif type(self.env.observation_space) == gym.spaces.Discrete:
                observations = torch.tensor(observations, dtype=torch.int32, device=device)
                obs_str = 'discrete_obs'
            
            # convert actions to tensors
            if type(self.env.action_space) == gym.spaces.Box:
                actions = torch.tensor(actions, dtype=torch.float32, device=device)
                action_str = 'continuous_actions'
            elif type(self.env.action_space) == gym.spaces.Discrete:
                actions = torch.tensor(actions, dtype=torch.int32, device=device)
                action_str = 'discrete_actions'

            episode_dict = {
                action_str: actions,
                obs_str: observations,
            }
            episode_dicts.append(episode_dict)

        return episode_dicts
        