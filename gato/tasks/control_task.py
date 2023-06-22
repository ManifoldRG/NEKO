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
            training_prompt_len_proportion=0.5, 
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
        self.training_prompt_len_proportion = training_prompt_len_proportion 
        assert self.training_prompt_len_proportion >= 0 and self.training_prompt_len_proportion <= 1
        
        # Specifies if prompt should come from the same episode as the main chunk
        self.share_prompt_episodes = share_prompt_episodes
        # Ways of sampling prompts
        self.prompt_types = ['start', 'end','uniform']

    # TODO
    def evaluate(self, model, n_iterations):
        return {}
    
    def sample_batch(self, vanilla_batch_size:int , prompted_batch_sizes: dict, device, max_tokens=1024):
        for prompt_type in prompt_types:
            assert prompt_type in self.prompt_types

        episode_dicts = []

        # Determine batch sizes
        prompted_batch_size = 0
        for prompt_type, batch_size in prompted_batch_sizes.items():
            assert prompt_type in self.prompt_types
            prompted_batch_size += batch_size

        batch_size = vanilla_batch_size + prompted_batch_size
        prompt_propotions = []
        prompt_types = []

        for i in range(vanilla_batch_size):
            prompt_propotions.append(0)
            prompt_types.append(None) # should not be used
        
        for prompt_type, prompt_batch_size in prompted_batch_sizes.items():
            prompt_propotions += [self.training_prompt_len_proportion] * prompt_batch_size
            prompt_types += [prompt_type] * prompt_batch_size
        
        assert len(prompt_propotions) == batch_size and len(prompt_types) == batch_size, f'Batch size mismatch: {len(prompt_propotions)} != {batch_size} or {len(prompt_types)} != {batch_size}'

        episode_dicts = self.sample_batch_configurable(
            batch_size,
            device,
            prompt_propotions,
            prompt_types,
            max_tokens=max_tokens,
            share_prompt_episodes=self.share_prompt_episodes
        )
        
        return episode_dicts

    #def sample_batch_configurable(self, batch_size, device, max_tokens=1024, prompt_proportion=0.5, prompt_type='end', share_prompt_episodes=True):
    def sample_batch_configurable(self, batch_size: int, device: str, prompt_proportions: list[float], prompt_types: list[str], max_tokens: int = 1024, share_prompt_episodes=True):
        # Samples a batch of episodes, where each episode has maximum of max_tokens tokens
        # This will return a list of dictionaries, where each dicionary contains variable length tensors,
        # This is in constrast to returning single tensors which contain all episodes with padding

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
        timesteps_for_mains = []
        timesteps_for_prompts = []
        for i, episode in enumerate(main_episodes):
            timesteps_for_main = round(num_timesteps * (1 - prompt_proportions[i]))
            timesteps_for_mains.append(timesteps_for_main) # max main size
            timesteps_for_prompts.append(num_timesteps - timesteps_for_main) # max prompt size

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
        for i, episode in enumerate(prompt_episodes):
            ep_len = len(episode)
            timesteps_for_prompt = timesteps_for_prompts[i]
            prompt_type = prompt_types[i]
            if timesteps_for_prompt > 0:
                assert prompt_type in self.prompt_types, 'Invalid prompt type'
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

                # Extract prompt and add to main chunk
                prompt_obs = episode.observations[prompt_start:(prompt_end + 1),]
                prompt_actions = episode.actions[prompt_start:(prompt_end + 1),]
                episodes_data['observations'][i] = np.concatenate([prompt_obs, episodes_data['observations'][i]], axis=0)
                episodes_data['actions'][i] = np.concatenate([prompt_actions, episodes_data['actions'][i]], axis=0)


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
        