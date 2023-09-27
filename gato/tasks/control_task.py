import gymnasium as gym
import numpy as np
import torch
import minari
import h5py
from minari.dataset.minari_dataset import EpisodeData

from gato.tasks.task import Task

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
    
   
class ControlTask(Task):
    def __init__(
            self, 
            env_name: str, 
            env: gym.Env, 
            dataset: minari.MinariDataset, 
            context_len: int,
            args,
            training_prompt_len_proportion=0.5, 
            share_prompt_episodes=True,
            top_k_prompting=None
        ):
        super().__init__()
        self.name = env_name
        self.is_atari = 'ALE' in env_name
        self.env = env
        self.dataset = dataset
        self.args = args

        self.action_type = type(self.env.action_space)
        self.observation_type = type(self.env.observation_space)
        assert self.action_type in supported_spaces, f'Unsupported action space: {self.env.action_space}'
        assert self.observation_type in supported_spaces, f'Unsupported observation space: {self.env.observation_space}'

        # Determine types of obseravation, action for task
        if type(self.env.observation_space) == gym.spaces.Box:
            if len(self.env.observation_space.shape) == 2 or len(self.env.observation_space.shape) == 3:
                obs_str = 'images'
            else:
                obs_str = 'continuous_obs'
        elif type(self.env.observation_space) == gym.spaces.Discrete:
            obs_str = 'discrete_obs'
        self.obs_str = obs_str

        if obs_str == 'images':
            self.image_transform = ControlImageTransform(env, args.patch_size)
        else:
            self.image_transform = None
        
        if type(self.env.action_space) == gym.spaces.Box:
            action_str = 'continuous_actions'
        elif type(self.env.action_space) == gym.spaces.Discrete:
            action_str = 'discrete_actions'
        self.action_str = action_str


        self.action_tokens = tokens_per_space(self.env.action_space)
        if obs_str == 'images':
            # Calculate tokens after image transform
            image_shape = self.image_transform.transform(torch.tensor(env.observation_space.sample())).shape
            self.observation_tokens = image_shape[-1] // args.patch_size * image_shape[-2] // args.patch_size
        else:
            self.observation_tokens = tokens_per_space(self.env.observation_space)

        self.tokens_per_timestep =  self.action_tokens + self.observation_tokens + 1 # additional separator token
        assert context_len >= self.tokens_per_timestep, f'Context length must be at least {self.tokens_per_timestep} for env {env_name}'

        # If sampled episode needs a prompt, this specifies what proportion of tokens should be from the prompt
        self.training_prompt_len_proportion = training_prompt_len_proportion 
        assert self.training_prompt_len_proportion >= 0 and self.training_prompt_len_proportion <= 1
        
        # Specifies if prompt should come from the same episode as the main chunk during training
        self.share_prompt_episodes = share_prompt_episodes

        # Ways of sampling prompts
        self.prompt_types = ['start', 'end','uniform']

        # If prompts should be sampled from top k episodes, or uniform during eval
        self.top_k_prompting = top_k_prompting
        if self.top_k_prompting is not None:
            assert self.top_k_prompting > 0 and self.top_k_prompting <= self.dataset.total_episodes, 'top k must be between 0 and total episodes for all datasets'
            # calculate top k ep ids
            ep_returns = np.array([ep.rewards.sum() for ep in self.dataset])
            self.top_ids = np.argsort(ep_returns)[-self.top_k_prompting:]
        else:
            self.top_ids = None
        
        # Calculate length of each episode
        self.episode_indices = self.dataset._episode_indices
        self.episode_lengths = []
        prev_index = -1
        for i in self.episode_indices:
            assert i == prev_index + 1, 'Episode indices must be consecutive'
            with h5py.File(self.dataset.spec.data_path, 'r') as f:
                self.episode_lengths.append(f[f'episode_{i}']['rewards'].shape[0])
            prev_index = i
    def evaluate(self, model, n_iterations, deterministic=True, promptless_eval=False):
        # serial evaluation
        returns = []
        clipped_returns = []
        ep_lens = []
        metrics = {}
        max_len = self.args.max_eval_len

        context_timesteps = model.context_len // self.tokens_per_timestep # amount of timesteps that fit into context

        for i in range(n_iterations):
            observation, info = self.env.reset()

            # sample prompt
            input_dict = self.sample_batch_configurable(batch_size=1, device=model.device, prompt_proportions=[1.], prompt_types = ['end'], max_tokens = model.context_len, share_prompt_episodes=True,ep_ids=self.top_ids)[0]
            
            # infer dtypes
            action_type = input_dict[self.action_str].dtype

            if promptless_eval:
                input_dict = None
            done = False
            ep_return = 0
            ep_clipped_return = 0
            ep_len = 0
            while not done:
                new_obs = torch.tensor(observation, device=model.device).unsqueeze(0)
                if self.image_transform is not None:
                    new_obs = self.image_transform.transform(new_obs)
                # append new observation, and pad actions
                if input_dict is not None:
                    input_dict[self.obs_str] = torch.cat([input_dict[self.obs_str], new_obs], dim=0)
                    input_dict[self.action_str] = torch.cat([input_dict[self.action_str], torch.zeros(1, self.action_tokens, device=model.device, dtype=action_type)], dim=0)
                else:
                    input_dict = {
                        self.obs_str: new_obs,
                        self.action_str: torch.zeros(1, self.action_tokens, device=model.device, dtype=action_type),
                    }

                # trim to context length
                input_dict[self.obs_str] = input_dict[self.obs_str][-context_timesteps:,]
                input_dict[self.action_str] = input_dict[self.action_str][-context_timesteps:,]
                action = model.predict_control(input_dict, task=self, deterministic=deterministic)
                input_dict[self.action_str][-1,] = action
                np_action = action.cpu().numpy()
                observation, reward, terminated, truncated, info = self.env.step(np_action)
                done = terminated or truncated
                ep_return += reward 
                ep_clipped_return += np.clip(reward, -1.0, 1.0)
                ep_len += 1
                if max_len is not None and ep_len >= max_len:
                    done = True
            returns.append(ep_return)
            clipped_returns.append(ep_clipped_return)
            ep_lens.append(ep_len)

        metrics['mean_return'] = np.mean(returns)
        metrics['mean_episode_len'] = np.mean(ep_lens)
        # Only log clipped return for atari
        if self.is_atari:
            metrics['mean_clipped_return'] = np.mean(clipped_returns)
        return metrics
    
    def sample_batch(self, vanilla_batch_size:int , prompted_batch_sizes: dict, device, max_tokens=1024):

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

    def sample_batch_configurable(
            self, batch_size: int, 
            device: str, 
            prompt_proportions: list, 
            prompt_types: list, 
            max_tokens: int = 1024, 
            share_prompt_episodes=True,
            ep_ids = None
        ):
        # Samples a batch of episodes, where each episode has maximum of max_tokens tokens
        # This will return a list of dictionaries, where each dicionary contains variable length tensors,
        # This is in constrast to returning single tensors which contain all episodes with padding

        # Maximum number of timesteps we can fit in context
        num_timesteps = max_tokens // self.tokens_per_timestep

        # sample episode indices
        episode_indices = np.random.choice(self.dataset._episode_indices, size=batch_size, replace=False)
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
            if type(self.env.observation_space) == gym.spaces.Box:
                observations = torch.tensor(observations, dtype=torch.float32, device=device)
            elif type(self.env.observation_space) == gym.spaces.Discrete:
                observations = torch.tensor(observations, dtype=torch.int32, device=device)
            
            # apply image transforms
            if self.image_transform is not None:
                observations = self.image_transform.transform(observations)
            
            # convert actions to tensors
            if type(self.env.action_space) == gym.spaces.Box:
                actions = torch.tensor(actions, dtype=torch.float32, device=device)
            elif type(self.env.action_space) == gym.spaces.Discrete:
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
        with h5py.File(self.dataset.spec.data_path, 'r') as f:
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

    # Extension of default Minari sample_episodes where custom episode_indices can be passed
    def sample_episodes(self, n_episodes: int, episode_indices: list = None):
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): number of episodes to sample.
        """
        if episode_indices is None:
            episode_indices = self.dataset._episode_indices
        indices = self.dataset._generator.choice(
            episode_indices, size=n_episodes, replace=False
        )
        episodes = self.dataset._data.get_episodes(indices)
        return list(map(lambda data: EpisodeData(**data), episodes))

        


class ControlImageTransform:
    def __init__(self, env, patch_size=16):
        self.env = env
        self.patch_size = patch_size

        assert type(self.env.observation_space) == gym.spaces.Box, 'Only supports Box observation space'
        assert len(self.env.observation_space.shape) == 3 or len(self.env.observation_space.shape) == 2, 'Only supports 2D or 3D observation space'

        self.channel_first = None
        self.grayscale = False

        # Check if grayscale or RGB
        if len(self.env.observation_space.shape) == 3:
            # Check if channel first or channel last
            assert self.env.observation_space.shape[0] == 3 or self.env.observation_space.shape[-1] == 3, '3 channel first or channel last'
            self.channel_first = self.env.observation_space.shape[0] == 3
            if self.channel_first:
                self.height = self.env.observation_space.shape[1]
                self.width = self.env.observation_space.shape[2]
            else:
                self.height = self.env.observation_space.shape[0]
                self.width = self.env.observation_space.shape[1]
        else:
            self.grayscale = True
            self.height = self.env.observation_space.shape[0]
            self.width = self.env.observation_space.shape[1]

        # check how much padding is needed
        self.padding_h = 0
        self.padding_w = 0
        if self.height % self.patch_size != 0:
            self.padding_h = self.patch_size - (self.height % self.patch_size)
        if self.width % self.patch_size != 0:
            self.padding_w = self.patch_size - (self.width % self.patch_size)

    def transform(self, images: torch.Tensor):
        if self.grayscale:
            images = images.reshape(-1, 1, self.height, self.width)
            images = images.repeat(1, 3, 1, 1)
        else:
            if not self.channel_first:
                images = images.permute(0, 3, 1, 2)
        # all images now B X 3 X H X W, add padding:
        images = torch.nn.functional.pad(images, (0, self.padding_w, 0, self.padding_h), value=0) # left, right, top, bottom padding
        return images