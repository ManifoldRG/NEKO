from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import asyncio

import gymnasium as gym
import numpy as np
import torch
import minari
from minari.dataset.minari_dataset import EpisodeData
import h5py

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

def _fetch_episode(args):
    data_path, ep_idx, decode_space, observation_space, action_space = args
    with h5py.File(data_path, "r") as file:
        ep_group = file[f"episode_{ep_idx}"]
        return {
            "id": ep_group.attrs.get("id"),
            "total_timesteps": ep_group.attrs.get("total_steps"),
            "seed": ep_group.attrs.get("seed"),
            # "observations": ep_group["observations"][()],
            # "actions": ep_group["actions"][()],
            "observations": decode_space(
                ep_group["observations"], observation_space
            ),
            "actions": decode_space(
                ep_group["actions"], action_space
            ),

            "rewards": ep_group["rewards"][()],
            "terminations": ep_group["terminations"][()],
            "truncations": ep_group["truncations"][()],
        }

   
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

    
    def evaluate(self, model, n_iterations, deterministic=True, promptless_eval=False):
        # serial evaluation
        returns = []
        clipped_returns = []
        ep_lens = []
        metrics = {}

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

        # List of numpy arrays for each episode 
        episodes_data = {
            'actions': [],
            'observations': [],
        }

        # Filter dataset if filter function is provided
        all_episodes = self.sample_episodes(n_episodes=batch_size, episode_indices=ep_ids)

        if share_prompt_episodes:
            main_episodes = all_episodes
            prompt_episodes = all_episodes
        else:
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
            ep_len = episode.total_timesteps
            
            if timesteps_for_main >= ep_len:
                # sample entire episode
                start = 0
                end = ep_len - 1
            else:
                # sample which timestep to start with
                start = np.random.randint(0, ep_len - timesteps_for_main)
                end = start + timesteps_for_main
            observations = episode.observations[start:end,]
            actions = episode.actions[start:end,]

            episodes_data['observations'].append(observations)
            episodes_data['actions'].append(actions)

        # add prompt
        for i, episode in enumerate(prompt_episodes):
            ep_len = episode.total_timesteps
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
        # if self.args.parallel_sampling:
        #     # custom parallel
        #     episodes = self.get_episodes(self.dataset._data, indices)
        # else:
        #     # Native minari get_episodes
        #     episodes = self.dataset._data.get_episodes(indices)
        episodes = self.dataset._data.get_episodes(indices)
        return list(map(lambda data: EpisodeData(**data), episodes))

    def get_episodes(self, _data, episode_indices):
        """Get a list of episodes.

        Args:
            _data (MinariData._data): MinariStorage object
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            episodes (List[dict]): list of episodes data
        """
        #out = []
        # def fetch_episode(ep_idx):
        #     with h5py.File(_data._data_path, "r") as file:
        #         ep_group = file[f"episode_{ep_idx}"]
        #         return {
        #             "id": ep_group.attrs.get("id"),
        #             "total_timesteps": ep_group.attrs.get("total_steps"),
        #             "seed": ep_group.attrs.get("seed"),
        #             "observations": _data._decode_space(
        #                 ep_group["observations"], _data.observation_space
        #             ),
        #             "actions": _data._decode_space(
        #                 ep_group["actions"], _data.action_space
        #             ),
        #             "rewards": ep_group["rewards"][()],
        #             "terminations": ep_group["terminations"][()],
        #             "truncations": ep_group["truncations"][()],
        #         }
        # with ThreadPoolExecutor() as executor:
        #     out = list(executor.map(fetch_episode, episode_indices))
        # return out
        slice_size = 500
        def read_slice(ep_idx, slice_idx, slice_size):
            with h5py.File(_data._data_path, 'r') as file:
                ep_group = file[f"episode_{ep_idx}"] 
                #TODO, make custom _decode_space instead of slicing directly for other oibs spaces
                slice_data = ep_group["observations"][slice_idx * slice_size:(slice_idx + 1) * slice_size]
            return slice_data


        out = []
        with h5py.File(_data._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                out.append(
                    {
                        "id": ep_group.attrs.get("id"),
                        "total_timesteps": ep_group.attrs.get("total_steps"),
                        "seed": ep_group.attrs.get("seed"),
                        # "observations": _data._decode_space(
                        #     egg, _data.observation_space
                        # ),
                        "actions": _data._decode_space(
                            ep_group["actions"], _data.action_space
                        ),
                        "rewards": ep_group["rewards"][()],
                        "terminations": ep_group["terminations"][()],
                        "truncations": ep_group["truncations"][()],
                    }
                )
                num_slices = (out[-1]['total_timesteps'] + 1 + slice_size - 1) // slice_size
                with ThreadPoolExecutor() as executor:
                    data_slices = list(executor.map(read_slice, repeat(ep_idx), range(num_slices), repeat(slice_size)))

                # Now, concatenate data_slices to get the full dataset
                observations = np.concatenate(data_slices, axis=0)
                out[-1]['observations'] = observations

        return out
    # def get_episodes(self, _data, episode_indices):
    #     """Get a list of episodes."""
    #     args = [
    #         (_data._data_path, ep_idx.item(), _data._decode_space, _data.observation_space, _data.action_space)
    #         for ep_idx in episode_indices
    #     ]
        
    #     with ProcessPoolExecutor() as executor:
    #         out = list(executor.map(_fetch_episode, args))
        
    #     # import pdb; pdb.set_trace()
    #     # Decoding after multiprocessing pool is complete
    #     # for i, ep_data in enumerate(out):
    #     #     ep_data["observations"] = _data._decode_space(ep_data["observations"], _data.observation_space)
    #     #     ep_data["actions"] = _data._decode_space(ep_data["actions"], _data.action_space)
        
    #     return out
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