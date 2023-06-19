import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward 
import numpy as np


def load_atari_env(env_name: str, load_kwargs: dict):
    assert 'v5' in env_name

    repeat_action_probability = 0 # 0.25
    clip_rewards = True

    repeat_action_probability = load_kwargs.get('repeat_action_probability', repeat_action_probability)
    clip_rewards = load_kwargs.get('clip_rewards', clip_rewards)

    env = gym.make(env_name, frameskip=1, repeat_action_probability=repeat_action_probability) # e.g. 'ALE/Breakout-v5'
    env = AtariPreprocessing(env, frame_skip=4, noop_max=0)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))
    return env
