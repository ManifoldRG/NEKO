import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward 
import numpy as np

def load_atari_env(env_name: str, load_kwargs: dict):
    assert 'v5' in env_name

    repeat_action_probability = 0 # 0.25
    clip_rewards = False

    repeat_action_probability = load_kwargs.get('repeat_action_probability', repeat_action_probability)
    clip_rewards = load_kwargs.get('clip_rewards', clip_rewards)
    render_mode = load_kwargs.get('render_mode', None)

    env = gym.make(env_name, frameskip=1, repeat_action_probability=repeat_action_probability, render_mode=render_mode) # e.g. 'ALE/Breakout-v5'
    env = AtariPreprocessing(env, frame_skip=4, noop_max=0)
    if clip_rewards:
        env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))
    return env

ALL_GAMES = [
    'Alien',
    'Amidar',
    'Assault',
    'Asterix',
    'Atlantis',
    'BankHeist',
    'BattleZone',
    'BeamRider',
    'Boxing',
    'Breakout',
    'Carnival',  
    'Centipede',
    'ChopperCommand',
    'CrazyClimber',
    'DemonAttack',
    'DoubleDunk',
    'Enduro',
    'FishingDerby',
    'Freeway',
    'Frostbite',
    'Gopher',
    'Gravitar',
    'Hero',
    'IceHockey',
    'Jamesbond',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'MsPacman',
    'NameThisGame',
    'Phoenix',
    'Pong',
    'Pooyan',  
    'Qbert',
    'Riverraid',
    'Robotank',
    'Seaquest',
    'SpaceInvaders',
    'StarGunner',
    'TimePilot',
    'UpNDown',
    'VideoPinball',
    'WizardOfWor',
    'YarsRevenge',
    'Zaxxon'
]

# Test games used in Scaled-QL
TEST_GAMES = [
    'Alien',
    'MsPacman',
    'Pong',
    'SpaceInvaders',
    'StarGunner'
]

TRAIN_GAMES = [game for game in ALL_GAMES if game not in TEST_GAMES]

assert len(TRAIN_GAMES) == 40
assert len(TEST_GAMES) == 5
assert len(ALL_GAMES) == 45
