import minari
import gymnasium as gym

from gato.envs.atari import load_atari_env, TRAIN_GAMES as ATARI_TRAIN, TEST_GAMES as ATARI_TEST

custom_env_loaders = {
    'ALE/': load_atari_env
}

minari_str = '{}-top1-s1-v0'
custom_key_words = {
    'TOP1_ATARI_TRAIN': [minari_str.format(game_name) for game_name in ATARI_TRAIN],
    'TOP1_ATARI_TEST': [minari_str.format(game_name) for game_name in ATARI_TEST]
}


def load_envs(dataset_names: list, load_kwargs: dict = {}):
    envs = []
    datasets = []

    new_dataset_names = []

    # scan for custom keywords
    for dataset_name in dataset_names:
        if dataset_name in custom_key_words:
            new_dataset_names.extend(custom_key_words[dataset_name])
        else:
            new_dataset_names.append(dataset_name)
    dataset_names = new_dataset_names

    for dataset_name in dataset_names:
        env, dataset = load_env_dataset(dataset_name, load_kwargs)
        envs.append(env)
        datasets.append(dataset)
    return envs, datasets


def load_env_dataset(dataset_name: str, load_kwargs: dict = {}):
    # load dataset
    dataset = minari.load_dataset(dataset_name)

    env_name = dataset._data.env_spec.id
    env = None

    # custom environment build if custom loader specified
    for prefix, loader in custom_env_loaders.items():
        if prefix in env_name:
            env = loader(env_name, load_kwargs)
            break

    # Default to recovering dataset from Minari
    if env is None:
        env = gym.make(dataset._data.env_spec, **load_kwargs)

    
    return env, dataset



if __name__ == '__main__':
    # load MuJoCo locomotion dataset, env
    mujoco_env, mujoco_dataset = load_env_dataset('d4rl_halfcheetah-expert-v2')

    # load atari
    atari_env, atari_dataset = load_env_dataset('Breakout-expert_s0-v0')

    import pdb; pdb.set_trace()
