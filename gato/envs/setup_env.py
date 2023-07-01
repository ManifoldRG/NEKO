import minari
import gymnasium as gym

from gato.envs.atari import load_atari_env

custom_env_loaders = {
    'ALE/': load_atari_env
}


def load_envs(dataset_names: list, load_kwargs: dict = {}):
    envs = []
    datasets = []
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