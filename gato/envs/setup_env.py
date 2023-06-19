import minari
#from gato.envs.atari import load_atari_env
from atari import load_atari_env

custom_env_loaders = {
    'ALE/': load_atari_env
}


def load_env_dataset(dataset_name: str, load_kwargs: dict = {}):
    # load dataset
    dataset = minari.load_dataset(dataset_name)

    # recover environment
    env = dataset.recover_environment()
    env_name = env.unwrapped.spec.id

    # rebuild environment if custom loader specified
    for prefix, loader in custom_env_loaders.items():
        if prefix in env_name:
            env = loader(env_name, load_kwargs)
            break
    
    return env, dataset



if __name__ == '__main__':
    # load MuJoCo locomotion dataset, env
    mujoco_env, mujoco_dataset = load_env_dataset('d4rl_halfcheetah-expert-v2')

    # load atari
    atari_env, atari_dataset = load_env_dataset('Breakout-expert_s0-v0')