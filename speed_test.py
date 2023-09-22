import minari
import os
import h5py
import numpy as np
import concurrent.futures
from multiprocessing import Manager



dataset_dir = '/home/daniel/.minari/datasets/'
dataset_name= 'Pong-top1-s1-v0'
path = os.path.join(dataset_dir, dataset_name, 'data')
dataset_path = os.path.join(path, 'main_data.hdf5')
compressed_name = dataset_name + '-compressed'
compressed_folder = os.path.join(dataset_dir, compressed_name, 'data')
os.makedirs(compressed_folder, exist_ok=True)
compressed_path = os.path.join(compressed_folder, 'main_data.hdf5')


def benchmark_reads(file_path, episodes=None, n_episodes=10):
    if episodes is None:
        episodes = np.random.randint(0,200,size=n_episodes)
    with h5py.File(file_path, 'r') as h5_f:
        #for i in range(n_episodes):
        for i in episodes:
            observations = h5_f[f'episode_{i}']['observations'][50:70,:,:]
            #print(observations.shape)


def read_episode(h5_f, episode):
    observations = h5_f[f'episode_{episode}']['observations'][50:70,:,:]

def benchmark_read_thread(file_path, episodes=None, n_episodes=10):
    if episodes is None:
        episodes = np.random.randint(0,200,size=n_episodes)
    with h5py.File(file_path, 'r') as h5_f:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda episode: read_episode(h5_f, episode), episodes)

def read_episode_multi(file_path, episode, shared_list):
    with h5py.File(file_path, 'r') as h5_f:
        observations = h5_f[f'episode_{episode}']['observations'][50:70,:,:]
        #shared_list.append(observations)
    #return observations
def benchmark_reads_multiprocessing(file_path, episodes=None, n_episodes=10):
    if episodes is None:
        episodes = np.random.randint(0,200,size=n_episodes)
    with Manager() as manager:
        shared_list = manager.list()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            return executor.map(read_episode_multi, [file_path] * len(episodes), episodes, [shared_list] * len(episodes))


@profile
def main():
    # old test
    n_episodes = 64
    episodes = np.random.randint(0,200,size=n_episodes)
    #episodes = np.arange(n_episodes)
    benchmark_reads(dataset_path, episodes)
    benchmark_reads(compressed_path, episodes)
    benchmark_read_thread(dataset_path, episodes)
    benchmark_read_thread(compressed_path, episodes)
    output = list(benchmark_reads_multiprocessing(dataset_path, episodes))
    del output
    benchmark_reads_multiprocessing(compressed_path, episodes)

    # lets try now using dataset
    dataset = minari.load_dataset(dataset_name)

    dataset._data.get_episodes(episodes)
    # # iterate over all episodes
    # eps = []
    # iterations = 10
    # for i in range(iterations):
    #     for episode in dataset:
    #         eps.append(episode)
    # # compute mean len
    # mean_len = np.mean([episode.total_timesteps for episode in eps])

if __name__ == '__main__':
    main()