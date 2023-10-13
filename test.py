# Minari testing
import minari
import os
import h5py

dataset_dir = '/home/daniel/.minari/datasets/'
dataset_name= 'Pong-top1-s1-v0'
path = os.path.join(dataset_dir, dataset_name, 'data')
dataset_path = os.path.join(path, 'main_data.hdf5')
compressed_name = dataset_name + '-compressed'
compressed_folder = os.path.join(dataset_dir, compressed_name, 'data')
os.makedirs(compressed_folder, exist_ok=True)
compressed_path = os.path.join(compressed_folder, 'main_data.hdf5')

# test loading original dataset
dataset = minari.load_dataset(dataset_name)
del dataset

# Open the existing file
with h5py.File(dataset_path, 'r') as original_file:
    
    # Create a new file
    with h5py.File(compressed_path, 'w') as compressed_file:
        
        # Iterate over all groups/datasets in the original file
        def copy_and_compress(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Create a compressed dataset in the new file
                compressed_file.create_dataset(
                    name,
                    data=obj[()],
                    compression='gzip',  # specify the compression type here
                    #compression_opts=9   # specify the compression level here
                    compression_opts=1
                )
                # Copy attributes of a group/dataset
                for attr_name, attr_value in obj.attrs.items():
                    compressed_file[name].attrs.create(attr_name, attr_value)

            elif isinstance(obj, h5py.Group):
                # Create a group in the new file
                compressed_group = compressed_file.create_group(name)
                
                # Copy attributes of the group
                for attr_name, attr_value in obj.attrs.items():
                    compressed_group.attrs.create(attr_name, attr_value)

        
        # Visit all items in the original file and apply the copy_and_compress function
        original_file.visititems(copy_and_compress)

# test loading compressed dataset
# load invidually
f = h5py.File(compressed_path, "r")
f.close()

# Initialize a list to hold the names of the groups
group_names = []

# Define a function to be applied to each item in the file
def find_groups(name, item):
    if isinstance(item, h5py.Group):
        group_names.append(name)

# Use the visititems method to apply the find_groups function to each item in the file

# let's benchmark loading episodes 
['episode_100']

import numpy as np
def benchmark_reads(file_path, episodes=None, n_episodes=10):
    if episodes is None:
        episodes = np.random.randint(0,200,size=n_episodes)
    with h5py.File(file_path, 'r') as h5_f:
        #for i in range(n_episodes):
        for i in episodes:
            observations = h5_f[f'episode_{i}']['observations'][()]
            print(observations.shape)

# old test
n_episodes = 10
episodes = np.random.randint(0,200,size=n_episodes)
benchmark_reads(dataset_path, episodes)
benchmark_reads(compressed_path, episodes)
import pdb; pdb.set_trace()

# lets try now using dataset
dataset = minari.load_dataset(dataset_name)

dataset._data.get_episodes(episodes)