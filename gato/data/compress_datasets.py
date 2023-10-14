import minari
import os
import h5py
import argparse
import multiprocessing

# get Minari dataset path from environment variable MINARI_DATASETS_PATH
minari_path = os.environ['MINARI_DATASETS_PATH']

def compress_dataset(name, compress_lvl):
    print(f'Compressing {name}')
    dataset_path = os.path.join(minari_path, name, 'data', 'main_data.hdf5')
    compressed_name = name + '-compressed'
    compressed_folder = os.path.join(minari_path, compressed_name, 'data')
    os.makedirs(compressed_folder, exist_ok=True)
    compressed_path = os.path.join(compressed_folder, 'main_data.hdf5')

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
                        compression='gzip',
                        compression_opts=compress_lvl
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress_lvl', type=int, default=1)
    args = parser.parse_args()

    local_datasets = list(minari.list_local_datasets().keys())
    print(local_datasets)

    # Create a pool of worker processes
    with multiprocessing.Pool() as pool:
        pool.starmap(compress_dataset, [(dataset_name, args.compress_lvl) for dataset_name in local_datasets if 'compressed' not in dataset_name])
