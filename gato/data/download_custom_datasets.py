import os
import gdown
import argparse
mujoco_datasets = {
    'd4rl_halfcheetah-expert-v2': 'https://drive.google.com/drive/folders/1YcUMTS7cMrUP8KJ6aQL87D9uYnrvGT02?usp=drive_link',
    'd4rl_hopper-expert-v2': 'https://drive.google.com/drive/folders/1upUt_aCRc3MCWhfVwpDlnW7YoVFEHre9?usp=drive_link',
    'd4rl_walker2d-expert-v2': 'https://drive.google.com/drive/folders/1ncu2DEhADWQBH6EeU_SrywQm8ETMM15M?usp=drive_link',
     #'Breakout-top1-s1-v0': 'https://drive.google.com/drive/folders/1Elos7A-NbpDzr5bPpPmoM-_2qY_68KFi?usp=drive_link' 
}

atari_top_1 = 'https://drive.google.com/uc?id=188H5MY76De0qd4l0cMbdIiubLa5Faql5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_group', type=str, default='MuJoCo', choices=['MuJoCo', 'TOP1_ATARI_TEST'])
    args = parser.parse_args()

    minari_dir = os.path.join(os.path.expanduser('~'), '.minari')
    # create diretories if they do  not exist
    if not os.path.exists(minari_dir):
        os.mkdir(minari_dir)
    datasets_dir = os.path.join(minari_dir, 'datasets')
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)

    # download datasets, if they do not exist already
    if args.download_group == 'TOP1_ATARI_TEST':
        # gdown.download_folder(url=atari_top_1, output=datasets_dir, quiet=False, use_cookies=False)
        gdown.download(url=atari_top_1, output=os.path.join(datasets_dir, 'TOP1_ATARI_TEST.zip'), quiet=False, use_cookies=False)
        gdown.download
        #unzip the file
        os.system(f'unzip {os.path.join(datasets_dir, "TOP1_ATARI_TEST.zip")} -d {datasets_dir}')
    else:
        for dataset_name, url in mujoco_datasets.items():
            target_path = os.path.join(datasets_dir, dataset_name)
            if os.path.exists(target_path):
                print(f'{dataset_name} already exists at {target_path}, skipping')
                continue
            gdown.download_folder(url=url, output=target_path, quiet=False, use_cookies=False)
