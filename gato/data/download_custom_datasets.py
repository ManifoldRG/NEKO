import os
import gdown
import logging

logger = logging.getLogger(__name__)

datasets = {
    'd4rl_halfcheetah-expert-v2': 'https://drive.google.com/drive/folders/1YcUMTS7cMrUP8KJ6aQL87D9uYnrvGT02?usp=drive_link',
    'd4rl_hopper-expert-v2': 'https://drive.google.com/drive/folders/1upUt_aCRc3MCWhfVwpDlnW7YoVFEHre9?usp=drive_link',
    'd4rl_walker2d-expert-v2': 'https://drive.google.com/drive/folders/1ncu2DEhADWQBH6EeU_SrywQm8ETMM15M?usp=drive_link',
     #'Breakout-top1-s1-v0': 'https://drive.google.com/drive/folders/1Elos7A-NbpDzr5bPpPmoM-_2qY_68KFi?usp=drive_link' 
}

if __name__ == '__main__':
    minari_dir = os.path.join(os.path.expanduser('~'), '.minari')
    # create diretories if they do  not exist
    if not os.path.exists(minari_dir):
        os.mkdir(minari_dir)
    datasets_dir = os.path.join(minari_dir, 'datasets')
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)

    # download datasets, if they do not exist already
    for dataset_name, url in datasets.items():
        target_path = os.path.join(datasets_dir, dataset_name)
        if os.path.exists(target_path):
            logger.info(f'{dataset_name} already exists at {target_path}, skipping')
            continue
        gdown.download_folder(url=url, output=target_path, quiet=False, use_cookies=False)
