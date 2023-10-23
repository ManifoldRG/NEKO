# Assume all datasets are downloaded and available from local directories
from gato.tasks.task import Task, TaskTypeEnum

import os
import tarfile
import webdataset as wds
import fnmatch
from torch.utils.data import DataLoader
from PIL import Image
import io # need to use BytesIO

import numpy as np
import math
import torch
from torch.nn import functional as F
from torch import nn
import json
import random

class CaptionTask(Task): 
    def __init__(self, task_type: TaskTypeEnum, caption_datasets, test_data_prop = 0.1, test_data_mask_file = None):
        """
        task_type should be CAPTION
        caption_datasets is a list of diretories, with each directory in the format of something similar to 
        /home/<user_name>/NEKO/Dataset01... etc.), and contains one image-caption dataset which is composed of 
        multiple .tar files downloaded with img2dataset. Each tar file contains multiple bundles, with each bundle 
        containing one .jpg, one txt and one json file. The .jpg and the txt file (the caption) are extracted and 
        placed into the data structures to be used for training and evaluation. 
        test_data_prop is a percentage of data for test data, and 1-test_data_prop is the percentage of training data.
        test_data_mask_file is a file containing a mask for the indices of test data in the dataset processed 
        during training phase when the dataset is separated into a training set and and a test set 
        And this mask keeps track of the indices of test data in the dataset.  
        """
        super().__init__(task_type)
        self.dataset = {}
        all_data = []
        for directory in caption_datasets:
            tar_files = []
            for file in os.listdir(directory):
                if fnmatch.fnmatch(file, '*.tar'):
                    tar_files.append(os.path.join(directory, file))
        
        # https://github.com/webdataset/webdataset#dataloader: WebDataset is just an instance of a standard IterableDataset
        # In the following, data from multiple tar files are combined into one WebDataset, and then wrapped into a DalaLoader
        data_loader = DataLoader(wds.WebDataset(tar_files)) 

        # Iterate through all of the bundles to extract jpg and txt (caption) and place them into the desiganted data structure
        for idx, bundle in enumerate(data_loader):
            item = {}
            img = Image.open(io.BytesIO(bundle['jpg'][0])) # bundle['jpg'] is a list of length 1
            img_data = np.asarray(img)
            # Through testing of processing multiple .tar files, we have figured out that we need "try except" in the following 
            # because sometimes the img_data is only (256, 256) insetad of (256, 256,3) (assuming all image sizes are 256x256)
            # and the following transpose will raise an error and everything grinds to a halt. It is perhaps a bug in the "img2dataset" unitlity
            # used to downlaod datasets into tar files. When such error occurs, we just ignore the current bundle and move to the next one
            try:
                img_data = img_data.transpose(2, 0, 1) # reshape from (256, 256, 3) to (3, 256, 256)
            except:
                continue
    
            # Need to add a new dimension to (3, 256, 256) so it becomes (1, 3, 256, 256) where the added dummy dimension at dim 0 is the num_images. 
            # In this case, num_images is always 1. This is for the purpose of aligning the data structure with that in the model training
            item['image'] = torch.tensor(img_data[np.newaxis, :])
            item['text'] = bundle['txt'][0].decode('utf-8')
            all_data.append(item)

        #test_data_mask_file is a .json file used to construct test data when evaluation of model is run separately, i.e. not run inside training loop
        if test_data_mask_file is not None: 
            with open(test_data_mask_file, 'r') as f:
                test_data_mask = json.load(f) # this should be a list on int
                assert len(test_data_mask) == len(all_data), "len(test_data_mask) must be equal to len(all_data)" 
                self.dataset['test'] = [item for item, mask in zip(all_data, test_data_mask) if mask == 1]

        else: # this is when model training is run, need to separate data into training and test set
            test_data_len = math.ceil(len(all_data)*test_data_prop)
            test_data_mask = [0]*len(all_data)
            test_data_indices = [random.randint(0, len(all_data)-1) for _ in range(test_data_len)]
            for index in test_data_indices:
                test_data_mask[index] = 1 # set the mask of test data item to 1, then training data are the element with mask 0
            self.dataset['train'] = [item for item, mask in zip(all_data, test_data_mask) if mask == 0]
            self.dataset['test'] = [item for item, mask in zip(all_data, test_data_mask) if mask == 1]

            with open('test_data_mask.json', 'w') as f:
                json.dump(test_data_mask, f)

    def sample_batch(self, batch_size):
        random_indices = np.random.randint(0, len(self.dataset['train']), size=batch_size)
        random_indices = [i.item() for i in random_indices]
        selected_examples = [self.dataset['train'][idx] for idx in random_indices]
        
        batch_dicts = []
        for item in selected_examples:
            batch_dict = {
                'images': item['image'],
                'text': item['text']
            }
            batch_dicts.append(batch_dict)

        return batch_dicts

    # The following function reuses some code from the evaluate() function defined for text task. Eventually, it will be the 
    # best if the common code between the two evaluate() functions can be defined in one function and reused by both
    def evaluate(self, model, num_examples_to_test=100, deterministic=True, log_examples_to_output=False):    
        tokenizer = model.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        if num_examples_to_test > len(self.dataset['test']):
            print(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.dataset['test'])

        if log_examples_to_output:
            print(f'--- examples ---')
        for idx in range(num_examples_to_test):
            image = self.dataset['test'][idx]['image']
            target_caption = self.dataset['test'][idx]['text']
            target_tokens = tokenizer.encode(target_caption)

            # Generate prediction
            pred_logits, pred_caption = model.predict_caption(image, max_length = len(target_tokens),deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                print(f'Target caption: {target_caption} \n Predicted caption : {pred_caption}')
                print("----")

            # Calculate loss
            loss = loss_fn(pred_logits, torch.tensor(target_tokens).to(model.device))
            total_loss += loss.item()
            total_tokens += len(target_tokens)
        if log_examples_to_output:
            print(f'--- examples end ---')

        avg_loss = total_loss / num_examples_to_test
        perplexity = torch.exp(torch.tensor(avg_loss))

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
        return metrics

# test code
if __name__ == '__main__':
    # replace the following directory with youe data directory
    task = CaptionTask(task_type = 'caption', dataset_directories = ['/home/<user_name>/Git/NEKO/your_data_path'], test_data_prop = 0.1)
    #print(task.dataset["train"][4]["images"][0][1][10])
    #print(task.dataset["train"][4]["images"][0][2][15])
    #print(task.dataset["train"][4]["text"])
    batch = task.sample_batch(5)
    #print(batch)
    print(type(batch))
    print(batch[0]['image'][0][1][10])
    print(batch[0]['image'][0][2][15])
    print(batch[0]['image'].shape)
    print(batch[0]['text'])