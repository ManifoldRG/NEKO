# Assume all datasets are downloaded and available from local directories
from gato.tasks.task import Task

import logging
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
from transformers import AutoTokenizer, GPT2Tokenizer

logger = logging.getLogger(__name__)

class CaptionTask(Task): 
    def __init__(self, tokenizer_model:str, caption_dataset, train_data, test_data = [],
                 test_data_prop = 0.1, test_data_mask_file = None):
        """
        caption_dataset is the directory for all of the data (training and test)
        train_data and test_data are list of sub_diretories under caption_dataset, with each directory containing 
        one training or test dataset which is composed of multiple .tar files downloaded with img2dataset. 
        Each tar file contains multiple bundles, with each bundle containing one .jpg, one txt and one json file. 
        The .jpg and the txt file (the caption) are extracted and placed into the data structures to be used for training and evaluation.
        
        The following two parameters are used when only train data is provided, in that case we need to split it 
        into training and test dataset: 
        test_data_prop is a percentage of data for test data, and 1-test_data_prop is the percentage of training data.
        test_data_mask_file is a file containing a mask for the indices of test data in the dataset processed 
        during training phase when the dataset is separated into a training set and and a test set 
        And this mask keeps track of the indices of test data in the dataset.  
        """
        super().__init__()

        if not caption_dataset.endswith('/'):
            caption_dataset = caption_dataset + '/'

        assert len(train_data) > 0, "Must provide train datasets for caption task" 
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.dataset = {}
        
        if len(test_data) > 0: # Note: len(train_data_directories)>0 also holds due to the abpve-mentioned assert
            self.dataset['train'] = self.process_data(caption_dataset, train_data)
            self.dataset['test'] = self.process_data(caption_dataset, test_data)
        else: # no test dataset is provided, need to aplit the training dataset 
            all_data = self.process_data(caption_dataset, train_data)
            #test_data_mask_file is a .json file used to construct test data when evaluation of model is run separately, i.e. not run inside training loop
            if test_data_mask_file is not None: 
                with open(test_data_mask_file, 'r') as f:
                    test_data_mask = json.load(f) # this should be a list on int
                    assert len(test_data_mask) == len(all_data), "len(test_data_mask) must be equal to len(all_data)" 
                    self.dataset['test'] = [item for item, mask in zip(all_data, test_data_mask) if mask == 1]
            else: # this is when model training is run, need to split data into training and test set
                test_data_len = math.ceil(len(all_data)*test_data_prop)
                test_data_mask = [0]*len(all_data)
                test_data_indices = [random.randint(0, len(all_data)-1) for _ in range(test_data_len)]
                for index in test_data_indices:
                    test_data_mask[index] = 1 # set the mask of test data item to 1, then training data are the element with mask 0
                self.dataset['train'] = [item for item, mask in zip(all_data, test_data_mask) if mask == 0]
                self.dataset['test'] = [item for item, mask in zip(all_data, test_data_mask) if mask == 1]

                with open('test_data_mask.json', 'w') as f:
                    json.dump(test_data_mask, f)
    
    def process_data(self, caption_dataset, data_directories):
        dataset = []
        for directory in data_directories:
            tar_files = []
            data_directory = caption_dataset + directory
            for file in os.listdir(data_directory):
                if fnmatch.fnmatch(file, '*.tar'):
                    tar_files.append(os.path.join(data_directory, file))
        
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
            dataset.append(item)
        return dataset       
    
    def sample_batch(self, batch_size):
        random_indices = [random.randint(0, len(self.dataset['train'])-1) for _ in range(batch_size)]
        selected_examples = [self.dataset['train'][idx] for idx in random_indices]
        
        batch_dicts = []
        for item in selected_examples:
            batch_dict = {
                'images': item['image'],
                'text':self.text_tokenizer.encode(item['text'])
            }
            batch_dicts.append(batch_dict)

        return batch_dicts

    def evaluate(self, model, num_examples_to_test=50, deterministic=True, log_examples_to_output=False):    
        tokenizer = model.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        if num_examples_to_test > len(self.dataset['test']):
            logger.info(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.dataset['test'])

        if log_examples_to_output:
            logger.info(f'--- examples ---')

        random_indices = [random.randint(0, len(self.dataset['test'])-1) for _ in range(num_examples_to_test)]
        selected_examples = [self.dataset['test'][idx] for idx in random_indices]

        for idx in range(num_examples_to_test):
            image = selected_examples[idx]['image']
            target_caption = selected_examples[idx]['text']
            target_tokens = tokenizer.encode(target_caption)

            # Generate prediction
            pred_logits, pred_caption = model.predict_caption(image, max_length = len(target_tokens),deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                logger.info(f'Target caption: {target_caption} \n Predicted caption : {pred_caption}')
                logger.info("----")

            # Calculate loss
            loss = loss_fn(pred_logits, torch.tensor(target_tokens).to(model.device))
            total_loss += loss.item()
            total_tokens += len(target_tokens)
        if log_examples_to_output:
            logger.info(f'--- examples end ---')

        avg_loss = total_loss / num_examples_to_test
        perplexity = torch.exp(torch.tensor(avg_loss))

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
        return metrics

# test code
if __name__ == '__main__':
    # replace the following directory with your data directory
    task = CaptionTask(tokenizer_model = 'gpt2', caption_dataset = '/home/<user name>/Git/NEKO/Caption_data', 
                       train_data = ['train'], test_data = ['test'], test_data_prop = 0.1)

    #logger.info(task.dataset["train"][4]["images"][0][1][10])
    #logger.info(task.dataset["train"][4]["images"][0][2][15])
    #logger.info(task.dataset["train"][4]["text"])
    batch = task.sample_batch(5)
    #logger.info(batch)
    logger.info(type(batch))
    logger.info(batch[0]['images'][0][1][10])
    logger.info(batch[0]['images'][0][2][15])
    logger.info(batch[0]['images'].shape)
    logger.info(batch[0]['text'])
