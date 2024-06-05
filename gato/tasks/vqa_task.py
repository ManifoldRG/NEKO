# Assume all datasets are downloaded and available from local directories
from gato.tasks.task import Task
from gato.policy.embeddings import Pretrained_ImageEmbedding
import os
from PIL import Image

import numpy as np
import math
import torch
from torch.nn import functional as F
from torch import nn
import json
import random
from transformers import AutoTokenizer, GPT2Tokenizer

class VqaTask(Task): 
    def __init__(self, tokenizer_model:str,
                 vqa_dataset, train_data, test_data, 
                 train_img_name_prefix, train_img_file_name_len, 
                 test_img_name_prefix, test_img_file_name_len,
                 questions_file, annotations_file):
        """
        vqa_dataset is the directory where the data for vqa task is located, should end with "/" 
        train_data and test_data are each a list of sub directories where training data and test data are located
        ***_img_name_prefix is a list, each item of the list holds the image file name prefix for each sub directory
        ***_img_file_name_len is a list, each item of the list holds the length of image file name for each sub directory
        ***_questions_file is a .json file name for for the file under each sub directory containining questions and images IDs
        ***_annotations_file is a .json file name for the file under each sub directory containing images IDs, quesitons and answers
        For the current implementaiton, it is required that these files are named "questions.json" and "annotations.json" under each sub diredctory
        Each sub directory should also contain the image files associated with the corresponding question and annotation files
        """
        super().__init__()

        assert len(train_data) == len(test_data), "Number of training data and test data sub director must be equal to each other" 

        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.dataset = {}
        self.image_embedding = Pretrained_ImageEmbedding()

        if not vqa_dataset.endswith('/'):
            vqa_dataset = vqa_dataset + '/'

        self.dataset['train'] = self.process_data(vqa_dataset, train_data, train_img_name_prefix, train_img_file_name_len, questions_file, annotations_file)
        self.dataset['test'] = self.process_data(vqa_dataset, test_data, test_img_name_prefix, test_img_file_name_len, questions_file, annotations_file)

    def process_data(self, vqa_dataset, data_directories, img_name_prefix, img_file_name_len, questions_file, annotations_file):
        dataset = []
        item = {}

        dir_idx = 0
        for directory in data_directories:
            data_directory = vqa_dataset + directory
            if not data_directory.endswith('/'):
                data_directory = data_directory + '/'

            with open(data_directory + annotations_file, 'r') as json_file:
                # This is a list, each item contains an image ID, a question ID, and its corresponding answers (multiple answers to one question)
                annotations = json.load(json_file)['annotations'] 

            with open(data_directory + questions_file, 'r') as json_file:
                questions = json.load(json_file)['questions'] # This is a list of question IDs and the corresponding questions

            assert len(annotations) == len(questions), "Number of annotations must be equal to number of questions" 
        
            for idx in range(len(questions)):
                assert annotations[idx]['image_id'] == questions[idx]['image_id'] and annotations[idx]['question_id'] == questions[idx]['question_id']
                image_id = str(annotations[idx]['image_id'])
                img_file_name = img_name_prefix[dir_idx] + '0' * (img_file_name_len[dir_idx] - len(image_id) - len(img_name_prefix[dir_idx])) + image_id + '.jpg'
                try:
                    # if the image file does not exist or some other error occurs, we simply discard this sample and move to next
                    img = Image.open(data_directory + img_file_name)
                except:
                    continue
                item['image'] = img
                item['question'] = questions[idx]['question']
                item['answers'] = annotations[idx]['answers']
                dataset.append(item)
            dir_idx = dir_idx + 1
        return dataset
    
    def sample_batch(self, batch_size):
        random_indices = [random.randint(0, len(self.dataset['train'])-1) for _ in range(batch_size)]
        selected_examples = [self.dataset['train'][idx] for idx in random_indices]

        batch_dicts = []
        for item in selected_examples:
            answer_idx = random.randint(0, len(item['answers'])-1) # randomly choose an answer out of the set of answers
            batch_dict = {
                # By default, image will be resized to 224*224, patch size is 16, image_embedding() will return embedding tensor
                # of the shape (1, 196, 768), where 1 is an added batch dimension, 196 is the numbe of patches, which is the 
                # result of 14 multiplied by 14, where 14 is the result of image size divided by patch size, i.e. 224/16=14
                'image_imbeddings': self.image_embedding(item['image']),
                # 'text' is to concat the question and a randomly chosen answer with a space in between
                'text': self.text_tokenizer.encode(item['question'] + ' ' + item['answers'][answer_idx]['answer']) 
            }
            batch_dicts.append(batch_dict)
        return batch_dicts

    def evaluate(self, model, num_examples_to_test=50, deterministic=True, log_examples_to_output=False):    
        tokenizer = model.module.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        if num_examples_to_test > len(self.dataset['test']):
            print(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.dataset['test'])

        if log_examples_to_output:
            print(f'--- examples ---')

        random_indices = [random.randint(0, len(self.dataset['test'])-1) for _ in range(num_examples_to_test)]
        selected_examples = [self.dataset['test'][idx] for idx in random_indices]

        for idx in range(num_examples_to_test):
            image = selected_examples[idx]['image']
            question = selected_examples[idx]['question']
            answer_idx = random.randint(0, len(selected_examples[idx]['answers'])-1) # randomly choose an answer out of the set of answers
            target_answer = selected_examples[idx]['answers'][answer_idx]['answer'] 
            target_tokens = tokenizer.encode(target_answer)

            # Generate prediction
            pred_logits, pred_answer = model.predict_answer(self.image_embedding(image).to(model.device), question, max_length = len(target_tokens),deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                print(f'Target answer: {target_answer} \n Predicted answer : {pred_answer}')
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
    # replace the following directories and files names with your directories and files
    task = VqaTask(vqa_dataset              = '/home/<user name>/Git/NEKO/VQA_Data/',
                   train_data               = ['train2014'], 
                   test_data                = ['val2014'],
                   train_img_name_prefix    = ['COCO_train2014_'], 
                   train_img_file_name_len  = [27], 
                   test_img_name_prefix     = ['COCO_val2014_'], 
                   test_img_file_name_len   = [25],
                   questions_file           = 'questions.json', 
                   annotations_file         = 'annotations.json'
                   )

    batch = task.sample_batch(5)
    print(type(batch))
    print(list(batch[0].keys()))
    print(batch[0]['images'][0][1][10])
    print(batch[0]['images'][0][2][15])
    print(batch[0]['images'].shape)
    print(batch[0]['text'])

