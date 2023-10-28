# Assume all datasets are downloaded and available from local directories
from gato.tasks.task import Task, TaskTypeEnum

import os
from PIL import Image
import io # need to use BytesIO

import numpy as np
import math
import torch
from torch.nn import functional as F
from torch import nn
import json
import random

class VqaTask(Task): 
    def __init__(self, task_type: TaskTypeEnum, vqa_dataset,
                 train_imgs_directory, train_questions_file, train_annotations_file, 
                 test_imgs_directory, test_questions_file, test_annotations_file,
                 train_img_name_prefix = 'COCO_train2014_', train_img_file_name_len = 27, 
                 test_img_name_prefix = 'COCO_val2014_', test_img_file_name_len = 25):
        """
        task_type should be VQA
        vqa_dataset is the directory where the data for vqa task is located, whould end with "/" 
        ***_questions_file is a .json file under the vqa_dataset directory containining questions and images IDs
        ***_annotations_file is a .json file under the vqa_dataset directory containing images IDs, quesitons and answers 
        ***_imgs_directory are sub-diretories under vqa_dataset where the training or testing images are located, should end with "/"
        ***_img_name_prefix is the prefix for each image name file
        ***_img_file_name_len is the length of image file names
        """
        super().__init__(task_type)
        self.dataset = {}
        self.dataset['train'] = self.process_data(vqa_dataset, train_imgs_directory, train_questions_file, train_annotations_file, train_img_name_prefix, train_img_file_name_len)
        self.dataset['test'] = self.process_data(vqa_dataset, test_imgs_directory, test_questions_file, test_annotations_file, test_img_name_prefix, test_img_file_name_len)
        #print(f"------len(self.dataset['train']) = {len(self.dataset['train'])}")
        #print(f"------len(self.dataset['test']) = {len(self.dataset['test'])}")

    def process_data(self, vqa_dataset, imgs_directory, questions_file, annotations_file, img_name_prefix, img_file_name_len):
        dataset = []
        item = {}

        if not vqa_dataset.endswith('/'):
            vqa_dataset = vqa_dataset + '/'
        if not imgs_directory.endswith('/'):
            imgs_directory = vqa_dataset + imgs_directory + '/'

        with open(vqa_dataset + annotations_file, 'r') as json_file:
            # This is a list, each item contains an image ID, a question ID, and its corresponding answers (multiple answers to one question)
            annotations = json.load(json_file)['annotations'] 
        with open(vqa_dataset + questions_file, 'r') as json_file:
            questions = json.load(json_file)['questions'] # This is a list question IDs and the corresponding questions
        assert len(annotations) == len(questions), "Number of annotations must be equal to number of questions" 

        for idx in range(len(questions)):
            assert annotations[idx]['image_id'] == questions[idx]['image_id'] and annotations[idx]['question_id'] == questions[idx]['question_id']
            image_id = str(annotations[idx]['image_id'])
            img_file_name = img_name_prefix + '0' * (img_file_name_len - len(image_id) - len(img_name_prefix)) + image_id + '.jpg'
            try:
                # if the file does not exist or transpose does not work due to damaged data, we simply discard this sample and move to next
                img = Image.open(imgs_directory + img_file_name) 
                img= img.resize((256, 256))
                img_data = np.asarray(img)
                img_data = img_data.transpose(2, 0, 1) # reshape from (256, 256, 3) to (3, 256, 256)
            except:
                continue
            # Need to add a new dimension to (3, 256, 256) so it becomes (1, 3, 256, 256) where the added dummy dimension at dim 0 is the num_images. 
            # In this case, num_images is always 1. This is for the purpose of aligning the data structure with that in the model training
            item['image'] = torch.tensor(img_data[np.newaxis, :])
            item['question'] = questions[idx]['question']
            item['answers'] = annotations[idx]['answers']
            dataset.append(item)
        return dataset
    
    def sample_batch(self, batch_size):
        random_indices = np.random.randint(0, len(self.dataset['train']), size=batch_size)
        random_indices = [i.item() for i in random_indices]
        selected_examples = [self.dataset['train'][idx] for idx in random_indices]

        batch_dicts = []
        for item in selected_examples:
            answer_idx = random.randint(0, len(item['answers'])-1) # randomly choose an answer out of the set of answers
            batch_dict = {
                'images': item['image'],
                # 'text' is to concat the question and a randomly chosen answer with a space in between
                'text': item['question'] + ' ' + item['answers'][answer_idx]['answer'] 
            }
            batch_dicts.append(batch_dict)
        return batch_dicts

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
            question = self.dataset['test'][idx]['question']
            answer_idx = random.randint(0, len(self.dataset['test'][idx]['answers'])-1) # randomly choose an answer out of the set of answers
            target_answer = self.dataset['test'][idx]['answers'][answer_idx]['answer'] 
            target_tokens = tokenizer.encode(target_answer)

            # Generate prediction
            pred_logits, pred_answer = model.predict_answer(image, question, max_length = len(target_tokens),deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                print(f'Target answer: {target_answer} \n Predicted answer : {pred_answer}')
                print("----")

            #print(f"pred_logits.shape = {pred_logits.shape}")
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
    task = VqaTask(task_type = 'vqa', 
                   vqa_dataset              = '/home/<user name>/Git/NEKO/VQA_Data/',
                   train_imgs_directory     = 'train2014', 
                   train_questions_file     = 'OpenEnded_mscoco_train2014_questions.json', 
                   train_annotations_file   = 'mscoco_train2014_annotations.json', 
                   test_imgs_directory      = 'val2014', 
                   test_questions_file      = 'OpenEnded_mscoco_val2014_questions.json', 
                   test_annotations_file    = 'mscoco_val2014_annotations.json')

    batch = task.sample_batch(5)
    print(type(batch))
    print(list(batch[0].keys()))
    print(batch[0]['images'][0][1][10])
    print(batch[0]['images'][0][2][15])
    print(batch[0]['images'].shape)
    print(batch[0]['text'])

