# supports dataset in huggingface datasets library for now
from datasets import load_dataset, concatenate_datasets
from gato.tasks.task import Task, TaskTypeEnum
import numpy as np
import math
from torch.nn import functional as F
from torch import nn
from typing import List
from transformers import AutoTokenizer, GPT2Tokenizer
import torch

class TextTask(Task): 
       
    def __init__(self, task_type, dataset_names:List[str]):
        super().__init__(task_type)
        text_datasets_list = []
        for text_dataset in dataset_names:
            text_datasets_list.append(load_dataset(path='wikitext', name=text_dataset))
        if len(text_datasets_list) == 1:
            self.text_dataset = text_datasets_list[0]
        else:            
            # https://huggingface.co/docs/datasets/v2.14.4/en/process#concatenate
            # must have the same feature columns
            self.text_dataset = concatenate_datasets(text_datasets_list)
        
        
    def sample_batch(self, batch_size):
        random_indices = np.random.randint(0, len(self.text_dataset['train']), size=batch_size)
        random_indices = [i.item() for i in random_indices]
        # may need more customisation as we switch up more datasets
        selected_text_examples = [self.text_dataset['train'][idx]['text'] for idx in random_indices]
        
        batch_dicts = []
        for text in selected_text_examples:
            batch_dict = {
                'text': text,
                'images': None,
                'continuous_obs': None,
                'discrete_obs': None,
                'continuous_actions': None,
                'discrete_actions': None
            }
            batch_dicts.append(batch_dict)

        # current format expected is a list of dict
        return batch_dicts

    def evaluate(self, model, num_examples_to_test=100, deterministic=True, log_examples_to_output=False):
        tokenizer = model.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        if num_examples_to_test > len(self.text_dataset['test']):
            print(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.text_dataset['test'])

        if log_examples_to_output:
            print(f'--- examples ---')
        for idx in range(num_examples_to_test):
            text = self.text_dataset['test'][idx]['text']
            if not text:
                continue

            # Tokenize the text using the model's tokenizer
            tokens = tokenizer.encode(text)
            
            # Split the tokens into input and target tokens
            ith_position = np.random.randint(1, len(tokens))
            input_tokens = tokens[:ith_position]
            target_tokens = tokens[ith_position:]

            # Generate prediction
            pred_logits, pred_tokens = model.predict_text(tokenizer.decode(input_tokens), max_length=len(tokens)-len(input_tokens), deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                print(f'Text Example : {text} \n Input passed to model : {tokenizer.decode(input_tokens)} \n Predicted output : {tokenizer.decode(pred_tokens.squeeze())}')
                print("----")

            # Calculate loss
            loss = loss_fn(pred_logits, torch.tensor(target_tokens))
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
