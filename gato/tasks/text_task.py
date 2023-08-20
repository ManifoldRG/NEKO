# supports dataset in huggingface datasets library for now
from datasets import load_dataset, concatenate_datasets
from gato.tasks.task import Task, TaskTypeEnum
import numpy as np
import math
from torch.nn import functional as F
from typing import List

class TextTask(Task): 
       
    def __init__(self, task_type, dataset_names:List[str]):
        super().__init__(task_type)
        text_datasets_list = []
        for text_dataset in dataset_names:
            text_datasets_list.append(load_dataset(text_dataset))
        if len(text_datasets_list) == 1:
            self.text_dataset = text_datasets_list[0]
        else:            
            # https://huggingface.co/docs/datasets/v2.14.4/en/process#concatenate
            # must have the same feature columns
            self.text_dataset = concatenate_datasets(text_datasets_list)
        
    def sample_batch(self, batch_size):
        random_indices = np.random.randint(0, len(self.text_dataset['train']), size=batch_size)
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

    def evaluate(self, model, num_examples_to_test=100, deterministic=True):
        
        loss = 0
        num_tokens = 0
        
        if num_examples_to_test > len(self.text_dataset['valid']):
            print(f'num_examples_to_test chosen is more than valid examples, so setting it to whole valid dataset.')
            num_examples_to_test = len(self.text_dataset['valid'])
        
        for idx in range(num_examples_to_test):
            text = self.text_dataset['valid'][idx]['text']
            target_text = self.text_dataset['valid'][idx]['text']
            
            output_ids = model.predict_text(text, max_length=100, deterministic=deterministic)
            output_text = self.text_tokenizer.decode(output_ids)
            print(f'Generated text:{output_text}')
            
            target_ids = self.text_tokenizer.encode(target_text)
            l = F.cross_entropy(output_ids, target_ids)
            loss += l
            num_tokens += len(target_ids)
        
        avg_loss = loss / num_tokens
        perplexity = math.exp(avg_loss)
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity
        }
        return metrics
    

    