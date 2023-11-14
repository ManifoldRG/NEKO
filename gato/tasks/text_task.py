# supports dataset in huggingface datasets library for now
from datasets import load_dataset, concatenate_datasets
from gato.tasks.task import Task, TaskTypeEnum
import numpy as np
import math
from torch.nn import functional as F
from torch import nn
from typing import List,Dict
from transformers import AutoTokenizer, GPT2Tokenizer
import torch
import copy
class TextTask(Task): 
       
    def __init__(self, task_type, dataset_names:List[str], context_length:int, tokenizer_model:str):
        super().__init__(task_type)
        self.context_length = context_length
        self.text_tokenizer = AutoTokenizer.from_pretrained('gpt2') # todo : remove hardcoded
        text_datasets_list = []
        for text_dataset in dataset_names:
            text_datasets_list.append(load_dataset(path='wikitext', name=text_dataset))
        if len(text_datasets_list) == 1:
            self.text_dataset = text_datasets_list[0]
        else:            
            # https://huggingface.co/docs/datasets/v2.14.4/en/process#concatenate
            # must have the same feature columns
            self.text_dataset = concatenate_datasets(text_datasets_list)

        
    def sample_batch(self, batch_size, is_test=False)->List[Dict]:
        """gets used while training...every step you need to fetch batch_size jitne examples."""
        partition = 'train' if not is_test else 'test'
        random_indices = np.random.randint(0, len(self.text_dataset[partition]), size=batch_size)
        tokenized_outputs = self.text_tokenizer(self.text_dataset[partition][random_indices]['text'], truncation=True,
            max_length=self.context_length,
            return_overflowing_tokens=True,
            return_length=True)
        
        batch_dicts = []
        count = 0 
        # todo - ii. vectorize this? Also do we wanna impose any length constraint?
        # this mechanism is lossy right now as it only picks up long text examples which have full context length
        # we can fix this but requires fixes in padding as well
        for length, input_ids in zip(tokenized_outputs["length"], tokenized_outputs["input_ids"]):
            if length > 0:
                batch_example_dict = {
                    'text': input_ids,  # list of tokens
                    'images': None,
                    'continuous_obs': None,
                    'discrete_obs': None,
                    'continuous_actions': None,
                    'discrete_actions': None
                }
                batch_dicts.append(batch_example_dict)
                count += 1
                if count == batch_size:
                    break
        
        return batch_dicts

    def sample_chunk(self, chunk, seq_len):
        if chunk.shape[1] == seq_len + 1:
            start_idx = 0
        elif chunk.shape[1] > seq_len + 1:
            start_idx = torch.randint(0, chunk.shape[1] - seq_len + 1, (1,)).item()
        else:
            raise Exception(f"Invalid sequence length: Sequence length {seq_len} > {chunk.shape[1]} Chunk size")

        # todo - the if/else logic at the beginning should be respected
        start_idx = torch.randint(0, chunk.shape[1] - seq_len + 1, (1,)).item()
        inputs = chunk[:, start_idx:start_idx+seq_len-1]
        targets = chunk[:, start_idx+1:start_idx+seq_len]
        return inputs, targets
        
    def evaluate(self, model, num_examples_to_test=50, deterministic=True, log_examples_to_output=False):
        tokenizer = model.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        if num_examples_to_test > len(self.text_dataset['test']):
            print(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.text_dataset['test'])

        if log_examples_to_output:
            print(f'--- examples ---')
        
        batch_dicts = self.sample_batch(num_examples_to_test, is_test=True)
        print(f'Num of examples to test : {num_examples_to_test} | Actual batch size of test data : {len(batch_dicts)}')
        
        actual_examples_tested = 0
        for idx in range(min(num_examples_to_test, len(batch_dicts))):
            batch_dict = batch_dicts[idx]
            
            # Split the tokens into input and target tokens
            tokens = batch_dict['text']
            ith_position = np.random.randint(1, len(tokens))
            input_tokens = tokens[:ith_position]
            target_tokens = tokens[ith_position:]

            # input_tokens, target_tokens = self.sample_chunk(torch.Tensor(batch_dict['text']).long().unsqueeze(0), 8)
            new_batch_dict = copy.deepcopy(batch_dict)
            new_batch_dict['text'] = input_tokens

            # Generate prediction
            # todo - max_length should not be 20. More dynamic.
            pred_logits, pred_tokens = model.predict_text(new_batch_dict, max_length=len(target_tokens), deterministic=deterministic)
            if log_examples_to_output and idx%10==0:
                # todo - remove debug statements
                print(f'Text Example : {tokenizer.decode(batch_dict["text"])} \n Input passed to model : {tokenizer.decode(new_batch_dict["text"])} \n Predicted output : {tokenizer.decode(pred_tokens)}')
                print("----")

            # Calculate loss
            target_tokens = torch.Tensor(target_tokens).long()
            loss = loss_fn(pred_logits, target_tokens.to(model.device))
            total_loss += loss.item()
            total_tokens += len(target_tokens)
            actual_examples_tested += 1
        if log_examples_to_output:
            print(f'--- examples end ---')

        avg_loss = total_loss / actual_examples_tested
        perplexity = torch.exp(torch.tensor(avg_loss))

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
        return metrics
