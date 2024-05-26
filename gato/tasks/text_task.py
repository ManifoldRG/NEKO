from __future__ import annotations
# supports dataset in huggingface datasets library for now
from datasets import load_dataset, concatenate_datasets
from gato.tasks.task import Task
import numpy as np
from torch import nn
from typing import TYPE_CHECKING, List,Dict
from transformers import AutoTokenizer
import torch
import copy

if TYPE_CHECKING:
    from gato.policy.gato_policy import GatoPolicy

class TextTask(Task): 
    def __init__(self, dataset_names:List[str], dataset_paths:List[str], context_length:int, tokenizer_model:str):
        super().__init__()
        self.context_length = context_length
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        text_datasets_list = []
        assert len(dataset_names) == len(dataset_paths), "The dataset names and paths parameters should have corresponding values and hence equal lengths"
        for i, text_dataset in enumerate(dataset_names):
            text_datasets_list.append(load_dataset(path=dataset_paths[i], name=text_dataset))
        if len(text_datasets_list) == 1:
            self.text_dataset = text_datasets_list[0]
        else:            
            # https://huggingface.co/docs/datasets/v2.14.4/en/process#concatenate
            # must have the same feature columns
            self.text_dataset = concatenate_datasets(text_datasets_list)

        
    def sample_batch(self, batch_size, is_test=False) -> List[Dict]:
        split = 'train' if not is_test else 'test'
        try:
            dataset_split = self.text_dataset[split]
        except Exception as e:
            print(f'WARNING: using train split since test split not available in Dataset')
            dataset_split = self.text_dataset['train']
            is_test=False

        if len(dataset_split) < batch_size:
            print(f"Warning: Requested batch size {batch_size} is larger than the dataset size {len(dataset_split)}.")
            batch_size = len(dataset_split)  # Adjust batch size to available data size

        if batch_size == 0:
            return []  # Early exit if no data is available

        
        sampled_indices = torch.randperm(len(dataset_split))[:batch_size]
        samples = dataset_split.select(sampled_indices)
        tokenized_outputs = self.text_tokenizer(samples['text'], truncation=True, padding="longest", max_length=self.context_length, return_tensors='pt')
    
        batch_dicts = []
        for input_ids in tokenized_outputs["input_ids"]:
            if input_ids.numel() > 0:  # Check if non-empty
                # Split into input and target tokens
                input_tokens = input_ids[:-1]
                target_tokens = input_ids[1:]
                batch_dicts.append({
                    'text': input_tokens,
                    'target': target_tokens,
                    'continuous_obs': None,
                    'discrete_obs': None,
                    'continuous_actions': None,
                    'discrete_actions': None
                })
    
        return batch_dicts
        
    def evaluate(self, model: GatoPolicy, num_examples_to_test=8, deterministic=False, is_test=True):
        # REMEMBER TO MAKE SURE THE num_examples_to_test <= total num of examples in dataset[split]
        if num_examples_to_test == 0:
            return {'loss': float('nan'), 'perplexity': float('nan')}
    
        batch_dicts = self.sample_batch(num_examples_to_test, is_test)

        # Forward pass    
        logits, loss = model(batch_dicts, compute_loss=True)
        
        # total_tokens = input_tokens.size(0) * input_tokens.size(1)
        # print(f'total tokens:{total_tokens}')
        avg_loss = loss.detach().cpu().item()
        perplexity = torch.exp(torch.tensor(avg_loss)).detach().cpu().item()
                        
        return {'loss': avg_loss, 'perplexity': perplexity}
