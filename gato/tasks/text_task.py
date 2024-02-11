from __future__ import annotations
# supports dataset in huggingface datasets library for now
from datasets import load_dataset, concatenate_datasets
from gato.tasks.task import Task
import logging
import numpy as np
from torch import nn
from typing import TYPE_CHECKING, List,Dict
from transformers import AutoTokenizer
import torch
import copy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gato.policy.gato_policy import GatoPolicy

class TextTask(Task): 
    def __init__(self, dataset_names:List[str], dataset_paths:List[str], context_length:int, tokenizer_model:str):
        super().__init__()
        self.context_length = context_length
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
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

        
    def sample_batch(self, batch_size, is_test=False)->List[Dict]:
        """Gets called during training and test both, fetch as many examples as batch_size param."""
        partition = 'train' if not is_test else 'test'
        random_indices = np.random.randint(0, len(self.text_dataset[partition]), size=batch_size)
        tokenized_outputs = self.text_tokenizer(self.text_dataset[partition][random_indices]['text'], truncation=True,
            max_length=self.context_length,
            return_overflowing_tokens=True,
            return_length=True)
        
        batch_dicts = []
        count = 0 
        # todo - ii. vectorize this? Also do we wanna impose any length constraint?
        for length, input_ids in zip(tokenized_outputs["length"], tokenized_outputs["input_ids"]):
            # we only pick non-empty string examples right now
            if length > 0:
                batch_example_dict = {
                    'text': input_ids,  # A list of tokens
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
        
    def evaluate(self, model: GatoPolicy, num_examples_to_test=50, deterministic=True, log_examples_to_output=False):
        tokenizer = model.text_tokenizer
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        if num_examples_to_test > len(self.text_dataset['test']):
            logger.info(f'num_examples_to_test chosen is more than test examples, so setting it to whole test dataset.')
            num_examples_to_test = len(self.text_dataset['test'])

        if log_examples_to_output:
            logger.info(f'--- examples ---')
        
        batch_dicts = self.sample_batch(num_examples_to_test, is_test=True)
        logger.info(f'Num of examples to test : {num_examples_to_test} | Actual batch size of test data : {len(batch_dicts)}')
        
        actual_examples_tested = 0
        for idx in range(min(num_examples_to_test, len(batch_dicts))):
            batch_dict = batch_dicts[idx]
            
            # Split the tokens into input and target tokens
            tokens = batch_dict['text']
            ith_position = np.random.randint(1, len(tokens))
            input_tokens = tokens[:ith_position]
            target_tokens = tokens[ith_position:]

            new_batch_dict = copy.deepcopy(batch_dict)
            new_batch_dict['text'] = input_tokens

            # Generate prediction
            pred_logits, pred_tokens = model.predict_text(new_batch_dict, max_length=len(target_tokens), deterministic=deterministic)
            # todo: pull 50 into a CLI argument in train.py
            if log_examples_to_output and idx%50==0:
                logger.info(f'Text Example : {tokenizer.decode(batch_dict["text"])} \n Input passed to model : {tokenizer.decode(new_batch_dict["text"])} \n Predicted output : {tokenizer.decode(pred_tokens)}')
                logger.info("----")

            # Calculate loss
            target_tokens = torch.Tensor(target_tokens).long()
            loss = loss_fn(pred_logits, target_tokens.to(model.device))
            total_loss += loss.item()
            total_tokens += len(target_tokens)
            actual_examples_tested += 1
        if log_examples_to_output:
            logger.info(f'--- examples end ---')

        avg_loss = total_loss / actual_examples_tested
        perplexity = torch.exp(torch.tensor(avg_loss))

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity.item()
        }
        return metrics
