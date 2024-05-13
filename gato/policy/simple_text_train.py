# RUN COMMAND : CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6" accelerate launch simple_text_train.py

from __future__ import annotations
# supports dataset in huggingface datasets library for now

import sys
sys.path.insert(0, '/home/bhavul/bhavul/NEKO/')

import torch
import time
import os

import wandb
import numpy as np
import torch
import random
import os
from datetime import datetime

import wandb
import torch

from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate import DataLoaderConfiguration

from datasets import load_dataset, concatenate_datasets
import numpy as np
from torch import nn
from typing import TYPE_CHECKING, List,Dict
from transformers import AutoTokenizer
import torch
import copy

from typing import Optional, Union, TYPE_CHECKING
import torch
import torch.nn as nn

import transformers
from transformers import AutoTokenizer

# import gato
from gato.transformers import GPT2Model
from gato.training.trainer import Trainer
from gato.training.schedulers import get_linear_warmup_cosine_decay_scheduler
from gato.tasks.task import Task
from gato.utils.utils import save_model
from gato.training.arguments import TrainingArgs


class GatoPolicy(nn.Module):
    def __init__(
        self,
        device: Union[torch.device, str],
        embed_dim: int,
        layers: int,
        heads: int,
        dropout: float,

        activation_fn='gelu',

        mu: int = 100,
        M: int = 256,

        patch_size: int = 16,
        resid_mid_channels: int = 132,
        num_groups: int = 32,
        position_vocab_size: int = 128,
        continuous_tokens: int = 1024,
        discrete_tokens: int = 1024,

        context_len=1024,

        use_pos_encoding: bool = True,
        use_patch_pos_encoding: bool = True,

        pretrained_lm: Optional[str] = None, # Optional, name of pretrained language model to use
        flash: bool = False, # TODO verify correctness
        tokenizer_model_name: str = 'gpt2',
        pad_seq: bool = False
    ):
        super().__init__()
        self.device = device
        self.context_len = context_len
        
        # Text Tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)        
        # tokens
        self.vocab_size = self.text_tokenizer.vocab_size 
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        

        if pretrained_lm is not None:
            print(f'loading pretrained GPT2 weights')
            config = transformers.GPT2Config.from_pretrained(pretrained_lm)
            config.attn_pdrop = dropout # 0.1
            config.resid_pdrop = dropout
            config.flash = flash
            config.gate = False
            config.attn_pdrop = dropout # 0.1
            config.resid_pdrop = dropout
            self.transformer = GPT2Model.from_pretrained(
                pretrained_lm,
                config=config,
            )
            embed_dim = config.n_embd
            # assert self.transformer.wte.weight.shape[0] == self.text_tokens, "pretrained token/expected mimsatch" # potentially make text_tokens dynamic
        else:
            gate = False
            if activation_fn == 'geglu':
                gate = True
                activation_fn = 'gelu'
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=embed_dim,
                n_head=heads,
                n_layer=layers,
                resid_pdrop=dropout,
                attn_pdrop=dropout,
                n_positions=context_len,
                n_inner=embed_dim * 4,
                activation_function=activation_fn,
            )
            config.n_ctx = context_len
            config.gate = gate
            config.flash = flash
            self.transformer = self.transformer = GPT2Model(config)
        
        # embedding tokens
        self.embed_token = nn.Embedding(self.vocab_size, embed_dim)
        if pretrained_lm is not None:
            self.embed_token.weight.data[:] = self.transformer.wte.weight.data
        
        
        # head
        self.predict_token = nn.Linear(embed_dim, self.vocab_size, bias=False)
        self.separator_token = nn.Parameter(torch.zeros(embed_dim))

    @property
    def module(self):
        return self

    def forward(self, inputs: Optional[list]=None, compute_loss=False, **kwargs):
        # tokenize inputs
        if inputs is not None:
            token_embeddings, tokens, token_masks, target_tokens, target_masks = self.tokenize_input_dicts(inputs)
        else:
            token_embeddings = kwargs['token_embeddings']
            tokens = kwargs['tokens']
            token_target_masks = kwargs['token_target_masks']
            token_masks = kwargs['token_masks']

        assert token_embeddings is not None, "token_embeddings is None"
        assert token_masks is not None, "token_masks is None"

        final_representations = self.transformer(inputs_embeds=token_embeddings, attention_mask=token_masks)['last_hidden_state']
        logits = self.predict_token(final_representations)
        # assert 'target' in kwargs, "target is not there in kwargs"

        # print(f"Type of target_tokens: {type(target_tokens)}")
        # print(f"Shape of target_tokens: {target_tokens.shape if isinstance(target_tokens, torch.Tensor) else 'N/A'}")
        # print(f"Type of pad_token_id: {type(self.text_tokenizer.pad_token_id)}")
        if compute_loss:
            # Ensuring target_tokens is a tensor
            if not isinstance(target_tokens, torch.Tensor):
                raise TypeError("target_tokens must be a torch.Tensor")
            
            # Correctly computing the loss mask
            loss_masks = (target_tokens != self.text_tokenizer.pad_token_id)
            if isinstance(loss_masks, torch.Tensor):
                loss_masks = loss_masks.float()  # Convert boolean tensor to float
            else:
                raise TypeError("Loss mask calculation did not return a tensor.")
            # loss_masks = (target_tokens != self.text_tokenizer.pad_token_id).float()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, self.vocab_size), target_tokens.view(-1), reduction='none')
            loss = (loss * loss_masks.view(-1)).sum() / loss_masks.sum()
        else:
            loss = None
    
        return logits, loss


    def tokenize_input_dicts(self, inputs: list):
        # if not inputs:
        #     return None, None, None, None
    
        batch_len = len(inputs)
        max_input_tokens = max(len(batch['text']) for batch in inputs)
        max_target_tokens = max(len(batch['target']) for batch in inputs) if 'target' in inputs[0] else 0
        
        # Allocate tensors for input tokens
        token_embeddings = torch.zeros((batch_len, max_input_tokens, self.embed_token.embedding_dim), device=self.device)
        tokens = torch.zeros((batch_len, max_input_tokens), dtype=torch.long, device=self.device)
        token_masks = torch.zeros((batch_len, max_input_tokens), device=self.device)
        
        # Allocate tensors for target tokens if they exist
        target_tokens = torch.zeros((batch_len, max_target_tokens), dtype=torch.long, device=self.device)
        target_masks = torch.zeros((batch_len, max_target_tokens), device=self.device)
    
        for i, batch in enumerate(inputs):
            # Process input tokens
            input_tokens = batch['text'].to(device=self.device) if isinstance(batch['text'], torch.Tensor) else torch.tensor(batch['text'], dtype=torch.long, device=self.device)
            n_input_timesteps = len(input_tokens)
            
            tokens[i, :n_input_timesteps] = input_tokens
            token_embeddings[i, :n_input_timesteps] = self.embed_token(input_tokens)
            token_masks[i, :n_input_timesteps] = 1
            
            # Process target tokens if they exist
            if 'target' in batch:
                target_data = batch['target'].to(device=self.device) if isinstance(batch['target'], torch.Tensor) else torch.tensor(batch['target'], dtype=torch.long, device=self.device)
                n_target_timesteps = len(target_data)
                target_tokens[i, :n_target_timesteps] = target_data
                target_masks[i, :n_target_timesteps] = 1
    
        return token_embeddings, tokens, token_masks, target_tokens, target_masks

    def predict_text(self, input_text, max_length=20, deterministic=True, context_length=1024):
        tokenized_outputs = self.text_tokenizer(input_text, truncation=True, padding="longest", max_length=context_length, return_tensors='pt')
        # using padding=max_length didn't work. causes CUDA OOM or other issues.

        input_tokens = tokenized_outputs['input_ids']
        predicted_tokens = input_tokens.clone()
    
        for _ in range(max_length):
            token_embeddings = self.embed_token(predicted_tokens.to(device))
            token_masks = torch.ones((predicted_tokens.to(device).shape[0], 1), device=device)

            logits, _ = self.forward(token_embeddings=token_embeddings, tokens=predicted_tokens.to(device), token_masks=token_masks, token_target_masks=None)
            logits = logits[:, -1, :]
                
    
            if deterministic:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)  # Ensure it keeps batch dimension
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # Sampling a token
    
            predicted_tokens = torch.cat([predicted_tokens.to(device), next_token.to(device)], dim=1)
    
        # all_logits = torch.cat(logits_list, dim=1)
        return predicted_tokens[:, input_tokens.size(1):]
    
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
    


def sample_text_batch(batch_size):
    batch_dicts = []
    text_tasks = [t for t in tasks if isinstance(t, TextTask)]
    for i,task in enumerate (text_tasks):
        return task.sample_batch(batch_size)

def train_step():
    logs = {}
    logs['training/learning_rate'] = scheduler.get_lr()[0] # store LR at current step
    # Build training batch
    start_time = time.time()

    # Calculate batch size for each task, the following need to be revised to including more new tasks
    text_batch_size = int(args.text_prop * args.batch_size)
    remainder = args.batch_size - text_batch_size

    if remainder > 0: 
        text_batch_size += remainder

    assert args.batch_size == text_batch_size, "Total batch size is not eqaual to the sum of each task's batch size" 

    text_batch_dicts = []

    # Sample text and control batches
    if text_batch_size > 0:
        text_batch_dicts = sample_text_batch(text_batch_size)

    if not text_batch_dicts:  # Handle empty batch case
        # print("Received an empty batch. Skipping this step.")
        return None  # You could return None or handle this case based on your training logic

    # print(f'text_batch_size:{text_batch_size}')

    logs['time/sample_batch'] = time.time() - start_time
    with accelerator.accumulate(model):
        # Compute loss and update model
        logits, loss = model.forward(inputs = text_batch_dicts, compute_loss=True)
        accelerator.backward(loss)

        if not args.disable_grad_clip and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return loss.detach().cpu().item(), logs

def train_iteration(num_steps, iter):
    logs = {}

    train_start = time.time()

    train_losses = []
    steps = 0
    model.train()
    for i in range(num_steps):
        steps += 1
        result = train_step()
        if result is None:
            # steps -= 1
            # print("Skipped a training step due to empty batch.")
            continue
        train_loss, step_logs = result
        train_losses.append(train_loss)

    # add logs from last train_step as well
    for log in step_logs:
        logs[log] = step_logs[log]

    logs['time/training'] = time.time() - train_start

    eval_start = time.time()
    model.eval()

    # loop over eval for each env
    with torch.no_grad():
        for task in tasks:
            eval_logs = {}
            if isinstance(task, TextTask):
                eval_logs = task.evaluate(model, num_examples_to_test=args.eval_text_num_examples, deterministic=deterministic)
                for k, v in eval_logs.items():
                    logs[f'evaluation/text/{k}'] = v
                pass

                if iter % 100 == 0 and args.eval_text_log_examples:
                    dataset_split = task.text_dataset['test']

                    sampled_indices = torch.randperm(len(dataset_split))[:5]
                    samples = dataset_split.select(sampled_indices)
                    
                    for sample in samples:
                        # GPT-2 hardcore limit of context length!
                        # If you don't set it, and you get an example > 1024 in length
                        # You face that weird error : tensor a (1024) must match dimension tensor b (1023) at singleton dimension 3 (something like this) 
                        actual_text = sample['text'][:1024]
                        # roughly speaking...splitting by spaces
                        words_list = actual_text.split()
                        if len(words_list) > 1:
                            split_index = random.randint(1, len(words_list)-1)
                            input_text, target_text = ' '.join(words_list[:split_index]), ' '.join(words_list[split_index:])
                            print(f'input text : {input_text} | split index: {split_index}')
                            pred_tokens = model.predict_text(input_text=actual_text, max_length=len(words_list[split_index:]), deterministic=deterministic)
                            decoded_target = task.text_tokenizer.decode(pred_tokens.squeeze(), skip_special_tokens=True)
                            print(f'Input: {input_text} | Output : {target_text} | Prediction: {decoded_target}')

    logs['time/total'] = time.time() - start_time
    logs['time/evaluation'] = time.time() - eval_start
    logs['training/train_loss_mean'] = np.mean(train_losses)
    logs['training/train_loss_std'] = np.std(train_losses)

    if accelerator.is_main_process:
        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter}')
            for k, v in logs.items():
                print(f'{k}: {v}')
            print('=' * 80)

    ## Save model
    if args.save_model and args.save_mode == 'checkpoint':
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_model(unwrapped_model, exp_dir, f'checkpoint_{steps}', args)
                

    return logs


args = TrainingArgs(
    training_steps=15000,
    log_eval_freq=10,
    warmup_steps=100,
    batch_size=8,
    gradient_accumulation_steps=8,
    sequence_length=1024,
    eval_episodes=5,
    text_prop=1,
    eval_text_log_examples=False, # set to false cuz accelerate/multigpu doesn't work with it
    # pretrained_lm='gpt2',
    text_datasets=['text'],
    text_datasets_paths=["JeanKaddour/minipile"],
    # text_datasets=['wikitext-2-v1'],
    # text_datasets_paths=['wikitext'],
    use_wandb=True,
    device='cuda:1',
    eval_mode='stochastic',
    eval_text_num_examples=16,
    # heads=8,
    # mixed_precision='fp16',
    cpu=False,
    save_dir='models_minipile',
    save_model=True,
    # disable_cosine_decay=True
)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
if args.use_wandb:
    log_with = 'wandb'
else:
    log_with = None
dl_config = DataLoaderConfiguration(split_batches=True)
accelerator = Accelerator(
    cpu=args.cpu,
    dataloader_config=dl_config, 
    # mixed_precision=args.mixed_precision,
    # gradient_accumulation_steps=args.gradient_accumulation_steps,
    kwargs_handlers=[ddp_kwargs],
    log_with=log_with
)
args.device = accelerator.device.type
exp_date = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
exp_name = f'neko-gato_{exp_date}'

model = GatoPolicy(
        device=args.device,
        embed_dim=args.embed_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        mu=args.mu,
        M=args.M,
        patch_size=args.patch_size,
        resid_mid_channels=args.resid_mid_channels,
        continuous_tokens=args.continuous_tokens,
        discrete_tokens=args.discrete_tokens,
        context_len=args.sequence_length,
        use_patch_pos_encoding=not args.disable_patch_pos_encoding,
        use_pos_encoding=not args.disable_inner_pos_encoding,
        activation_fn=args.activation_fn,
        pretrained_lm=args.pretrained_lm,
        flash=args.flash,
        tokenizer_model_name=args.tokenizer_model_name,
        pad_seq=args.pad_seq,
    )
model = accelerator.prepare(model)
model.device = args.device

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.beta_1, args.beta_2),
    eps=args.adam_eps,
    weight_decay=args.weight_decay,
)

scheduler = get_linear_warmup_cosine_decay_scheduler(optimizer, args.warmup_steps, args.training_steps, base_lr=args.learning_rate, init_lr=args.init_lr, min_lr=args.learning_rate / args.min_factor, cosine_decay=not args.disable_cosine_decay)
optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

if args.use_wandb:
    accelerator.init_trackers(args.wandb_project, init_kwargs={'wandb': {'name': exp_name, 'config': args}})
else:
    accelerator.init_trackers('')

tasks = [TextTask(args.text_datasets, args.text_datasets_paths, args.sequence_length, tokenizer_model=args.tokenizer_model_name)]
args = args
print_logs = True # args.print_logs
device = torch.device(args.device)

min_lr = args.learning_rate / args.min_factor
deterministic = args.eval_mode == 'deterministic'

exp_name = exp_name
exp_dir = os.path.join(args.save_dir, exp_name)

steps = 0
start_time = None

# Create save dir if does not exist
if args.save_model and not os.path.exists(args.save_dir):
    print(f'saving model to {args.save_dir}')
    os.makedirs(args.save_dir)
    
start_time = time.time()
iters = args.training_steps // args.log_eval_freq
print(f'iters:{iters}')
for i in range(iters):
    logs = train_iteration(args.log_eval_freq, i)
    accelerator.log(logs)

## Save model at end of training only if not saving checkpoints
if args.save_model and args.save_mode == 'last':
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_model(unwrapped_model, exp_dir, f'checkpoint_{steps}', args)
        torch.cuda.empty_cache()    

accelerator.end_training()