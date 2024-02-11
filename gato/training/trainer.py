import logging
import time
import os

import wandb
import numpy as np
import torch
from gato.tasks.control_task import ControlTask
from gato.tasks.text_task import TextTask
from gato.tasks.caption_task import CaptionTask
from gato.tasks.vqa_task import VqaTask

from gato.utils.utils import save_model

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        accelerator,
        scheduler,
        tasks,
        exp_name,
        args
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.tasks = tasks
        self.args = args
        self.print_logs = True # args.print_logs
        self.device = args.device

        self.min_lr = self.args.learning_rate / self.args.min_factor
        self.deterministic = self.args.eval_mode == 'deterministic'

        self.exp_name = exp_name
        self.exp_dir = os.path.join(self.args.save_dir, self.exp_name)

        self.steps = 0
        self.start_time = None

    def train(self):
        self.start_time = time.time()
        iters = self.args.training_steps // self.args.log_eval_freq
        for i in range(iters):
            logs = self.train_iteration(self.args.log_eval_freq, i)
            if self.args.use_wandb and self.accelerator.is_main_process:
                wandb.log(logs)

        ## Save model at end of training only if not saving checkpoints
        if self.args.save_model and self.args.save_mode == 'last':
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                save_model(unwrapped_model, self.exp_dir, f'checkpoint_{self.steps}', self.args)


    def train_iteration(self, num_steps, iter):
        logs = {}

        train_start = time.time()

        train_losses = []

        self.model.train()
        for i in range(num_steps):
            self.steps += 1
            train_loss, step_logs = self.train_step()
            train_losses.append(train_loss)

        # add logs from last train_step as well
        for log in step_logs:
            logs[log] = step_logs[log]

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()
        self.model.eval()

        # loop over eval for each env
        with torch.no_grad():
            for task in self.tasks:
                eval_logs = {}
                if isinstance(task, ControlTask):
                    if self.args.eval_episodes > 0 :
                        eval_logs = task.evaluate(self.model, n_iterations=self.args.eval_episodes, deterministic=self.deterministic, promptless_eval=self.args.promptless_eval)
                    for k, v in eval_logs.items():
                        logs[f'evaluation/{task.name}/{k}'] = v
                elif isinstance(task, TextTask):
                    eval_logs = task.evaluate(self.model, num_examples_to_test=self.args.eval_text_num_examples, deterministic=self.deterministic, log_examples_to_output=self.args.eval_text_log_examples)
                    for k, v in eval_logs.items():
                        logs[f'evaluation/text/{k}'] = v
                    pass
                elif isinstance(task, CaptionTask):
                    eval_logs = task.evaluate(self.model, num_examples_to_test=self.args.eval_caption_num_examples, deterministic=self.deterministic, log_examples_to_output=self.args.eval_caption_log_examples)
                    for k, v in eval_logs.items():
                        logs[f'evaluation/caption/{k}'] = v
                elif isinstance(task, VqaTask):
                    eval_logs = task.evaluate(self.model, num_examples_to_test=self.args.eval_vqa_num_examples, deterministic=self.deterministic, log_examples_to_output=self.args.eval_caption_log_examples)
                    for k, v in eval_logs.items():
                        logs[f'evaluation/VQA/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if self.accelerator.is_main_process:
            if self.print_logs:
                logger.info('=' * 80)
                logger.info(f'Iteration {iter}')
                for k, v in logs.items():
                    logger.info(f'{k}: {v}')
                logger.info('=' * 80)

        ## Save model
        if self.args.save_model and self.args.save_mode == 'checkpoint':
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                save_model(unwrapped_model, self.exp_dir, f'checkpoint_{self.steps}', self.args)

        return logs

    def train_step(self):
        logs = {}
        logs['training/learning_rate'] = self.scheduler.get_lr()[0] # store LR at current step
        # Build training batch
        start_time = time.time()

        # Calculate batch size for each task, the following need to be revised to including more new tasks
        control_prop = 1-self.args.text_prop-self.args.caption_prop-self.args.vqa_prop
        text_batch_size = int(self.args.text_prop * self.args.batch_size)
        caption_batch_size = int(self.args.caption_prop * self.args.batch_size)
        vqa_batch_size = int(self.args.vqa_prop * self.args.batch_size)
        control_batch_size = int(control_prop*self.args.batch_size)

        remainder = self.args.batch_size - text_batch_size - caption_batch_size - vqa_batch_size - control_batch_size

        if remainder > 0: 
            # if this happens, we dispense the remainder to the task drawn from a multinomial distribution based off the following residulas
            # over the long run of trainig steps, the remainder dispensing will be proporttional to the amount of residuals
            residuals = [self.args.text_prop * self.args.batch_size - text_batch_size, self.args.caption_prop * self.args.batch_size - caption_batch_size,
                         self.args.vqa_prop * self.args.batch_size - vqa_batch_size, control_prop*self.args.batch_size - control_batch_size]
            idx = torch.multinomial(torch.tensor(residuals), num_samples=1).item()
            residuals = [0]*len(residuals)
            residuals[idx] = remainder
            text_batch_size = text_batch_size + residuals[0]
            caption_batch_size = caption_batch_size + residuals[1]
            vqa_batch_size = vqa_batch_size + residuals[2]
            control_batch_size = control_batch_size + residuals[3]
        assert self.args.batch_size == text_batch_size + caption_batch_size + vqa_batch_size + control_batch_size, "Total atch size is not eqaual to the sum of each task's batch size" 

        text_batch_dicts = []
        caption_batch_dicts = []
        vqa_batch_dicts = []
        control_batch_dicts = []

        # Sample text and control batches
        if text_batch_size > 0:
            text_batch_dicts = self.sample_text_batch(text_batch_size)
        if caption_batch_size > 0:
            caption_batch_dicts = self.sample_caption_batch(caption_batch_size)
        if vqa_batch_size > 0:
            vqa_batch_dicts = self.sample_vqa_batch(vqa_batch_size)        
        if control_batch_size > 0:
            control_batch_dicts = self.sample_control_batch(control_batch_size)

        # Combine the batches
        combined_batch_dicts = text_batch_dicts + caption_batch_dicts + vqa_batch_dicts+ control_batch_dicts


        logs['time/sample_batch'] = time.time() - start_time
        with self.accelerator.accumulate(self.model):
            # Compute loss and update model
            logits, loss = self.model.forward(inputs = combined_batch_dicts, compute_loss=True)
            self.accelerator.backward(loss)

            if not self.args.disable_grad_clip and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.detach().cpu().item(), logs

    def sample_text_batch(self, batch_size):
        batch_dicts = []
        text_tasks = [t for t in self.tasks if isinstance(t, TextTask)]
        for i,task in enumerate (text_tasks):
            batch_dicts.extend(task.sample_batch(batch_size))
        return batch_dicts

    def sample_caption_batch(self, batch_size):
        batch_dicts = []
        caption_tasks = [t for t in self.tasks if isinstance(t, CaptionTask)]
        for i,task in enumerate (caption_tasks):
            batch_dicts.extend(task.sample_batch(batch_size))
        return batch_dicts
    
    def sample_vqa_batch(self, batch_size):
        batch_dicts = []
        vqa_tasks = [t for t in self.tasks if isinstance(t, VqaTask)]
        for i,task in enumerate (vqa_tasks):
            batch_dicts.extend(task.sample_batch(batch_size))
        return batch_dicts

    def sample_control_batch(self, batch_size):
        batch_dicts = []

        sampled_task_indices = []
        control_tasks = [t for t in self.tasks if isinstance(t, ControlTask)]
        n_tasks = len(control_tasks)
        while len(sampled_task_indices) < batch_size:
            max_n = min(n_tasks, batch_size - len(sampled_task_indices))
            new_tasks = np.random.choice(np.arange(n_tasks), size=max_n, replace=False).tolist()
            sampled_task_indices.extend(new_tasks)

        n_prompted_episodes = round(batch_size * self.args.prompt_ep_proportion)

        # determine prompted episodes and their prompting type (end or uniform)
        prompt_indices = np.random.choice(batch_size, size=n_prompted_episodes, replace=False).tolist()
        end_indices = np.random.choice(prompt_indices, size=round(len(prompt_indices) / 2), replace=False).tolist()
        uniform_indices = [i for i in prompt_indices if i not in end_indices]

        # aggregate acrosss tasks sampled multiple times
        for i, task in enumerate(control_tasks):
            total_task_batch_size = 0
            task_vanilla_batch_size = 0
            task_prompted_batch_sizes = {}
            for type_index, task_index in enumerate(sampled_task_indices):
                if task_index == i:
                    total_task_batch_size += 1
                    if type_index in end_indices:
                        task_prompted_batch_sizes['end'] = task_prompted_batch_sizes.get('end', 0) + 1
                    elif type_index in uniform_indices:
                        task_prompted_batch_sizes['uniform'] = task_prompted_batch_sizes.get('uniform', 0) + 1
                    else:
                        task_vanilla_batch_size += 1
            # sample episodes from dataset
            if total_task_batch_size > 0:
                task_episode_dicts = task.sample_batch(task_vanilla_batch_size, task_prompted_batch_sizes, self.device, max_tokens=self.args.sequence_length)
                batch_dicts.extend(task_episode_dicts)
        return batch_dicts
