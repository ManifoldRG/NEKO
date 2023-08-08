import time
import os

import wandb
import numpy as np
import torch

from gato.utils.utils import save_model

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        accelerator,
        scheduler,
        tasks,
        text_dataset,
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
        self.text_dataset = text_dataset

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
                eval_logs = task.evaluate(self.model, n_iterations=self.args.eval_episodes, deterministic=self.deterministic, promptless_eval=self.args.promptless_eval)
                for k, v in eval_logs.items():
                    logs[f'evaluation/{task.name}/{k}'] = v
            # todo : add text eval as well.

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if self.accelerator.is_main_process:
            if self.print_logs:
                print('=' * 80)
                print(f'Iteration {iter}')
                for k, v in logs.items():
                    print(f'{k}: {v}')

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
        # for simplicity sake - assume interleave training on control & text
        if self.steps % 2 == 0:
            batch_dicts = self.sample_control_batch(self.args.batch_size)
        else:
            batch_dicts = self.process_text_batch(next(iter(self.text_dataset)))

        with self.accelerator.accumulate(self.model):
            # Compute loss and update model
            logits, loss = self.model.forward(inputs = batch_dicts, compute_loss=True)
            self.accelerator.backward(loss)

            if not self.args.disable_grad_clip and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.detach().cpu().item(), logs

    def process_text_batch(self, batch_size):
        # Select batch_size number of random indices from the size of the text dataset.
        random_indices = np.random.randint(0, len(self.text_dataset['train']), size=batch_size)
        selected_text_examples = [self.text_dataset['train'][idx]['text'] for idx in random_indices]

        input_dict = {}

        text_tokens = [self.model.text_tokenizer.encode(text) for text in selected_text_examples]
        input_dict['text'] = torch.stack(text_tokens).to(self.args.device)
        # The targets are the same as the input, but shifted one position to the left
        # append the appropriate end-of-sequence token as well
        input_dict['text_targets'] = torch.cat(
            [tokens[1:], torch.tensor([self.model.text_tokenizer.tokenizer.eos_token_id]).expand(1, tokens.shape[1])],
            dim=0
        ).to(self.args.device)

        return input_dict

    def sample_control_batch(self, batch_size):
        batch_dicts = []

        sampled_task_indices = []
        n_tasks = len(self.tasks)
        while len(sampled_task_indices) < batch_size:
            max_n = min(n_tasks, batch_size - len(sampled_task_indices))
            new_tasks = np.random.choice(np.arange(n_tasks), size=max_n, replace=False).tolist()
            sampled_task_indices.extend(new_tasks)

        n_prompted_episodes = round(batch_size * self.args.prompt_ep_proportion)
        vanilla_batch_size = batch_size - n_prompted_episodes

        # determine prompted episodes and their prompting type (end or uniform)
        prompt_indices = np.random.choice(batch_size, size=n_prompted_episodes, replace=False).tolist()
        end_indices = np.random.choice(prompt_indices, size=round(len(prompt_indices) / 2), replace=False).tolist()
        uniform_indices = [i for i in prompt_indices if i not in end_indices]


        # aggregate acrosss tasks sampled multiple times
        for i, task in enumerate(self.tasks):
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
