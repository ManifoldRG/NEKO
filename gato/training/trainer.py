import time

import wandb
import numpy as np
import torch

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        tasks,
        args
    ):
        self.model = model
        self.optimizer = optimizer
        self.tasks = tasks
        self.args = args
        self.print_logs = True # args.print_logs
        self.device = args.device

        self.min_lr = self.args.learning_rate / self.args.min_factor
        self.deterministic = self.args.eval_mode == 'deterministic'

        self.steps = 0
        self.start_time = None
    
    def train(self):
        self.start_time = time.time()
        iters = self.args.training_steps // self.args.log_eval_freq
        for i in range(iters):
            logs = self.train_iteration(self.args.log_eval_freq, i)
            if self.args.use_wandb:
                wandb.log(logs)
    
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
                eval_logs = task.evaluate(self.model, n_iterations=self.args.eval_episodes, deterministic=self.deterministic)
                for k, v in eval_logs.items():
                    logs[f'evaluation/{task.name}/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if self.print_logs:
            print('=' * 80)
            print(f'Iteration {iter}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        logs = {}
        base_lr = self.args.learning_rate
        min_lr = self.min_lr
        init_lr = self.args.init_lr

        # Calculate learning rate relative to current step
        lr = linear_warmup_cosine_decay(self.steps, self.args.warmup_steps, self.args.training_steps, base_lr, init_lr, min_lr, disable_cosine_decay=self.args.disable_cosine_decay)
        logs['training/learning_rate'] = lr

        # Apply
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Build training batch
        batch_dicts = self.sample_control_batch(self.args.batch_size)

        # Compute loss and update model
        logits, loss = self.model.forward(inputs = batch_dicts, compute_loss=True)

        self.optimizer.zero_grad()
        loss.backward()
        if not self.args.disable_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        return loss.detach().cpu().item(), logs

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

def linear_warmup_cosine_decay(current_step, warmup_steps, max_steps, base_lr, init_lr, min_lr, disable_cosine_decay=False):
    # Linear Warmup from init_lr to base_lr over warmup_steps
    if current_step <= warmup_steps:
        lr = init_lr + (base_lr - init_lr) * current_step / warmup_steps
    elif not disable_cosine_decay:
        # cosine decay from base_lr to min_lr over remaining steps
        progress = (current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    else:
        lr = base_lr
    return lr


if __name__ == '__main__':

    # Test LR schedule
    import matplotlib.pyplot as plt
    init_lr = 1e-7
    base_lr = 1e-4
    min_lr = base_lr / 10
    warmup_steps = 15_000
    max_steps = 1_015_000
    current_steps = np.arange(1, max_steps + 1)
    lr = np.zeros_like(current_steps, dtype=np.float32)
    for step in current_steps:
        lr[step - 1] = linear_warmup_cosine_decay(step, warmup_steps, max_steps, base_lr, init_lr, min_lr)
    plt.plot(current_steps, lr)
    plt.show()