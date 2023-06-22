import time

import wandb
import numpy as np

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
        self.print_logs = args.print_logs

        self.min_lr = self.args.learning_rate / self.args.min_factor

        self.steps = 0
    
    def train(self):
        iters = self.training_steps // self.args.log_eval_freq
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
        for task in self.tasks:
            eval_logs = task.evaluate(self.model)
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
        lr = linear_warmup_cosine_decay(self.steps, self.args.warmup_steps, base_lr, min_lr, init_lr, disable_cosine_decay=self.args.disable_cosine_decay)
        logs['training/learning_rate'] = lr

        # Apply
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # TODO, training


        return loss, logs

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