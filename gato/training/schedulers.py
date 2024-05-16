from functools import partial

import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_linear_warmup_cosine_decay_scheduler(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, base_lr: float, init_lr: float, min_lr: float , cosine_decay: bool = True, last_epoch=-1):

    lr_lambda = partial(
        _linear_warmup_cosine_decay,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        base_lr=base_lr,
        init_lr=init_lr,
        min_lr=min_lr,
        cosine_decay=cosine_decay,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _linear_warmup_cosine_decay(current_step: int, *, num_warmup_steps: int,num_training_steps: int, base_lr: float, init_lr: float, min_lr: float, cosine_decay: bool):
    if current_step <= num_warmup_steps:
        lr = init_lr + (base_lr - init_lr) * current_step / num_warmup_steps
    elif cosine_decay:
        # cosine decay from base_lr to min_lr over remaining steps
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    else:
        lr = base_lr
    # convert to proportion of base_lr
    proportion = lr / base_lr
    return proportion

if __name__ == '__main__':
    # Test LR schedule
    import matplotlib.pyplot as plt
    init_lr = 1e-7
    base_lr = 1e-4
    min_lr = base_lr / 10
    warmup_steps = 15_000
    max_steps = 1_000_000
    current_steps = np.arange(1, max_steps + 1)
    lr = np.zeros_like(current_steps, dtype=np.float32)
    for step in current_steps:
        lr[step - 1] = _linear_warmup_cosine_decay(step, warmup_steps, max_steps, base_lr, init_lr, min_lr)
    plt.plot(current_steps, lr)
    plt.show()
