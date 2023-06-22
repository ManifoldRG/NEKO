import argparse
import random

import wandb
import torch

from gato.utils.utils import DotDict
from gato.policy.gato_policy import GatoPolicy
from gato.envs.setup_env import load_envs
from gato.training.trainer import Trainer
from gato.tasks.control_task import ControlTask

def main(args):
    exp_id = random.randint(int(1e5), int(1e6) - 1)
    exp_name = f'gato-control-{exp_id}'
    if args.use_wandb:
        wandb.init(
            name = exp_name,
            project=args.wandb_project,
            config=args,
        )

    envs, datasets = load_envs(args.datasets) # Load Minari datasets and corresponding Gym environments

    tasks = []
    for env, dataset in zip(envs, datasets):
        task = ControlTask(
            env.unwrapped.spec.id, 
            env, 
            dataset,
            training_prompt_len_proportion=args.prompt_len_proportion,
            share_prompt_episodes = not args.unique_prompt_episodes
        )
        tasks.append(task)

    model = GatoPolicy(
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
    )
    model = model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        tasks = tasks,
        args=args
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cpu') # e.g. cuda:0

    # Input & tokenization
    parser.add_argument('--sequence_length', '-k', type=int, default=1024) # number of tokens in seq
    parser.add_argument('--patch_size', type=int, default=16) # image patch size
    parser.add_argument('--resid_mid_channels', type=int, default=128) # number of channels in residual MLP
    parser.add_argument('--num_groups', type=int, default=32) # GroupNorm groups in ResNet
    parser.add_argument('--patch_position_vocab_size', type=int, default=128)
    parser.add_argument('--disable_patch_pos_encoding', action='store_true', default=False)
    parser.add_argument('--disable_inner_pos_encoding', action='store_true', default=False)

    parser.add_argument('--mu','-mu', type=int, default=100) # mu-law encoding
    parser.add_argument('--M', '-M', type=int, default=256) 

    #parser.add_argument('--vocab_size', type=int, default=32000) # number of tokens from SentencePiece
    parser.add_argument('--continuous_tokens', type=int, default=1024) # number of tokens for continuous values (e.g. actions, observations)
    parser.add_argument('--discrete_tokens', type=int, default=1024) # number of discrete action tokens

    # transformer architecture hyperparameters
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--heads', type=int, default=24)

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.95)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    parser.add_argument('--grad_norm_clip', type=float, default=1.0)
    parser.add_argument('--disable_grad_clip', action='store_true', default=False)

    parser.add_argument('--warmup_steps', type=int, default=15000)
    parser.add_argument('--init_lr', type=float, default=1e-7) # starting LR for warmup
    parser.add_argument('--learning_rate', '-lr',type=float, default=1e-4) # the maximum LR after warmup
    
    parser.add_argument('--min_factor', type=float, default=10.0) # the minimum LR factor, e.g. w/ 10, base 1e-4 -> 1e-5 for Cosine Decay
    parser.add_argument('--disable_cosine_decay', action='store_true', default=False) # disable cosine decay

    parser.add_argument('--training_steps', type=int, default=1_000_000)
    parser.add_argument('--log_eval_freq', type=int, default=100_000)

    # evaluation
    parser.add_argument('--eval_episodes', type=int, default=10)

    # datasets / envs
    parser.add_argument('--datasets', type=str, nargs='+', default=['d4rl_halfcheetah-expert-v2'])

    # params for sampling from datasets
    parser.add_argument('--prompt_ep_proportion', type=float, default=0.25) # proportion of episodes that are prompted
    parser.add_argument('--prompt_len_proportion', type=float, default=0.5) # proportion of context consumed by prompt
    parser.add_argument('--unique_prompt_episodes', default=False, action='store_true')


    # logging
    parser.add_argument('--use_wandb', '-w', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='gato-control')
    
    args = parser.parse_args()
    args = DotDict(vars(args))

    # Checks
    assert args.training_steps % args.log_eval_freq == 0, 'training_steps must be divisible by eval_freq'
    assert args.training_steps > args.warmup_steps, 'training_steps must be greater than warmup_steps'
    assert args.learning_rate > args.init_lr, 'learning_rate must be greater than init_lr'

    # make sure proportions are between 0 and 1
    assert 0 <= args.prompt_ep_proportion <= 1, 'prompt_ep_proportion must be between 0 and 1'
    assert 0 <= args.prompt_len_proportion <= 1, 'prompt_len_proportion must be between 0 and 1'


    main(args)