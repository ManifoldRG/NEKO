import argparse
import os
import json
import time

import numpy as np
import torch

from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator


from gato.utils.utils import DotDict
from gato.policy.gato_policy import GatoPolicy
from gato.envs.setup_env import load_envs
from gato.training.trainer import Trainer
from gato.tasks.control_task import ControlTask
from gato.tasks.task import TaskTypeEnum


def main(args):
    # load checkpoint
    gato_checkpoint = torch.load(args.model_path, map_location=args.device)

    # load args
    if args.args_path is None:
        args_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
    else:
        args_path = args.args_path
    
    training_args = json.load(open(args_path, 'r'))
    if not ('lora' in training_args and training_args['lora']):
        training_args['pretrained_lm'] = None

    # update args with eval_args
    for k, v in args.items():
        if v is not None:
            training_args[k] = v

    eval_args = DotDict(training_args)

    env_args = {
        'render_mode': 'human' if args.render else None,
    }

    envs, datasets = load_envs(eval_args.control_datasets, env_args) # Load Minari datasets and corresponding Gym environments

    tasks = []
    env_names = []
    for env, dataset in zip(envs, datasets):
        task = ControlTask(
            TaskTypeEnum.CONTROL.value,
            env.unwrapped.spec.id, 
            env, 
            dataset,
            args = eval_args,
            context_len=eval_args.sequence_length,
            training_prompt_len_proportion=eval_args.prompt_len_proportion,
            share_prompt_episodes = not eval_args.unique_prompt_episodes,
            top_k_prompting = args.top_k
        )
        env_names.append(env.unwrapped.spec.id)
        tasks.append(task)
    print('Evaluating on envs:', env_names)

    model = GatoPolicy(
        device=eval_args.device,
        embed_dim=eval_args.embed_dim,
        layers=eval_args.layers,
        heads=eval_args.heads,
        dropout=eval_args.dropout,
        mu=eval_args.mu,
        M=eval_args.M,
        patch_size=eval_args.patch_size,
        resid_mid_channels=eval_args.resid_mid_channels,
        continuous_tokens=eval_args.continuous_tokens,
        discrete_tokens=eval_args.discrete_tokens,
        context_len=eval_args.sequence_length,
        use_patch_pos_encoding=not eval_args.disable_patch_pos_encoding,
        use_pos_encoding=not eval_args.disable_inner_pos_encoding,
        activation_fn=eval_args.activation_fn,
        pretrained_lm=eval_args.pretrained_lm,
        flash=eval_args.flash
    )
    if eval_args.get('lora', False):
        assert eval_args.pretrained_lm is not None, 'Must specify pretrained LM for LORA'
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=eval_args.lora_r, lora_alpha=eval_args.lora_alpha, lora_dropout=eval_args.lora_dropout)
        model.transformer = get_peft_model(model.transformer, peft_config)

    model.load_state_dict(gato_checkpoint)

    accelerator = Accelerator(cpu=eval_args.cpu, mixed_precision=eval_args.mixed_precision)
    model = accelerator.prepare(model)
    eval_args.device = accelerator.device                


    model = model.to(eval_args.device)
    model.device = eval_args.device

    logs = {}
    model.eval()
    eval_start = time.time()
    
    # loop over eval for each env
    with torch.no_grad():
        for task in tasks:
            eval_logs = task.evaluate(model, n_iterations=eval_args.eval_episodes, deterministic=eval_args.eval_mode == 'deterministic', promptless_eval=eval_args.promptless_eval)
            for k, v in eval_logs.items():
                logs[f'evaluation/{task.name}/{k}'] = v

    logs['time/evaluation'] = time.time() - eval_start

    print('=' * 80)
    print(f'Evaluation results:')
    for k, v in logs.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None) # path to model checkpoint
    parser.add_argument('--args_path', type=str, default=None) # path to args.json file, will use args from same dir if None

    parser.add_argument('--cpu', default=False, action='store_true')

    # evaluation
    parser.add_argument('--eval_episodes', type=int, default=None)
    parser.add_argument('--eval_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    parser.add_argument('--promptless_eval', action='store_true', default=None)
    parser.add_argument('--top_k', type=int, default=None) # sample prompts only from top k episodes
    parser.add_argument('--render', action='store_true', default=None)

    # datasets / envs
    parser.add_argument('--control_datasets', type=str, nargs='+', default=None)

    args = parser.parse_args()
    args = DotDict(vars(args))
    main(args)
