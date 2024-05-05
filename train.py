import random
import os
from datetime import datetime

import wandb
import torch

from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate import DataLoaderConfiguration

from gato.utils.typed_argparser import TypedArgumentParser
from gato.training.arguments import TrainingArgs
from gato.policy.gato_policy import GatoPolicy
from gato.envs.setup_env import load_envs
from gato.training.trainer import Trainer
from gato.training.schedulers import get_linear_warmup_cosine_decay_scheduler
from gato.tasks.control_task import ControlTask
from gato.tasks.text_task import TextTask
from gato.tasks.caption_task import CaptionTask
from gato.tasks.vqa_task import VqaTask


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dl_config = DataLoaderConfiguration(split_batches=True)
    accelerator = Accelerator(cpu=args.cpu, dataloader_config=dl_config, mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    args.device = accelerator.device.type

    exp_date = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    exp_name = f'neko-gato_{exp_date}'

    tasks = []
    # add control datasets and env
    envs, control_datasets = load_envs(args.control_datasets) # Load Minari datasets and corresponding Gym environments
    for env, dataset in zip(envs, control_datasets):
        task = ControlTask(
            env.unwrapped.spec.id,
            env,
            dataset,
            args = args,
            context_len = args.sequence_length,
            training_prompt_len_proportion=args.prompt_len_proportion,
            share_prompt_episodes = not args.unique_prompt_episodes,
            top_k_prompting = args.top_k
        )
        tasks.append(task)
    
    if len(args.text_datasets) > 0:
        # add text datasets
        tasks.append(TextTask(args.text_datasets, args.text_datasets_paths, args.sequence_length, tokenizer_model=args.tokenizer_model_name))
    else:
        assert (args.text_prop == 0), 'text_prop must be 0 if no text datasets are specified'
 
    if len(args.caption_dataset) > 0:
        # add caption datasets
        tasks.append(CaptionTask(args.tokenizer_model_name, args.caption_dataset, args.caption_train_data, args.caption_test_data, args.test_data_prop))
    else:
        assert (args.caption_prop == 0), 'caption_prop must be 0 if no text datasets are specified'
    
    if len(args.vqa_dataset) > 0:
        # add vqa datasets
        tasks.append(VqaTask(args.tokenizer_model_name, 
                             args.vqa_dataset, args.vqa_train_data, args.vqa_test_data, 
                             args.train_img_name_prefix, args.train_img_file_name_len, 
                             args.test_img_name_prefix, args.test_img_file_name_len, 
                             args.questions_file, args.annotations_file))
    else:
        assert (args.vqa_prop == 0), 'vqa_prop must be 0 if no text datasets are specified'


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
    args.embed_dim = model.embed_dim
    model = accelerator.prepare(model)
    
    if args.lora:
        assert args.pretrained_lm is not None, 'Must specify pretrained LM for LORA'
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        model.transformer = get_peft_model(model.transformer, peft_config)

    if args.init_checkpoint is not None:
        with accelerator.main_process_first():
            print('Loading model from checkpoint:', args.init_checkpoint)
            model.load_state_dict(torch.load(args.init_checkpoint, map_location=args.device))

    # print trainable parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable Parameters:', '{}M'.format(params / 1e6))
    args.trainable_params = params


    model.device = args.device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta_1, args.beta_2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # Setup scheduler
    scheduler = get_linear_warmup_cosine_decay_scheduler(optimizer, args.warmup_steps, args.training_steps, base_lr=args.learning_rate, init_lr=args.init_lr, min_lr=args.learning_rate / args.min_factor, cosine_decay=not args.disable_cosine_decay)

    # setup up Accelerate, without dataloader:
    #model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    if args.use_wandb:
        accelerator.init_trackers(args.wandb_project, init_kwargs={'wandb': {'name': exp_name, 'config': args}})
    else:
        accelerator.init_trackers('')

    # Create save dir if does not exist
    if args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        accelerator = accelerator,
        tasks = tasks,
        exp_name = exp_name,
        args=args
    )
    trainer.train()

if __name__ == '__main__':
    parser = TypedArgumentParser(TrainingArgs)
    (args,) = parser.parse_args_into_dataclasses()

    # Checks
    assert args.training_steps % args.log_eval_freq == 0, 'training_steps must be divisible by eval_freq'
    assert args.training_steps > args.warmup_steps, 'training_steps must be greater than warmup_steps'
    assert args.learning_rate > args.init_lr, 'learning_rate must be greater than init_lr'

    # make sure proportions are between 0 and 1
    assert 0 <= args.prompt_ep_proportion <= 1, 'prompt_ep_proportion must be between 0 and 1'
    assert 0 <= args.prompt_len_proportion <= 1, 'prompt_len_proportion must be between 0 and 1'
    main(args)
