import argparse
import logging
import os
from datetime import datetime

import wandb
import torch

from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


from gato.utils.utils import DotDict
from gato.policy.gato_policy import GatoPolicy
from gato.envs.setup_env import load_envs
from gato.training.trainer import Trainer
from gato.training.schedulers import get_linear_warmup_cosine_decay_scheduler
from gato.tasks.control_task import ControlTask
from gato.tasks.text_task import TextTask
from gato.tasks.caption_task import CaptionTask
from gato.tasks.vqa_task import VqaTask

logger = logging.getLogger(__name__)


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, split_batches=True, gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    args.device = accelerator.device

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
            logger.info('Loading model from checkpoint:', args.init_checkpoint)
            model.load_state_dict(torch.load(args.init_checkpoint, map_location=args.device))

    # print trainable parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Trainable Parameters:', '{}M'.format(params / 1e6))
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
        wandb.init(
            name = exp_name,
            project=args.wandb_project,
            config=args,
        )

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
    parser = argparse.ArgumentParser()

    # Accelerate args
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

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
    parser.add_argument('--tokenizer_model_name', type=str, default='gpt2')
    parser.add_argument('--pretrained_lm', type=str, default=None) # Init with pretrained LM override embed_dim, layers, heads, activation_fn
    parser.add_argument('--flash', default=False, action='store_true') # enable flash attention
    parser.add_argument('--init_checkpoint', type=str, default=None) # Will not override architecture, only load weights from Gato checkpoint

    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--heads', type=int, default=24)
    parser.add_argument('--activation_fn', type=str, default='gelu')
    #parser.add_argument('--activation_fn', type=str, default='geglu')

    # PEFT hyperparameters
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # training hyperparameters
    parser.add_argument('--text_prop', type=float, default=0) # proportion of text data in batch
    parser.add_argument('--caption_prop', type=float, default=0) # proportion of image caption data in batch
    parser.add_argument('--vqa_prop', type=float, default=0) # proportion of vqa data in batch
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1) # simulate larger batch size
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

    parser.add_argument('--pad_seq', action='store_true', default=False) # pad sequences to max length


    # evaluation
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--eval_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    parser.add_argument('--promptless_eval', action='store_true', default=False)
    parser.add_argument('--eval_text_num_examples', type=int, default=100)
    parser.add_argument('--eval_text_log_examples', action='store_true', default=False) # for debugging if you wish to show predictions from model in eval for text

    # datasets / envs
    parser.add_argument('--control_datasets', type=str, nargs='+', default=[])

    parser.add_argument('--text_datasets', type=str, nargs='+', default=[]) # ['wikitext-2-v1']
    parser.add_argument('--text_datasets_paths', type=str, nargs='+', default=[]) # ['wikitext']

    parser.add_argument('--caption_dataset', type=str, default='') # the directory for all of the data (training and test)
    parser.add_argument('--caption_train_data', type=str, nargs='+', default=[]) # list of sub directories for training data
    parser.add_argument('--caption_test_data', type=str, nargs='+', default=[]) # list of sub directories for test data
    parser.add_argument('--test_data_prop', type=str, nargs='+', default=0.1) # the proportion of test data if needing to split training dataset into training and test
    
    parser.add_argument('--vqa_dataset', type=str, default='') # the directory for all of the data (traing and test)
    parser.add_argument('--vqa_train_data', type=str, nargs='+', default=[]) # list of sub directories for training data
    parser.add_argument('--vqa_test_data', type=str, nargs='+', default=[]) # list of sub directories for test data
    parser.add_argument('--train_img_name_prefix', type=str, nargs='+', default=[]) # each sub directory has one such image name file prefix
    parser.add_argument('--train_img_file_name_len', type=int, nargs='+', default=[]) # each sub directory has one such image file name length
    parser.add_argument('--test_img_name_prefix', type=str, nargs='+', default=[]) # each sub directory has one such image name file prefix
    parser.add_argument('--test_img_file_name_len', type=int, nargs='+', default=[]) # each sub directory has one such image file name length
    parser.add_argument('--questions_file', type=str, default='questions.json') # it is required to give the same name to all questions json files (no ambiguity since they are under different directories)
    parser.add_argument('--annotations_file', type=str, default='annotations.json') # it is required to give the same name to all annotations json files (no ambiguity since they are under different directories)

    parser.add_argument('--eval_caption_num_examples', type=int, default=100)
    parser.add_argument('--eval_caption_log_examples', action='store_true', default=False) # for debugging if you wish to show predictions from model in eval for text

    parser.add_argument('--eval_vqa_num_examples', type=int, default=100)
    parser.add_argument('--eval_vqa_log_examples', action='store_true', default=False) # for debugging if you wish to show predictions from model in eval for text

    # params for sampling from datasets
    parser.add_argument('--prompt_ep_proportion', type=float, default=0.25) # proportion of episodes that are prompted
    parser.add_argument('--prompt_len_proportion', type=float, default=0.5) # proportion of context consumed by prompt
    parser.add_argument('--unique_prompt_episodes', default=False, action='store_true')
    parser.add_argument('--top_k', type=int, default=None) # sample prompts only from top k episodes

    # logging
    parser.add_argument('--use_wandb', '-w', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='gato-control')

    # saving
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_mode', type=str, default='last', choices=['checkpoint', 'last']) # Checkpoit saves model every after each log_eval_freq steps
    parser.add_argument('--save_dir', type=str, default='models')

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
