from utils.argparsing import Arg, ParseArger, TypedNamespace
import random
import os

import wandb
import torch

from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import transformers

from gato.utils.utils import DotDict
from gato.policy.gato_policy import GatoPolicy
from gato.envs.setup_env import load_envs
from gato.training.trainer import Trainer
from gato.training.schedulers import get_linear_warmup_cosine_decay_scheduler
from gato.tasks.control_task import ControlTask
from gato.tasks.text_task import TextTask
from gato.tasks.task import TaskTypeEnum


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, split_batches=True, gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    args.device = accelerator.device.type

    exp_id = random.randint(int(1e5), int(1e6) - 1)
    exp_name = f'neko-gato-{exp_id}'

    tasks = []
    # add control datasets and env
    envs, control_datasets = load_envs(args.control_datasets) # Load Minari datasets and corresponding Gym environments
    for env, dataset in zip(envs, control_datasets):
        task = ControlTask(
            TaskTypeEnum.CONTROL.value,
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
        tasks.append(TextTask(TaskTypeEnum.TEXT.value, args.text_datasets, args.text_datasets_paths, args.sequence_length, tokenizer_model=args.tokenizer_model_name)) 
    else:
        assert (args.text_prop == 0), 'text_prop must be 0 if no text datasets are specified'

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


class TrainArgs(TypedNamespace):
    # Accelerate args
    cpu = Arg[int]('--cpu', action='store_true', default=False, help='Force script to execute on CPU. Passed to Accelerator.')
    mixed_precision = Arg[str](
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

    # Input & tokenization
    sequence_length = Arg[int]('--sequence_length', '-k', type=int, default=1024, help="Used as context_length in models and tokenizers.")
    patch_size = Arg[int]('--patch_size', type=int, default=16, help="Images are reshaped to be a multiple of patch_size.")
    resid_mid_channels = Arg[int]('--resid_mid_channels', type=int, default=128, help="Number of channels in residual MLP. (Passed as to `nn.Conv2d` as `out_channels`.)")
    num_groups = Arg[int]('--num_groups', type=int, default=32) # GroupNorm groups in ResNet
    patch_position_vocab_size = Arg[int]('--patch_position_vocab_size', type=int, default=128)
    disable_patch_pos_encoding = Arg[int]('--disable_patch_pos_encoding', action='store_true', default=False)
    disable_inner_pos_encoding = Arg[int]('--disable_inner_pos_encoding', action='store_true', default=False)

    mu = Arg[int]('--mu','-m', type=int, default=100) # mu-law encoding
    M = Arg[int]('--M', '-M', type=int, default=256)

    #vocab_size = Arg[int]('--vocab_size', type=int, default=32000) # number of tokens from SentencePiece
    continuous_tokens = Arg[int]('--continuous_tokens', type=int, default=1024, help="Number of tokens to use for continuous values (e.g. actions, observations).")
    discrete_tokens = Arg[int]('--discrete_tokens', type=int, default=1024, help="Number of tokens to use for discrete actions.")

    # transformer architecture hyperparameters
    tokenizer_model_name = Arg[str]('--tokenizer_model_name', type=str, default='gpt2')
    pretrained_lm = Arg[str]('--pretrained_lm', type=str, default=None, help="Initialize with a pretrained language model, overriding --embed-dim, --layers, --heads, --activation-fn")
    flash = Arg[int]('--flash', default=False, action='store_true') # enable flash attention
    init_checkpoint = Arg[str]('--init_checkpoint', type=str, default=None) # Will not override architecture, only load weights from Gato checkpoint

    embed_dim = Arg[int]('--embed_dim', type=int, default=768)
    layers = Arg[int]('--layers', type=int, default=8)
    heads = Arg[int]('--heads', type=int, default=24)
    activation_fn = Arg[str]('--activation_fn', type=str, default='gelu')
    #activation_fn = Arg[str]('--activation_fn', type=str, default='geglu')

    # PEFT hyperparameters
    lora = Arg[int]('--lora', action='store_true', default=False)
    lora_r = Arg[int]('--lora_r', type=int, default=8)
    lora_alpha = Arg[int]('--lora_alpha', type=int, default=32)
    lora_dropout = Arg[float]('--lora_dropout', type=float, default=0.1)

    # training hyperparameters
    text_prop = Arg[float]('--text_prop', type=float, default=0.5) # proportion of text data in batch
    gradient_accumulation_steps = Arg[int]('--gradient_accumulation_steps', type=int, default=1) # simulate larger batch size
    batch_size = Arg[int]('--batch_size', type=int, default=512)
    dropout = Arg[float]('--dropout', type=float, default=0.1)

    beta_1 = Arg[float]('--beta_1', type=float, default=0.9)
    beta_2 = Arg[float]('--beta_2', type=float, default=0.95)
    adam_eps = Arg[float]('--adam_eps', type=float, default=1e-8)
    weight_decay = Arg[float]('--weight_decay', type=float, default=0.1)

    grad_norm_clip = Arg[float]('--grad_norm_clip', type=float, default=1.0)
    disable_grad_clip = Arg[int]('--disable_grad_clip', action='store_true', default=False)

    warmup_steps = Arg[int]('--warmup_steps', type=int, default=15000)
    init_lr = Arg[float]('--init_lr', type=float, default=1e-7) # starting LR for warmup
    learning_rate = Arg[float]('--learning_rate', '-lr',type=float, default=1e-4) # the maximum LR after warmup

    min_factor = Arg[float]('--min_factor', type=float, default=10.0) # the minimum LR factor, e.g. w/ 10, base 1e-4 -> 1e-5 for Cosine Decay
    disable_cosine_decay = Arg[int]('--disable_cosine_decay', action='store_true', default=False) # disable cosine decay

    training_steps = Arg[int]('--training_steps', type=int, default=1_000_000)
    log_eval_freq = Arg[int]('--log_eval_freq', type=int, default=100_000)

    pad_seq = Arg[int]('--pad_seq', action='store_true', default=False) # pad sequences to max length


    # evaluation
    eval_episodes = Arg[int]('--eval_episodes', type=int, default=10)
    eval_mode = Arg[str]('--eval_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    promptless_eval = Arg[int]('--promptless_eval', action='store_true', default=False)
    eval_text_num_examples = Arg[int]('--eval_text_num_examples', type=int, default=100)
    eval_text_log_examples = Arg[int]('--eval_text_log_examples', action='store_true', default=False) # for debugging if you wish to show predictions from model in eval for text

    # datasets / envs
    control_datasets = Arg[str]('--control_datasets', type=str, nargs='+', default=[])
    text_datasets = Arg[str]('--text_datasets', type=str, nargs='+', default=[]) # ['wikitext-2-v1']
    text_datasets_paths = Arg[str]('--text_datasets_paths', type=str, nargs='+', default=[]) # ['wikitext']

    # params for sampling from datasets
    prompt_ep_proportion = Arg[float]('--prompt_ep_proportion', type=float, default=0.25) # proportion of episodes that are prompted
    prompt_len_proportion = Arg[float]('--prompt_len_proportion', type=float, default=0.5) # proportion of context consumed by prompt
    unique_prompt_episodes = Arg[int]('--unique_prompt_episodes', default=False, action='store_true')
    top_k = Arg[int]('--top_k', type=int, default=None) # sample prompts only from top k episodes

    # logging
    use_wandb = Arg[int]('--use_wandb', '-w', action='store_true', default=False)
    wandb_project = Arg[str]('--wandb_project', type=str, default='gato-control')

    # saving
    save_model = Arg[int]('--save_model', action='store_true', default=False)
    save_mode = Arg[str]('--save_mode', type=str, default='last', choices=['checkpoint', 'last']) # Checkpoit saves model every after each log_eval_freq steps
    save_dir = Arg[str]('--save_dir', type=str, default='models')

if __name__ == '__main__':
    parser = ParseArger(namespace=TrainArgs())
    args = parser.parse_args()

    # Checks
    assert args.training_steps % args.log_eval_freq == 0, 'training_steps must be divisible by eval_freq'
    assert args.training_steps > args.warmup_steps, 'training_steps must be greater than warmup_steps'
    assert args.learning_rate > args.init_lr, 'learning_rate must be greater than init_lr'

    # make sure proportions are between 0 and 1
    assert 0 <= args.prompt_ep_proportion <= 1, 'prompt_ep_proportion must be between 0 and 1'
    assert 0 <= args.prompt_len_proportion <= 1, 'prompt_len_proportion must be between 0 and 1'


    main(args)
