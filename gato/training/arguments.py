from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field, fields
from typing import List, Literal, Optional

@dataclass
class TrainingArgs(ArgumentParser):
    # Accelerate args
    cpu: Optional[int] = field(default=False, metadata={"help": 'Force script to execute on CPU. Passed to Accelerator.'})
    mixed_precision: Optional[str] = field(
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp8"],
        metadata={"help": "Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU."},
    )

    # Input & tokenization
    sequence_length: Optional[int] = field(default=1024, metadata={"help": "Used as context_length in models and tokenizers."})
    patch_size: Optional[int] = field(default=16, metadata={"help": "Images are reshaped to be a multiple of patch_size."})
    resid_mid_channels: Optional[int] = field(default=128, metadata={"help": "Number of channels in residual MLP. (Passed as to `nn.Conv2d` as `out_channels`.)"})
    num_groups: Optional[int] = field(default=32) # GroupNorm groups in ResNet
    patch_position_vocab_size: Optional[int] = field(default=128)
    disable_patch_pos_encoding: Optional[int] = field(default=False)
    disable_inner_pos_encoding: Optional[int] = field(default=False)

    mu: Optional[int] = field(default=100) # mu-law encoding
    M: Optional[int]= field(default=256)

    #vocab_size: Optional[int]('--vocab_size', type=int, default=32000) # number of tokens from SentencePiece
    continuous_tokens: Optional[int] = field(default=1024, metadata={"help": "Number of tokens to use for continuous values (e.g. actions, observations)."})
    discrete_tokens: Optional[int] = field(default=1024, metadata={"help": "Number of tokens to use for discrete actions."})

    # transformer architecture hyperparameters
    tokenizer_model_name: Optional[str] = field(default='gpt2')
    pretrained_lm: Optional[str] = field(default=None, metadata={"help": "Initialize with a pretrained language model, overriding --embed-dim, --layers, --heads, --activation-fn"})
    flash: Optional[int] = field(default=False) # enable flash attention
    init_checkpoint: Optional[str] = field(default=None) # Will not override architecture, only load weights from Gato checkpoint

    embed_dim: Optional[int] = field(default=768)
    layers: Optional[int] = field(default=8)
    heads: Optional[int] = field(default=24)
    activation_fn: Optional[str] = field(default='gelu')
    #activation_fn: Optional[str]('--activation_fn', type=str, default='geglu')

    # PEFT hyperparameters
    lora: Optional[int] = field(default=False)
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)

    # training hyperparameters
    text_prop: Optional[float] = field(default=0.5) # proportion of text data in batch
    gradient_accumulation_steps: Optional[int] = field(default=1) # simulate larger batch size
    batch_size: Optional[int] = field(default=512)
    dropout: Optional[float] = field(default=0.1)

    beta_1: Optional[float] = field(default=0.9)
    beta_2: Optional[float] = field(default=0.95)
    adam_eps: Optional[float] = field(default=1e-8)
    weight_decay: Optional[float] = field(default=0.1)

    grad_norm_clip: Optional[float] = field(default=1.0)
    disable_grad_clip: Optional[int] = field(default=False)

    warmup_steps: Optional[int] = field(default=15000)
    init_lr: Optional[float] = field(default=1e-7) # starting LR for warmup
    learning_rate: Optional[float] = field(default=1e-4) # the maximum LR after warmup

    min_factor: Optional[float] = field(default=10.0) # the minimum LR factor, e.g. w/ 10, base 1e-4 -> 1e-5 for Cosine Decay
    disable_cosine_decay: Optional[int] = field(default=False) # disable cosine decay

    training_steps: Optional[int] = field(default=1_000_000)
    log_eval_freq: Optional[int] = field(default=100_000)

    pad_seq: Optional[int] = field(default=False) # pad sequences to max length


    # evaluation
    eval_episodes: Optional[int] = field(default=10)
    eval_mode: Literal["deterministic", "stochastic"] = field(default='deterministic')
    promptless_eval: Optional[int] = field(default=False)
    eval_text_num_examples: Optional[int] = field(default=100)
    eval_text_log_examples: Optional[bool] = field(default=False) # for debugging if you wish to show predictions from model in eval for text

    # datasets / envs
    control_datasets: Optional[List[str]] = field(default=[], metadata={"nargs": "+"})
    text_datasets: Optional[List[str]] = field(default=[], metadata={"nargs": "+"}) # ['wikitext-2-v1']
    text_datasets_paths: Optional[List[str]] = field(default=[], metadata={"nargs": "+"}) # ['wikitext']

    # params for sampling from datasets
    prompt_ep_proportion: Optional[float] = field(default=0.25) # proportion of episodes that are prompted
    prompt_len_proportion: Optional[float] = field(default=0.5) # proportion of context consumed by prompt
    unique_prompt_episodes: Optional[bool] = field(default=False)
    top_k: Optional[int] = field(default=None) # sample prompts only from top k episodes

    # logging
    use_wandb: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default='gato-control')

    # saving
    save_model: Optional[int] = field(default=False)
    save_mode: Optional[List[Literal['checkpoint', 'last']]] = ['last'] # Checkpoit saves model every after each log_eval_freq steps
    save_dir: Optional[str] = field(default='models')

