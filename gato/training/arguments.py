from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class TrainingArgs:
    # Pass this class to the initialization of
    # [TypedArgumentParser](/gato/utils/typed_argparser.py) to create an
    # [argparse.ArgumentParser](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser).
    # Then call `parser.parse_args_into_dataclasses()` to
    cpu: bool = field(default=False, metadata={"help": 'Force script to execute on CPU. Passed to Accelerator.'})
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = field(
        default="no",
        metadata={"help": "Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU."},
    )

    # Input & tokenization
    sequence_length: int = field(default=1024, metadata={"help": "Used as context_length in models and tokenizers."})
    patch_size: int = field(default=16, metadata={"help": "Images are reshaped to be a multiple of patch_size."})
    resid_mid_channels: int = field(default=128, metadata={"help": "Number of channels in residual MLP. (Passed as to `nn.Conv2d` as `out_channels`.)"})
    num_groups: int = field(default=32) # GroupNorm groups in ResNet
    patch_position_vocab_size: int = field(default=128)
    disable_patch_pos_encoding: int = field(default=False)
    disable_inner_pos_encoding: int = field(default=False)

    mu: int = field(default=100) # mu-law encoding
    M: Optional[int]= field(default=256)

    #vocab_size: Optional[int]('--vocab_size', type=int, default=32000) # number of tokens from SentencePiece
    continuous_tokens: int = field(default=1024, metadata={"help": "Number of tokens to use for continuous values (e.g. actions, observations)."})
    discrete_tokens: int = field(default=1024, metadata={"help": "Number of tokens to use for discrete actions."})

    # transformer architecture hyperparameters
    tokenizer_model_name: str = field(default='gpt2')
    pretrained_lm: Optional[str] = field(default=None, metadata={"help": "Initialize with a pretrained language model, overriding --embed-dim, --layers, --heads, --activation-fn"})
    flash: int = field(default=False) # enable flash attention
    init_checkpoint: Optional[str] = field(default=None) # Will not override architecture, only load weights from Gato checkpoint

    embed_dim: int = field(default=768)
    layers: int = field(default=8)
    heads: int = field(default=24)
    activation_fn: str = field(default='gelu')
    #activation_fn: Optional[str]('--activation_fn', type=str, default='geglu')

    # PEFT hyperparameters
    lora: int = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)

    # training hyperparameters
    text_prop: float = field(default=0.5) # proportion of text data in batch
    gradient_accumulation_steps: int = field(default=1) # simulate larger batch size
    batch_size: int = field(default=512)
    dropout: float = field(default=0.1)

    beta_1: float = field(default=0.9)
    beta_2: float = field(default=0.95)
    adam_eps: float = field(default=1e-8)
    weight_decay: float = field(default=0.1)

    grad_norm_clip: float = field(default=1.0)
    disable_grad_clip: int = field(default=False)

    warmup_steps: int = field(default=15000)
    init_lr: float = field(default=1e-7) # starting LR for warmup
    learning_rate: float = field(default=1e-4) # the maximum LR after warmup

    min_factor: float = field(default=10.0) # the minimum LR factor, e.g. w/ 10, base 1e-4 -> 1e-5 for Cosine Decay
    disable_cosine_decay: int = field(default=False) # disable cosine decay

    training_steps: int = field(default=1_000_000)
    log_eval_freq: int = field(default=100_000)

    pad_seq: int = field(default=False) # pad sequences to max length


    # evaluation
    eval_episodes: int = field(default=10)
    eval_mode: Literal["deterministic", "stochastic"] = field(default='deterministic')
    promptless_eval: int = field(default=False)
    eval_text_num_examples: int = field(default=100)
    eval_text_log_examples: bool = field(default=False) # for debugging if you wish to show predictions from model in eval for text

    # datasets / envs
    control_datasets: List[str] = field(default_factory=list, metadata={"nargs": "+"})
    text_datasets: List[str] = field(default_factory=list, metadata={"nargs": "+"}) # ['wikitext-2-v1']
    text_datasets_paths: List[str] = field(default_factory=list, metadata={"nargs": "+"}) # ['wikitext']

    # params for sampling from datasets
    prompt_ep_proportion: float = field(default=0.25) # proportion of episodes that are prompted
    prompt_len_proportion: float = field(default=0.5) # proportion of context consumed by prompt
    unique_prompt_episodes: bool = field(default=False)
    top_k: Optional[int] = field(default=None) # sample prompts only from top k episodes

    # logging
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default='gato-control')

    # saving
    save_model: int = field(default=False)
    save_mode: List[Literal['checkpoint', 'last']] = field(default_factory=lambda: ["last"]) # Checkpoit saves model every after each log_eval_freq steps
    save_dir: str = field(default='models')
