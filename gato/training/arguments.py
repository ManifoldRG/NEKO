from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class TrainingArgs:
    """TrainingArgs is a subset of the arguments we use in our example scripts
    which relate to the training loop itself.

    You can use this class as part of parsing command line arguments passing
    this class to the initialization of
    [TypedArgumentParser](/gato/utils/typed_argparser.py) to create an
    [argparse.ArgumentParser](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser)
    and then call `parser.parse_args_into_dataclasses()`.

    Parameters:
        cpu (bool, default=False):
            Whether to execute the script on the CPU or GPU/TPU. Gets passed through to [Accelerator](https://github.com/huggingface/accelerate/blob/403c0714d1dd019a481022afce5df75a9963ecd9/src/accelerate/accelerator.py#L175).

        mixed_precision (str, default="no"):
            Gets passed through to [Accelerator](https://github.com/huggingface/accelerate/blob/403c0714d1dd019a481022afce5df75a9963ecd9/src/accelerate/accelerator.py#L166)
            Possible values are:
                - "no"
                - "fp16"
                - "bf16"
                - "fp8"

    ### Input and tokenization
        sequence_length (int, default=16):

        patch_size (int default=16):
            Images are reshaped to be a multiple of this size.

        resid_mid_channels: (int, default=128):
            Number of channels in residual MLP. (Passed as to `nn.Conv2d` as `out_channels`.)

        num_groups: (int, default=32)
            GroupNorm groups in ResNet

        patch_position_vocab_size: (int, default=128)

        disable_patch_pos_encoding: (int, default=False)

        disable_inner_pos_encoding: (int, default=False)

        mu: (int, default=100) # mu-law encoding

        M: Optional[int]= field(default=256)

        continuous_tokens: (int, default=1024)
            Number of tokens to use for continuous values (e.g. actions, observations).

        discrete_tokens: (int, default=1024)
            Number of tokens to use for discrete actions.

    ### Transformer architecture hyperparameters:

        tokenizer_model_name: (str, default='gpt2')

        pretrained_lm: Optional[str] = field(default=None,
            Initialize with a pretrained language model, overriding --embed-dim, --layers, --heads, --activation-fn

        flash: (int, default=False) # enable flash attention

        init_checkpoint: Optional[str] = field(default=None)
            Will not override architecture, only load weights from Gato checkpoint

        embed_dim: (int, default=768)

        layers: (int, default=8)

        heads: (int, default=24)

        activation_fn: (str, default='gelu')

    ### PEFT hyperparameters

        lora: (int, default=False)
        lora_r: (int, default=8)
        lora_alpha: (int, default=32)
        lora_dropout: (float, default=0.1)

    ### Training hyperparameters

        text_prop: (float, default=0.5)
            proportion of text data in batch

        gradient_accumulation_steps: (int, default=1)
            simulate larger batch size

        batch_size: (int, default=512)

        dropout: (float, default=0.1)

        beta_1: (float, default=0.9)

        beta_2: (float, default=0.95)

        adam_eps: (float, default=1e-8)

        weight_decay: (float, default=0.1)

        grad_norm_clip: (float, default=1.0)

        disable_grad_clip: (int, default=False)

        warmup_steps: (int, default=15000)

        init_lr: (float, default=1e-7)
            starting LR for warmup

        learning_rate: (float, default=1e-4)
            the maximum LR after warmup

        min_factor: (float, default=10.0)
            the minimum LR factor, e.g. w/ 10, base 1e-4 -> 1e-5 for Cosine Decay

        disable_cosine_decay: (bool, default=False)
            disable cosine decay

        training_steps: (int, default=1_000_000)

        log_eval_freq: (int, default=100_000)

        pad_seq: (int, default=False)
            pad sequences to max length


    ### Evaluation

        eval_episodes: (int, default=10)

        eval_mode: Literal["deterministic", "stochastic"] = field(default='deterministic')

        promptless_eval: (int, default=False)

        eval_text_num_examples: (int, default=100)

        eval_text_log_examples: (bool, default=False)
            for debugging if you wish to show predictions from model in eval for text

    ### Datasets / envs

        control_datasets: (List[str], default_factory=list)

        text_datasets: (List[str], default_factory=list)
            ['wikitext-2-v1']

        text_datasets_paths: (List[str], default_factory=list)
            ['wikitext']

    ### Params for sampling from datasets

        prompt_ep_proportion: (float, default=0.25)
            proportion of episodes that are prompted

        prompt_len_proportion: (float, default=0.5)
            proportion of context consumed by prompt

        unique_prompt_episodes: (bool, default=False)

        top_k: Optional[int] = field(default=None)
            sample prompts only from top k episodes

    ### Logging

        use_wandb: (bool, default=False)

        wandb_project: (str, default='gato-control')

    ### Saving

        save_model: (bool, default=False)

        save_mode: Literal['checkpoint', 'last'] = field(default="last")
            Checkpoint saves model every after each log_eval_freq steps

        save_dir: (str, default='models')
    """
    cpu: bool = field(default=False, metadata={"help": 'Force script to execute on CPU. Passed to Accelerator.'})
    device: Literal["cpu", "xpu", "cuda", "mps", "npu"] = field(default="cpu", metadata={"help": "PyTorch device to run on."})
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = field(
        default="no",
        metadata={"help": "Whether to use mixed precision. Choose"
        "between no, fp16, bf16 (bfloat16), and fp8. Bf16 requires PyTorch >= 1.10."
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
    M: int = field(default=256)

    continuous_tokens: int = field(default=1024, metadata={"help": "Number of tokens to use for continuous values (e.g. actions, observations)."})
    discrete_tokens: int = field(default=1024, metadata={"help": "Number of tokens to use for discrete actions."})

    # Transformer architecture hyperparameters
    tokenizer_model_name: str = field(default='gpt2')
    pretrained_lm: Optional[str] = field(default=None, metadata={"help": "Initialize with a pretrained language model, overriding --embed-dim, --layers, --heads, --activation-fn"})
    flash: bool = field(default=False) # enable flash attention
    init_checkpoint: Optional[str] = field(default=None) # Will not override architecture, only load weights from Gato checkpoint

    embed_dim: int = field(default=768)
    layers: int = field(default=8)
    heads: int = field(default=24)
    activation_fn: str = field(default='gelu')

    # PEFT hyperparameters
    lora: int = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)

    # Training hyperparameters
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
    disable_cosine_decay: bool = field(default=False) # disable cosine decay

    training_steps: int = field(default=1_000_000)
    log_eval_freq: int = field(default=100_000)

    pad_seq: int = field(default=False) # pad sequences to max length

    # Evaluation
    eval_episodes: int = field(default=10)
    eval_mode: Literal["deterministic", "stochastic"] = field(default='deterministic')
    promptless_eval: int = field(default=False)
    eval_text_num_examples: int = field(default=100)
    eval_text_log_examples: bool = field(default=False) # for debugging if you wish to show predictions from model in eval for text

    # Datasets / envs
    control_datasets: List[str] = field(default_factory=list, metadata={"nargs": "+"})
    text_datasets: List[str] = field(default_factory=list, metadata={"nargs": "+"}) # ['wikitext-2-v1']
    text_datasets_paths: List[str] = field(default_factory=list, metadata={"nargs": "+"}) # ['wikitext']

    # Params for sampling from datasets
    prompt_ep_proportion: float = field(default=0.25) # proportion of episodes that are prompted
    prompt_len_proportion: float = field(default=0.5) # proportion of context consumed by prompt
    unique_prompt_episodes: bool = field(default=False)
    top_k: Optional[int] = field(default=None) # sample prompts only from top k episodes

    # Logging
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default='gato-control')

    # Saving
    save_model: bool = field(default=False)
    save_mode: Literal['checkpoint', 'last'] = field(default="last") # Checkpoint saves model every after each log_eval_freq steps
    save_dir: str = field(default='models')
