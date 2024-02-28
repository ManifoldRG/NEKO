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

        save_mode: Literal['checkpoint', 'last'] = field(default="last")
            Checkpoint saves model every after each log_eval_freq steps

        save_dir: (str, default='models')
    """
    cpu: bool = field(default=False, metadata={"help": 'Whether to execute the script on the CPU or GPU/TPU. Gets passed to HuggingFace\'s Accelerator.'})
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

    # Mu-law companding. See section 2.1 Tokenization of the GATO paper.
    # Values described in the GATO paper (100, 256) cause a rapid ~convergence to values between -0.7 and 0.7.
    mu: int = field(default=100, metadata={'help': 'Higher values cause more rapid sloping.'})
    M: int = field(default=256, metadata={'help': 'Higher values cause smaller upper and lower bounds.'})

    continuous_tokens: int = field(default=1024, metadata={"help": "Number of tokens to use for continuous values (e.g. actions, observations)."})
    discrete_tokens: int = field(default=1024, metadata={"help": "Number of tokens to use for discrete actions."})

    # Transformer architecture hyperparameters
    tokenizer_model_name: str = field(default='gpt2')
    pretrained_lm: Optional[str] = field(default=None, metadata={"help": "Initialize with a pretrained language model, overriding --embed-dim, --layers, --heads, --activation-fn"})
    flash: bool = field(default=False, metadata={'help': "Enable flash attention."})
    init_checkpoint: Optional[str] = field(default=None, metadata={'help': 'Will not override architecture, only load weights from GATO checkpoint.'})

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
    text_prop: float = field(default=0.0, metadata={'help': 'Proportion of text data in batch.'})
    caption_prop: float = field(default=0.0, metadata={'help': 'Proportion of caption data in batch.'})
    vqa_prop: float = field(default=0.0, metadata={'help': 'Proportion of text vqa in batch.'})
    gradient_accumulation_steps: int = field(default=1, metadata={'help': 'Simulate a larger batch size.'})
    batch_size: int = field(default=512)
    dropout: float = field(default=0.1)

    beta_1: float = field(default=0.9)
    beta_2: float = field(default=0.95)
    adam_eps: float = field(default=1e-8)
    weight_decay: float = field(default=0.1)

    grad_norm_clip: float = field(default=1.0)
    disable_grad_clip: int = field(default=False)

    warmup_steps: int = field(default=15000)
    init_lr: float = field(default=1e-7)
    learning_rate: float = field(default=1e-4)

    min_factor: float = field(default=10.0, metadata={'help': 'The minimum LR factor, e.g. w/ 10, base 1e-4 -> 1e-5 for cosine decay.'})
    disable_cosine_decay: bool = field(default=False)

    training_steps: int = field(default=1_000_000)
    log_eval_freq: int = field(default=100_000)

    pad_seq: int = field(default=False)

    # Evaluation
    eval_episodes: int = field(default=10)
    eval_mode: Literal["deterministic", "stochastic"] = field(default='deterministic')
    promptless_eval: int = field(default=False)
    eval_text_num_examples: int = field(default=100)
    eval_text_log_examples: bool = field(default=False, metadata={'help': 'For debugging if you wish to show predictions from model in eval for text.'})

    # Datasets / envs
    control_datasets: List[str] = field(default_factory=list, metadata={'nargs': '+'})
    text_datasets: List[str] = field(default_factory=list, metadata={'nargs': '+', 'help': 'e.g. \'wikitext-2-v1\''})
    text_datasets_paths: List[str] = field(default_factory=list, metadata={'nargs': '+', 'help': 'e.g. \'wikitext\''})

    # Caption/VQA datasets
    caption_dataset: str = field(default='', metadata={'help': 'The directory for all of the data (training and test).'})
    caption_train_data: List[str] = field(default_factory=list, metadata={'nargs': '+', 'help': 'List of sub directories for training data.'})
    caption_test_data: List[str] = field(default_factory=list, metadata={'nargs': '+', 'help': 'List of sub directories for test data.'})
    test_data_prop: float = field(default=0.1, metadata={'help': 'The proportion of test data if needing to split training dataset into training and test.'})

    vqa_dataset: str = field(default='', metadata={'help': 'The directory for all of the data (training and test).'})
    vqa_train_data: List[str] = field(default_factory=list, metadata={'nargs': '+', 'help': 'List of sub directories for training data.'})
    vqa_test_data: List[str] = field(default_factory=list, metadata={'nargs': '+', 'help': 'List of sub directories for test data.'})
    train_img_name_prefix: List[str] = field(default_factory=list, metadata={'help': 'Each sub directory has one such image name file prefix.'})
    train_img_file_name_len: List[int] = field(default_factory=list, metadata={'help': 'Each sub directory has one such image name file length.'})
    test_img_name_prefix: List[str] = field(default_factory=list, metadata={'help': 'Each sub directory has one such image name file prefix.'})
    test_img_file_name_len: List[int] = field(default_factory=list, metadata={'help': 'Each sub directory has one such image file name length.'})
    questions_file: str = field(default='questions.json', metadata={'help': 'It is required to give the same name to all questions json files (no ambiguity since they are under different directories).'})
    annotations_file: str = field(default='annotations.json', metadata={'help': 'It is required to give the same name to all annotations json files (no ambiguity since they are under different directories).'})

    eval_caption_num_examples: int = field(default=100)
    eval_caption_log_examples: bool = field(default=False, metadata={'help': 'For debugging if you wish to show predictions from model in eval for text.'})

    eval_vqa_num_examples: int = field(default=100)
    eval_vqa_log_examples: bool = field(default=False, metadata={'help': 'For debugging if you wish to show predictions from model in eval for text.'})

    # Params for sampling from datasets
    prompt_ep_proportion: float = field(default=0.25, metadata={'help': 'Proportion of episodes that are prompted.'})
    prompt_len_proportion: float = field(default=0.5, metadata={'help': 'Proportion of context consumed by prompt.'})
    unique_prompt_episodes: bool = field(default=False)
    top_k: Optional[int] = field(default=None, metadata={'help': 'Sample prompts only from top k episodes.'})

    # Logging
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default='gato-control')

    # Saving
    save_model: bool = field(default=False)
    save_mode: Literal['checkpoint', 'last'] = field(default="last", metadata={'help': 'Checkpoint saves model after each log_eval_freq steps.'})
    save_dir: str = field(default='models')
