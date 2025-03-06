from dataclasses import dataclass, field

@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        default="CohereForAI/aya-expanse-8b", metadata={"help": "Name to a huggingface native pretrained model or path to a model on disk."})
    model_pretrained_lora_weights: str = field(
        default=None, metadata={"help": "Path to a checkpoint directory."})
    output_filepath: str = field(
        default="rewards_examples_idan_mini.json", metadata={"help": "Path to the decode result or to a dir containing such files."}) # ADD output filepath
    result_filename: str = field(
        default=None, metadata={"help": "The path to the result json file. If not provided, will automatically create one. "})
    per_device_batch_size: int = field(
        default=12, metadata={"help": "The path to the output json file."})
    flash_attn: bool = field(default=False, metadata={"help": "If True, uses Flash Attention."})
    bfloat16: bool = field(
        default=False, metadata={"help": "If True, uses bfloat16. If lora and four_bits are True, bfloat16 is used for the lora weights."})

    # peft / quantization
    use_lora: bool = field(default=False, metadata={"help": "If True, uses LoRA."})
    load_in_4bit: bool = field(default=False, metadata={"help": "If True, uses 4-bit quantization."})
    load_in_8bit: bool = field(default=False, metadata={"help": "If True, uses 8-bit quantization."})

    # reward model specific args
    reward_output_fmt: str = field(default=None, metadata={"help": "If 0, takes the softmax-ed output at index 0. If 1-0, takes the softmax-ed output at index 1 - index 0. Otherwise, just takes the raw output."})
    soft_preference: bool = field(default=False, metadata={"help": "If True, uses soft preference."})
    apply_sigmoid_to_reward: bool = field(default=False, metadata={"help": "If True, applies sigmoid to the reward."})

    transformer_cache_dir: str = field(
        default=None,
        metadata={
            "help": "Path to a directory where transformers will cache the model. "
            "If None, transformers will use the default cache directory."
        },)
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "If True, enables unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})

    def __post_init__(self):
        # separate multiple model names or paths by comma
        if self.model_name_or_path is not None:
            self.model_name_or_path = self.model_name_or_path.split(',')

            # if loading 1 model, convert to string like normal
            if len(self.model_name_or_path) == 1:
                self.model_name_or_path = self.model_name_or_path[0]
