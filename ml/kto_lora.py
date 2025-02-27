import os
import torch
from dataclasses import dataclass
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format
from kto_dataset_processor import process_feel_dataset, SupportedLanguages
from datetime import datetime
import wandb
from enum import Enum
from typing import Optional


# PEFT library: attach and load adapters
from peft import get_peft_model, PeftModel

####################################
#  CONFIGURATION
####################################


@dataclass
class ScriptArguments:
    """
    Configuration for the script.
    """
    process_dataset_func: callable = process_feel_dataset
    checkpoint_path: str = None
    push_to_hub: bool = True
    language: str = "English"  # Default to English

    def __post_init__(self):
        """Validate the language after initialization"""
        try:
            # This will raise ValueError if language is not in the enum
            SupportedLanguages(self.language)
        except ValueError:
            supported_langs = "\n- ".join([lang.value for lang in SupportedLanguages])
            raise ValueError(
                f"Invalid language: '{self.language}'\n"
                f"Supported languages are:\n- {supported_langs}"
            )

@dataclass
class ModelArguments(ModelConfig):
    """
    Configuration for the model.
    """
    model_name: str = "CohereForAI/aya-expanse-8b"
    use_peft: bool = True
    lora_target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 16
    trust_remote_code: bool = True

@dataclass
class TrainingArguments(KTOConfig):
    """
    Configuration for the KTO trainer.
    """
    output_dir: str = f"kto_{ModelArguments.model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-7
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    eval_steps: int = 500
    warmup_ratio: float = 0.1
    bf16: bool = True
    logging_first_step: bool = True

# Initialize configurations
script_args = ScriptArguments()
training_args = TrainingArguments()
model_args = ModelArguments()

####################################
#  HELPER FUNCTIONS
####################################

def load_model_and_tokenizer(model_args):
    """
    Load the base model and tokenizer from the Hugging Face Hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=model_args.trust_remote_code
    )

    # Set pad token if it is missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup chat format if not available on the tokenizer
    if not getattr(tokenizer, "chat_template", None):
        model, tokenizer = setup_chat_format(model, tokenizer)

    return model, tokenizer

####################################
#  MAIN LOGIC
####################################

def main():
    # Initialize wandb for logging
    wandb.init(project="kto")

    print("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args)
    ref_model, _ = load_model_and_tokenizer(model_args)
    print("Models and tokenizer loaded.")

    # -----------------------------
    # Adapter Loading or Initialization
    # -----------------------------
    # Configure the PEFT / LoRA adapter settings
    peft_config = get_peft_config(model_args)
    adapter_dir = os.path.join("adapters", script_args.language)

    if os.path.isdir(adapter_dir):
        # If an adapter for this language already exists, load it into the base model.
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"Loaded existing adapter for language '{script_args.language}' from {adapter_dir}.")
    else:
        # Otherwise, initialize a new LoRA adapter.
        model = get_peft_model(model, peft_config)
        print(f"No adapter found for language '{script_args.language}'. Initialized new adapter.")

    # -----------------------------
    # Data Preparation and Training
    # -----------------------------
    print("Processing dataset...")
    dataset = script_args.process_dataset_func(script_args.language)
    print("Dataset processed.")

    print("Initializing trainer...")
    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Training
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    # Evaluation
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(f"Metrics: {metrics}")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Log metrics to wandb
    wandb.log({
        "epoch": metrics.get("epoch"),
        "grad_norm": metrics.get("grad_norm"),
        "kl": metrics.get("kl"),
        "learning_rate": metrics.get("learning_rate"),
        "logits/chosen": metrics.get("logits/chosen"),
        "logits/rejected": metrics.get("logits/rejected"),
        "logps/chosen": metrics.get("logps/chosen"),
        "logps/rejected": metrics.get("logps/rejected"),
        "loss": metrics.get("loss"),
        "rewards/chosen": metrics.get("rewards/chosen"),
        "rewards/margins": metrics.get("rewards/margins"),
        "rewards/rejected": metrics.get("rewards/rejected"),
        "step": metrics.get("step")
    })

    # -----------------------------
    # Adapter Saving
    # -----------------------------
    print("Saving adapter...")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    print(f"Adapter for language '{script_args.language}' saved to: {adapter_dir}")

    if script_args.push_to_hub:
        # Using a consistent naming pattern that links to the FEEL project
        repo_id = f"feel-fl/kto-lora-adapter-{script_args.language}"
        print(f"Pushing adapter to Hugging Face Hub at {repo_id}...")
        model.push_to_hub(repo_id=repo_id)

    print("Process completed.")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
