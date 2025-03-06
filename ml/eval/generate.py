from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, maybe_unpair_preference_dataset, setup_chat_format
from tqdm import tqdm
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

####################################
#  CONFIGURATION
####################################

@dataclass
class ScriptArguments:
    """
    The arguments for the script.
    """
    dataset_name: str = "OpenAssistant/oasst1"
    kto_model_path: str = "mistralai/Mistral-7B-v0.1"
    kto_output_file: str = "kto_generations_mini.json"
    sft_output_file: str = "sft_generations_mini.json"


# Initialize arguments
script_args = ScriptArguments()

# Set `device` to "cuda" if available, otherwise "cpu"
# If you don't want this to run on GPU set device = "cpu"

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

####################################
#  UTILITY FUNCTIONS
####################################

def format_prompt(prompt):
    """
    Convert a conversation (list of dictionaries) into a string format suitable for the tokenizer.
    """
    return "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in prompt])

def load_model_and_tokenizer(model_path, trust_remote_code=False, use_auth_token=False):
    """Load a model and its tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_auth_token=use_auth_token,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, use_auth_token=use_auth_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Setup chat format if not present
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer

def generate_responses(model, tokenizer, dataset, num_examples=None):
    """Generate responses for a dataset using a given model and tokenizer."""
    results = []

    # Limit dataset to num_examples if specified
    items = list(dataset.data.items())
    if num_examples is not None:
        items = items[:num_examples]

    for prompt, key in tqdm(items):
        prompt = tokenizer.apply_chat_template(key.prompt, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=4000)
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Keys are in alpacaeval format
        results.append({
            "instruction": prompt,
            "output": output
        })
    return results


def load_oasst_test_dataset():
    """Load and prepare the dataset."""

    # Load oasst test dataset
    test_dataset = load_dataset(split='test')
    return test_dataset


def prepare_oasst_sft_results(test_dataset, tokenizer, num_examples=None):
    """
    Prepare SFT results for a test dataset using a tokenizer.

    Parameters:
    - test_dataset: The dataset containing prompts and keys.
    - tokenizer: The tokenizer to process inputs and outputs.
    - num_examples: Optional; the number of examples to process.
                    If None, process the entire dataset.
    """
    sft_results = []
    # Limit dataset to num_examples if specified
    items = list(test_dataset.data.items())
    if num_examples is not None:
        items = items[:num_examples]

    for prompt, key in items:  # Iterate over limited dataset
        for i, j in key.pairs:  # Process each preference pair
            # Add prompt and corresponding chosen/rejected completions
            prompt = tokenizer.apply_chat_template(key.prompt, tokenize=False)
            output = key.generations[key.sft_index]

            # Keys are in alpacaeval format
            sft_results.append({
                "instruction": prompt,
                "output": output
            })
    return sft_results


def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


####################################
#  MAIN SCRIPT
####################################

def generate_files():
    # Load model and tokenizer
    print("Loading kto fine-tuned model...")
    kto_model, kto_tokenizer = load_model_and_tokenizer(script_args.kto_model_path, use_auth_token=True)
    print("kto fine-tuned model loaded.")

    # Load dataset
    print("Loading dataset...")
    test_dataset = load_oasst_test_dataset()
    print("Dataset loaded.")


    # Generate responses for reference model
    print("Generating responses for kto model...")
    kto_results = generate_responses(kto_model, kto_tokenizer, test_dataset, num_examples=10)
    save_results(kto_results, script_args.kto_output_file)

    # Generate SFT responses file
    print("Generating SFT responses file...")
    sft_results = prepare_oasst_sft_results(test_dataset, kto_tokenizer, num_examples=10)
    save_results(sft_results, script_args.sft_output_file)
    print("GENERATION COMPLETED.")


if __name__ == "__main__":
    generate_files()
