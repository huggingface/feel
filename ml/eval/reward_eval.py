import sys
import os
from typing import Any, Dict, List
import json 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CohereConfig, AutoModel
from accelerate import Accelerator
from tqdm import tqdm

# Add script directory to system path for importing local modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from eval.utils import jload, jdump
from eval.evaluate_arguments import EvalArguments


# set `device` to "cuda" if a GPU is available. otherwise, defaults to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_model(model_name: str):
    """
    loads pre-trained reward model and moves it onto device
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", num_labels=1).to("cuda")
    return model


def create_tokenizer(model_name):
    # loads the tokenizer that pairs with the model for encoding the text data
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    return tokenizer


def MyAccelerator(mixed_precision: str):
    """
    accelerator initialization (wrapper) for handling mixed precision
    """
    return Accelerator(mixed_precision=mixed_precision)
    
def get_reward_output_fn(reward_output_format: str, sigmoid: bool):
    def default(x): 
        return x.squeeze().cpu().detach().numpy().tolist()
    reward_fn_map = {
        '0': lambda x: x.squeeze().cpu().detach().softmax(dim=-1).numpy()[0].tolist(),
        '1': lambda x: x.squeeze().cpu().detach().softmax(dim=-1).numpy()[1].tolist(),
        '1-0': lambda x: (x.squeeze().cpu().detach().softmax(dim=-1).numpy()[1] - x.squeeze().cpu().detach().softmax(dim=-1).numpy()[0]).tolist()
    }
    reward_output_fn = reward_fn_map.get(reward_output_format, default)
    if sigmoid: 
        return lambda x: torch.sigmoid(torch.tensor(x)).numpy().tolist()
    return reward_output_fn

def evaluate_data(args, model, tokenizer, eval_data_list_dict) -> List[Dict[str, Any]]:
    """
    Evaluate the dataset using the reward model.
    """
    reward_output_fn = get_reward_output_fn(args.reward_output_fmt, args.apply_sigmoid_to_reward)
    pbar = tqdm(total=len(eval_data_list_dict), desc="Evaluating Rewards")
    rewards_list = []

    for idx in range(0, len(eval_data_list_dict), args.per_device_batch_size):
        batch_list_dict = eval_data_list_dict[idx:idx+args.per_device_batch_size]

        # Create prompt-response pairs
        batch_full_outputs = [
            f"{l['prompt']} {l['output']}" for l in batch_list_dict
        ] if 'prompt' in batch_list_dict[0] else [f"Below is an instruction: {l['instruction']} Response: {l['output']}" for l in batch_list_dict]

        # Tokenize reponse and send to device
        encoded_full_responses = tokenizer(batch_full_outputs, return_tensors="pt", padding=True, truncation=True)
        encoded_full_responses = encoded_full_responses.to(model.device)

        # Generate rewards
        with torch.inference_mode():
            reward_outputs = model(**encoded_full_responses)
            rewards = reward_output_fn(reward_outputs.logits)
            rewards_list.extend(rewards)

        pbar.update(len(batch_list_dict))

    # Adding reward scores to original data
    for i, data in enumerate(eval_data_list_dict):
        data['reward'] = rewards_list[i]

    return eval_data_list_dict

def process_evaluation(args, model_name: str, eval_data_list_dict) -> List[Dict[str, Any]]:
    """
    Main function for processing evaluation, takes model name as input.
    """
    # mixed_precision = 'bf16' if args.bfloat16 else 'fp16'
    
    # Initialize accelerator and model
    # accelerator = MyAccelerator(mixed_precision)
    model = create_model(model_name)
    tokenizer = create_tokenizer(model_name)

    model.eval()

    eval_data = evaluate_data(args, model, tokenizer, eval_data_list_dict)

    result_filename = args.result_filename or f"{os.path.basename(args.output_filepath).split('.')[0]}_reward_results.json"
    with open(result_filename, "w") as f:
        json.dump(eval_data, f)

    return eval_data


# ONLY FOR TESTING: 
if __name__ == '__main__':
    args = EvalArguments(bfloat16=True, 
                         reward_output_fmt='1-0', 
                         apply_sigmoid_to_reward=False,
                         per_device_batch_size=8,
                         output_filepath= '/path/to/your/data.json',
                         result_filename=None,
                         model_name_or_path="CohereForAI/aya-expanse-8b")


    eval_data_list_dict = [{"prompt": "How are you?", "output": "I'm doing great!"}, {"prompt": "What's your name?", "output": "Assistant"}]

    process_evaluation(args, model_name="CohereForAI/aya-expanse-8b", eval_data_list_dict=eval_data_list_dict)