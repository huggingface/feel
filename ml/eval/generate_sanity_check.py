"""
This script loads a fine tuned model and a reference model,
generates responses for some basic prompts for sanity check testing the the fined tuned model is better.
"""


import torch
from dataclasses import dataclass

from accelerate import PartialState
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, maybe_unpair_preference_dataset, setup_chat_format
from tqdm import tqdm
import json


####################################
#  ARGS
####################################


ref_model_args = ModelConfig(
    model_name_or_path="trl-lib/qwen1.5-1.8b-sft",
)

model_args = ModelConfig(
    model_name_or_path="../kto_nov_2",
)

# set `device` to "cuda" if a GPU is available. otherwise, defaults to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

output_file_path = 'generate_sanity_check.json'


####################################
#  LOAD REFERENCE MODEL & TOKENIZER
####################################

# load model
ref_model = AutoModelForCausalLM.from_pretrained(
    ref_model_args.model_name_or_path, trust_remote_code=ref_model_args.trust_remote_code
).to("cuda")
print(f'loaded reference model')

# load a tokenizer
ref_tokenizer = AutoTokenizer.from_pretrained(
    ref_model_args.model_name_or_path, trust_remote_code=ref_model_args.trust_remote_code
)

if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
print(f'loaded reference tokenizer')


####################################
#  LOAD FINE-TUNED MODEL & TOKENIZER
####################################


# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, use_auth_token=True).to("cuda")
print(f'loaded new model')

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_auth_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'loaded new tokenizer')


####################################
#  PROMPTS
####################################
prompts = [
    "Tell me a joke.",
]


####################################
#  GENERATE RESPONSES
####################################


for ix in range(len(prompts)):
    prompt = prompts[ix]

    # Generate reference model output
    ref_inputs = ref_tokenizer(prompt, return_tensors="pt").to("cuda")
    ref_output_ids = ref_model.generate(**ref_inputs)
    ref_output = ref_tokenizer.batch_decode(ref_output_ids, skip_special_tokens=True)[0]


    # Generate fine-tuned model output
    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    model_output_ids = model.generate(**model_inputs)
    model_output = tokenizer.batch_decode(model_output_ids, skip_special_tokens=True)[0]

    # print responses
    print("PROMPT:")
    print(f'{prompt}\n')

    print("REFERENCE MODEL RESPONSE:")
    print(f'{ref_output}\n')

    print("FINE-TUNED MODEL RESPONSE:")
    print(f'{model_output}\n')


# save results in json files
results = {}
results['prompt'] = prompt
results['ref_output'] = ref_output
results['fine_tuned_output'] = model_output
with open(output_file_path, "w") as f:
    json.dump(results, f, indent=4)

print('GENERATION COMPLETED.')
