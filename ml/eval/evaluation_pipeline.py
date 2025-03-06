###########
# IMPORTS #
###########
from reward_eval import process_evaluation
from generate import generate_files
from alpaca import alpaca_evaluator, judge_responses
from bt import bradley_terry_comparison, load_rewards
from evaluate_arguments import EvalArguments
import pandas as pd
import numpy as np

#############
# EVALUATOR #
#############
'''
Evaluation Pipeline 

Parameters: 
eval_dataset: list of dictionaries that contain the prompt and response in the same form as below: 
            [{"prompt": "How are you?", "output": "I'm doing great!"}, {"prompt": "What's your name?", "output": "Assistant"}]
reward_output_filepath: string (must end in .json) that represents the path of the output of the reward score evaluation
model: base model that is being evaluated (defaults to starter base model - Aya-23-8B )
all_responses: should be a path to a csv file that has all the model's responses and their corresponding prompts with the following
                format: response1 --> col 1, response2 --> col 2, prompt --> col 3

language: which language is being used for this model (needs to be a valid FeeLLanguage object once FeeLLanguage class is updated)
'''
def evaluator_master_fn(eval_dataset: list[dict], 
                        reward_output_filepath: str, 
                        all_responses: str, 
                        language: str, 
                        new_model,
                        old_model="CohereForAI/aya-expanse-8b"): 
    # language is string for now, will be an object later with FeeLLanguage class definition with specific lanugage 
    # functionalities (will also store latest model and be much easier to handle such functions) 
    
    # 1. Reward score evaluation: 
    args = EvalArguments(bfloat16=True, 
                         reward_output_fmt='1-0', 
                         apply_sigmoid_to_reward=False,
                         per_device_batch_size=8,
                         output_filepath="new_evaluation",
                         result_filename=None,
                         model_name_or_path=new_model)
    reward_score_result = process_evaluation(args, model_name=new_model, eval_data_list_dict=eval_dataset)

    # 2. Alpaca Eval - Judging Responses 
    judge_df = pd.read_csv(all_responses)
    judge_df["winner"] = judge_df.apply(lambda r: judge_responses(r["response1"], r["response2"], r["prompt"]), axis = 1) # axis = 1 -- loops rows

    # 3. Alpaca Eval - model comparison 
    alpaca_results = alpaca_evaluator(new_model, num_samples=200) # can adjust num_samples as needed, potentially based on language

    # 4. Bradley Terry Evaluation
    bt_results = bradley_terry_comparison(load_rewards(old_model), load_rewards(new_model))

    return reward_score_result, judge_df, alpaca_results, bt_results

