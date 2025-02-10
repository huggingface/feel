from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from ipdb import set_trace as st



def process_feel_dataset():
    """
    Processes the feel dataset into a format suitable for KTO training using TRL.

    Args:
        data (list): A list of dictionaries containing conversation data.

    Returns:
        dict: A dictionary containing the 'train' and 'test' splits of the dataset in KTO format, as Hugging Face Dataset objects.
    """

    # Load feel dataset
    # Load the JSON file
    file_path = "../data/example_data.json"
    with open(file_path, "r") as file:
        feel_dataset = json.load(file)


    kto_data = []

    # Function to transform a single conversation into KTO format
    def transform_conversation(entry):
        conversation = entry["conversation"]
        data_points = []
        user_timestamp = None

        for i in range(len(conversation)):
            message = conversation[i]
            if message["role"] == "user":
                user_timestamp = entry["timestamp"]
            if (
                message["role"] == "assistant" and
                message["rating"] in [1, -1]  # Only process feedback with positive or negative ratings
            ):
                user_content = conversation[i - 1]["content"] if i > 0 and conversation[i - 1]["role"] == "user" else ""
                data_points.append({
                    "prompt": user_content.strip(),
                    "completion": message["content"].strip(),
                    "label": message["rating"] == 1,  # True for positive feedback, False for negative (KTO Trainer format)
                    "timestamp": user_timestamp,
                    "session_id": entry["session_id"],
                    "conversation_id": entry["conversation_id"]
                })
        return data_points

    # Process all conversations in the dataset
    for entry in feel_dataset:
        kto_data.extend(transform_conversation(entry))

    # Convert to DataFrame
    kto_df = pd.DataFrame(kto_data)

    # Split into train and test sets (70% train, 30% test)
    train_df, test_df = train_test_split(kto_df, test_size=0.3, random_state=42)

    # Reset index to remove '__index_level_0__'
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return {"train": train_dataset, "test": test_dataset}




# def process_dataset_ultrafeedback():
#     """
#     Processes the 'train_prefs' and 'test_prefs' splits of the 'HuggingFaceH4/ultrafeedback_binarized' dataset
#     into a unified format for preference modeling.

#     Returns:
#         dict: A dictionary containing the unified 'train' and 'test' splits of the dataset in the KTO format.
#               Each split is a Hugging Face Dataset object.
#     """
#     # Load the relevant splits of the dataset
#     dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
#     train_prefs = load_dataset(dataset_name, split="train_prefs")
#     test_prefs = load_dataset(dataset_name, split="test_prefs")

#     # Function to transform a single example into the desired schema
#     def transform_data(example):
#         data_points = []
#         # Chosen completion
#         chosen_completion = example["chosen"][1]["content"]
#         if chosen_completion.strip():  # Check for non-empty completions
#             data_points.append({
#                 "prompt": example["prompt"],
#                 "completion": chosen_completion.strip(),
#                 "label": True
#             })
#         # Rejected completion
#         rejected_completion = example["rejected"][1]["content"]
#         if rejected_completion.strip():  # Check for non-empty completions
#             data_points.append({
#                 "prompt": example["prompt"],
#                 "completion": rejected_completion.strip(),
#                 "label": False
#             })
#         return data_points

#     # Process train and test splits
#     train_data = []
#     test_data = []

#     for example in train_prefs:
#         train_data.extend(transform_data(example))

#     for example in test_prefs:
#         test_data.extend(transform_data(example))

#     # Convert unified data to DataFrames
#     train_df = pd.DataFrame(train_data)
#     test_df = pd.DataFrame(test_data)


#     # Convert to Hugging Face Dataset
#     unified_train = Dataset.from_pandas(train_df)
#     unified_test = Dataset.from_pandas(test_df)

#     return {"train": unified_train, "test": unified_test}


if __name__ == "__main__":
    kto_dataset = process_feel_dataset()
    st()
