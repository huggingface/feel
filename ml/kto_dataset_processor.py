from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from ipdb import set_trace as st
from transformers import AutoTokenizer
from enum import Enum

class SupportedLanguages(str, Enum):
    """Enumeration of supported languages"""
    ENGLISH = "English"
    DUTCH = "Dutch"
    ITALIAN = "Italian"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    PORTUGUESE = "Portuguese"
    RUSSIAN = "Russian"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"

def transform_conversation(
    entry: dict,
    model_name: str,
    max_history_turns: int = 10,
    max_history_tokens: int = 4000
) -> list:
    """Transform conversation into KTO format with history"""
    data_points = []
    conversation = entry["conversation"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    for i, message in enumerate(conversation):
        # Only create data points for assistant messages that have ratings
        if message["role"] != "assistant" or message["rating"] not in [1, -1]:
            continue

        # Get previous messages up to limits
        formatted_history = []
        formatted_prompt = ""
        tokens = 0
        pairs = 0  # Count complete user/assistant pairs

        # Start from the current message and work backwards
        current_idx = i - 1
        while current_idx >= 0 and pairs < max_history_turns:
            # We need both user and assistant messages to form a pair
            if current_idx > 0 and conversation[current_idx]["role"] == "user" and conversation[current_idx-1]["role"] == "assistant":
                # Add the pair to history
                formatted_history.insert(0, conversation[current_idx-1])  # assistant
                formatted_history.insert(1, conversation[current_idx])    # user

                # Check token limit
                try:
                    current_formatted = tokenizer.apply_chat_template(formatted_history, tokenize=False)
                    current_tokens = len(tokenizer.encode(current_formatted))

                    if current_tokens > max_history_tokens:
                        formatted_history = formatted_history[2:]  # Remove the oldest pair
                        break

                    formatted_prompt = current_formatted
                    tokens = current_tokens
                    pairs += 1
                    current_idx -= 2
                except Exception:
                    # If template application fails, remove the last added pair
                    formatted_history = formatted_history[2:]
                    break
            else:
                current_idx -= 1

        # Add the final user message that prompted the rated response
        if i > 0 and conversation[i-1]["role"] == "user":
            last_history = formatted_history + [conversation[i-1]]
            try:
                formatted_prompt = tokenizer.apply_chat_template(last_history, tokenize=False)
            except Exception:
                # If template application fails, use the previous valid prompt
                pass

        data_points.append({
            "prompt": formatted_prompt.strip(),
            "completion": message["content"].strip(),
            "label": message["rating"] == 1,
            "timestamp": entry["timestamp"],
            "session_id": entry["session_id"],
            "conversation_id": entry["conversation_id"],
            "language": entry["language"]
        })

    return data_points

def process_feel_dataset(
    language: str,
    model_name: str = "CohereForAI/aya-expanse-8b",
    max_history_turns: int = 10,
    max_history_tokens: int = 4000
):
    """
    Processes the feel dataset into a format suitable for KTO training using TRL.

    Args:
        language: Language to filter the dataset for (must be one of SupportedLanguages)
        model_name: Name of the model to format for
        max_history_turns: Maximum number of previous turns to include in history
        max_history_tokens: Maximum number of tokens allowed in history

    Returns:
        dict: A dictionary containing the 'train' and 'test' splits of the dataset in KTO format

    Raises:
        ValueError: If language is not provided or not in SupportedLanguages
    """
    # Validate language
    if not language:
        raise ValueError("Language parameter is required")

    try:
        # Validate that it's a supported language
        SupportedLanguages(language)
    except ValueError:
        supported_langs = "\n- ".join([lang.value for lang in SupportedLanguages])
        raise ValueError(
            f"Invalid language: '{language}'\n"
            f"Supported languages are:\n- {supported_langs}"
        )

    # Load feel dataset from HuggingFace
    feel_dataset = load_dataset("feel-fl/feel-feedback")["train"]

    # Filter dataset by language
    feel_dataset = feel_dataset.filter(lambda x: x["language"] == language)

    if len(feel_dataset) == 0:
        raise ValueError(f"No data found for language: {language}")

    kto_data = []

    # Process all conversations in the filtered dataset
    for entry in feel_dataset:
        kto_data.extend(transform_conversation(
            entry,
            model_name,
            max_history_turns,
            max_history_tokens
        ))

    if len(kto_data) == 0:
        raise ValueError(f"No valid training examples found for language: {language}")

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

    print(f"Processed {len(kto_data)} examples for language: {language}")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return {"train": train_dataset, "test": test_dataset}

if __name__ == "__main__":
    # Process the dataset
    datasets = process_feel_dataset("English")

    # Print distribution of positive/negative labels
    train_labels = datasets['train']['label']
    test_labels = datasets['test']['label']

    print("\nLabel Distribution:")
    print("Train set:")
    print(f"Positive feedback: {sum(train_labels)}")
    print(f"Negative feedback: {len(train_labels) - sum(train_labels)}")
    print(f"Positive ratio: {sum(train_labels)/len(train_labels):.2%}")

    print("\nTest set:")
    print(f"Positive feedback: {sum(test_labels)}")
    print(f"Negative feedback: {len(test_labels) - sum(test_labels)}")
    print(f"Positive ratio: {sum(test_labels)/len(test_labels):.2%}")

    # Load original FEEL dataset
    feel_dataset = load_dataset("feel-fl/feel-feedback", split="train")

    # Print one original conversation
    print("\nOriginal conversation from FEEL dataset:")
    print(json.dumps(feel_dataset[0], indent=2))

    # Print sample entries from processed dataset
    print("\nSample entries from processed KTO dataset:")
    print("\n" + "="*80 + "\nTRAIN SET SAMPLES\n" + "="*80)

    # Export datasets to CSV
    train_df = datasets['train'].to_pandas()
    test_df = datasets['test'].to_pandas()

    train_df.to_csv('kto_train_dataset.csv', index=False)
    test_df.to_csv('kto_test_dataset.csv', index=False)

    print("\nDatasets exported to 'kto_train_dataset.csv' and 'kto_test_dataset.csv'")
