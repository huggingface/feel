from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch

def train_dpo_model(model, dataset, learning_rate=5e-5, num_train_epochs=3, per_device_train_batch_size=16):
    """
    Trains a model using Direct Preference Optimization (DPO).

    Args:
        model: The language model to be trained.
        dataset: The dataset used for training, should be in Hugging Face Dataset format.
        learning_rate: Learning rate for the optimizer.
        num_train_epochs: Number of epochs to train.
        per_device_train_batch_size: Batch size per device during training.
    """
    model.train()
    
    training_args = TrainingArguments(
        output_dir="./dpo_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
    )
    
    trainer.train()
    
    return model

