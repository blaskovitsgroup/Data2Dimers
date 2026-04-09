import os
import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer, RobertaForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    EarlyStoppingCallback
)

output_dir = os.path.expanduser("PATH_TO/chemberta_380k/")

# Pretrained model and tokenizer from seyonec, saved locally
tokenizer = RobertaTokenizer.from_pretrained("../seyonec_chemberta")
model = RobertaForMaskedLM.from_pretrained("../seyonec_chemberta")

# Load dataset and split into train/validation (5%)
dataset = load_dataset('text', data_files= '380k_smiles_data.txt')
train_val = dataset['train'].train_test_split(test_size=0.05, seed=42)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenize and remove original text column
tokenized_datasets = train_val.map(tokenize_function, batched=True, remove_columns=["text"])

# Shuffle training set for better randomness
tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(seed=42)

# Data collator with MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    eval_steps=5000,
    save_steps=5000,
    save_total_limit=2,
    logging_steps=500,
    logging_dir=os.path.join(output_dir, "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42
)

# Define Trainer with EarlyStopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Perform training
trainer.train()

# Save tokenizer and model
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

