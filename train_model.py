import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

# Set cache directories (optional)
os.environ["HF_HOME"] = "E:/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "E:/huggingface"
os.environ["TORCH_HOME"] = "E:/torch_cache"

# 1. Load dataset (limit 1000 lines from your preprocessed JSON)
data = []
with open("processed/processed_dataset.json", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 1000:
            break
        data.append(json.loads(line))

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# 2. Load DialoGPT-small tokenizer and model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # <--- IMPORTANT: set pad_token here

model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Tokenization function
def tokenize(batch):
    inputs = [f"User: {inp}\nBot: {out}" for inp, out in zip(batch["input"], batch["output"])]
    encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    labels = encodings["input_ids"].copy()
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in seq] for seq in labels]
    encodings["labels"] = labels
    return encodings

# Map the dataset with tokenization (batched)
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# 4. Data collator for causal LM (no MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

# 7. Resume training if checkpoint exists
latest_checkpoint = None
if os.path.exists("./model"):
    checkpoints = [d for d in os.listdir("./model") if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = os.path.join("./model", checkpoints[-1])

if latest_checkpoint:
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    trainer.train()

# 8. Save final model and tokenizer in your project root directory
final_model_path = "./dialoGPT_finetuned"
os.makedirs(final_model_path, exist_ok=True)
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"✅ DialoGPT training complete. Model saved to {final_model_path}")
