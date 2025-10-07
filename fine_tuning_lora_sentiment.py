from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch

# --------------------------
# 1. Load dataset
# --------------------------
raw = load_dataset("rotten_tomatoes")

# smaller subset for faster training .... here we are training the dataset using smaller number of
# records for training (the dataset actually has 8536 records), we are using small number of validation
# examples

raw["train"] = raw["train"].select(range(1000))
raw["validation"] = raw["validation"].select(range(200))

# --------------------------
# 2. Tokenizer... we are using the bert model, we are tokenizing it to see which tokenization rule
# the model uses.
# --------------------------
MODEL_CHECKPOINT = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# To tokenize the dataset for training,
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

encoded = raw.map(tokenize_fn, batched=True)

# --------------------------
# 3. Labels ... ID2LABEL is for Converting model output IDs → human-readable labels
#               LABEL2ID Converts human-readable labels → numeric IDs for training
# --------------------------
ID2LABEL = {0: "Negative", 1: "Positive", 2:"Neutral"}
LABEL2ID = {"Negative": 0, "Positive": 1, "Neutral":2}

# --------------------------
# 4. Model with LoRA
# This prepares DistilBERT to predict sentiment (or any 3-class task) on your dataset,
# using pre-trained language knowledge plus a classification layer for your specific labels.
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# Ensure classifier head is trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Apply LoRA on q_lin + v_lin
# This LoraConfig essentially creates a tiny trainable adapter inside your frozen model,
# affecting only the attention query/value layers, with a moderate rank, scaling, and
# some dropout to prevent overfitting.
# This allows fast, memory-efficient fine-tuning on small datasets.

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
    bias="none",
    task_type="SEQ_CLS"
)

# Applying the lora config to the model
model = get_peft_model(model, lora_config)

# --------------------------
# 5. TrainingArguments (new + old compatibility)
#  Training the model for efficient fine-tuning
# --------------------------
try:
    training_args = TrainingArguments(
        output_dir="lora-text-classifier",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",   # newer versions
        save_strategy="no",
        logging_steps=20,
        report_to="none"
    )
except TypeError:
    training_args = TrainingArguments(
        output_dir="lora-text-classifier",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        do_eval=True,                  # older versions fallback
        logging_steps=20
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    tokenizer=tokenizer
)

# --------------------------
# 6. Inference before training
# --------------------------
print("\nUntrained model predictions:")
texts = [
    "It was good.",
    "Not a fan, don't recommend.",
    "Better than the first one.",
    "This is not worth watching even once.",
    "This one is a pass."
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    print(f"{text} -> {ID2LABEL[pred]}")

# --------------------------
# 7. Train the model
# --------------------------
trainer.train()

# --------------------------
# 8. Inference after training
# --------------------------
print("\nTrained model predictions:")
for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    print(f"{text} -> {ID2LABEL[pred]}")
