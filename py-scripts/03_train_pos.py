import os
import numpy as np
import json
import datasets
import torch
import huggingface_hub
from transformers import(
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report
import evaluate

# Labels
with open('/work/Ccp-HERMOD/data/pos_label2id.json', 'r', encoding='utf-8') as f:
    pos_label2id = json.load(f)

with open('/work/Ccp-HERMOD/data/pos_id2label.json', 'r', encoding='utf-8') as file:
    pos_id2label_raw = json.load(file)
    pos_id2label = {int(k): v for k, v in pos_id2label_raw.items()}

# Model
model_checkpoint = 'CALDISS-AAU/DA-BERT_Old_News_NER'
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(pos_label2id), 
    ignore_mismatched_sizes=True
    )

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


model.config.label2id = pos_label2id
model.config.id2label = pos_id2label

# Importing data
tokenized_pos = datasets.load_from_disk('/work/Ccp-HERMOD/data/tokenized_pos')
tokenized_pos = tokenized_pos.remove_columns(["tokens", "pos_tags"])

# Training args
args = TrainingArguments(
    "/work/Ccp-HERMOD/modelling/DA-OldNews-BERT-VerbPOS-v1",
    eval_strategy = "epoch",
    learning_rate = 1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    remove_unused_columns=False,
)

# Data Collator
data_collator = \
    DataCollatorForTokenClassification(tokenizer)

# Metrics
metric = evaluate.load('seqeval')

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [pos_id2label[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [pos_id2label[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_pos['train'],
    eval_dataset=tokenized_pos['test'],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print(model.config.num_labels)
print(set(label for batch in tokenized_pos["train"]["labels"] for label in batch))

# Train plz
trainer.train()

