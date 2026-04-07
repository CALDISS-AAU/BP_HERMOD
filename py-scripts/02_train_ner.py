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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

with open("/work/Ccp-HERMOD/data/ner_label2id.json", "r", encoding="utf-8") as f:
    ner_label2id = json.load(f)

with open("/work/Ccp-HERMOD/data/ner_id2label.json", "r", encoding="utf-8") as f:
    ner_id2label_raw = json.load(f)
    ner_id2label = {int(k): v for k, v in ner_id2label_raw.items()}

# Model
model_checkpoint = 'CALDISS-AAU/DA-BERT_Old_News_V3'
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(ner_label2id))
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model.config.label2id = ner_label2id
model.config.id2label = ner_id2label

# Importing data
tokenized_ner = datasets.load_from_disk('/work/Ccp-HERMOD/data/tokenized_ner')
tokenized_ner = tokenized_ner.remove_columns(["tokens", "ner_tags"])

# Training args
args = TrainingArguments(
    "/work/Ccp-HERMOD/modelling/DA-OldNews-BERT-NER",
    eval_strategy = "epoch",
    learning_rate = 1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
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
        [ner_id2label[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [ner_id2label[l] for p, l in zip(pred, lab) if l != -100]
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
    train_dataset=tokenized_ner['train'],
    eval_dataset=tokenized_ner['test'],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print(model.config.num_labels)
print(set(label for batch in tokenized_ner["train"]["labels"] for label in batch))

# Train plz
trainer.train()

# Predict on test set
preds = trainer.predict(tokenized_ner["test"]).predictions.argmax(-1)
true = trainer.predict(tokenized_ner["test"]).label_ids

# Convert indices to tag names, excluding padding (-100)
decoded_preds = [[ner_id2label[p] for p, t in zip(pred_seq, true_seq) if t != -100]
                 for pred_seq, true_seq in zip(preds, true)]
decoded_true = [[ner_id2label[t] for p, t in zip(pred_seq, true_seq) if t != -100]
                for pred_seq, true_seq in zip(preds, true)]

# Print entity-level precision/recall/F1
print(classification_report(decoded_true, decoded_preds))