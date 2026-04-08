import re
import json
import copy
from datasets import Dataset
import os
import numpy as np
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
from transformers import EarlyStoppingCallback

my_input = "/work/Ccp-HERMOD/data/HERMOD_extra_training_data_original_experiment.jsonl"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        jsonlines = [json.loads(line) for line in f if line.strip()]
        return jsonlines

def tokenize_with_offsets(text):
    tokens, offsets = [], []
    for m in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def convert_to_bio_tags(my_example):
    text = my_example["text"]
    tokens, offsets = tokenize_with_offsets(text)

    tags = ["O"] * len(tokens)

    for start, end, label in my_example.get("label", []):

        entity = label.replace("B-", "").replace("I-", "")

        overlapping = [
            i for i, (ts,te) in enumerate(offsets)
            if not ( te <= start or ts >= end) #test for overlap
        ]

        if not overlapping:
            continue


        tags[overlapping[0]] = f"B-{entity}"

        for i in overlapping[1:]:
            tags[i] = f"I-{entity}"

    return {
        "id" : my_example.get("id"),
        "tokens": tokens,
        "ner_tags" : tags

    }

entity_data = load_jsonl(my_input)
converted_entities = [convert_to_bio_tags(ex) for ex in entity_data]



#save this to an empty file please
with open("/work/Ccp-HERMOD/data/HERMOD_extra_training_data_original_FINISHED.jsonl", "w", encoding = "utf-8") as f:
    for row in converted_entities:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

bio_data = load_jsonl("/work/Ccp-HERMOD/data/HERMOD_extra_training_data_original_FINISHED.jsonl")
dataset = Dataset.from_list(bio_data)


#Train ner and #train pos
label_list = sorted({t for ex in converted_entities for t in ex["ner_tags"]})
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42) # Seed should be fixed by the way so that we dont get random results based on different splitting!

dataset = dataset.map(lambda ex: {"ner_tags" : [label2id[t]  for t in ex["ner_tags"]]})
print(label_list)


model_checkpoint = "CALDISS-AAU/DA-OldNews-BERT-NER-VerbPOS-v1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align(examples):
    tok = tokenizer(
        examples["tokens"],
        is_split_into_words = True,
        truncation = True
    )

    labels = []

    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tok.word_ids(batch_index=i)
        previous = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous:
                label_ids.append(word_labels[word_id])

            else:
                #label subword pieces same as the word
                label_ids.append(word_labels[word_id])
            previous = word_id 
        labels.append(label_ids)


    tok["labels"] = labels
    return tok

tokenized_ds = dataset.map(tokenize_and_align, batched = True)
tokenized_ds = tokenized_ds.remove_columns(["id", "tokens", "ner_tags"])

#Just copy the creation of the model from train_ner.py

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, 
num_labels=len(label_list),
ignore_mismatched_sizes=True)

model.config.label2id = label2id
model.config.id2label = id2label
# Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)
# Metrics
metric = evaluate.load('seqeval')
#This is copied from train_ner 100% I just changed the vars a littlebuit
def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis = 2)
    labels = p.label_ids

    true_predictions = [
        [id2label[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Training args
args = TrainingArguments(
    output_dir = "/work/Ccp-HERMOD/output",
    eval_strategy = "epoch",
    save_strategy = "epoch",  
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss", 
    greater_is_better=False,

    learning_rate = 2e-5, #Changed the learningrate a little
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model = model,
    args =args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
)

print(model.config.num_labels)
print(set(label for batch in tokenized_ds["train"]["labels"] for label in batch))

# Train plz
trainer.train()

