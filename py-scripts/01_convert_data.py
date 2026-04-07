import json
from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer


# Loading data
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Training data - ents (addresses + cities)
entity_data = load_jsonl('/work/Ccp-HERMOD/data/2021-22_tagging-coercion/2021-22_tagging-coercion/data/tc_anno-ents_comb.jsonl')
# Training data - VERBS
verb_data = load_jsonl('/work/Ccp-HERMOD/data/2021-22_tagging-coercion/2021-22_tagging-coercion/data/tc_anno-verbs_reviewed.jsonl')


# BIO tags convertion
def convert_to_bio(example):
    '''
    converting Prodigy tags to BIO tags to align with HF eco.
    arranges labels with token key and their spans
    creates new key mer_tags based on the new labels
    '''
    labels = ["O"] * len(example["tokens"])
    for span in example.get("spans", []):
        start = span["token_start"]
        end = span["token_end"]
        label = span["label"]
        labels[start] = f"B-{label}"
        for i in range(start + 1, end + 1):
            labels[i] = f"I-{label}"
    return {
        "tokens": [tok["text"] for tok in example["tokens"]],
        "ner_tags": labels
    }



# Apply
converted_entities = [convert_to_bio(ex) for ex in entity_data]
converted_verbs = [convert_to_bio(ex) for ex in verb_data]




# Counting the amount of each tag
def count_labels(dataset):
    tag_counter = Counter()
    
    for sample in dataset:
        tag_counter.update(sample["ner_tags"])
    
    return tag_counter

tag_counts_ents = count_labels(converted_entities)
tag_counts_verbs = count_labels(converted_verbs)


# Print sorted tag counts
for tag, count in tag_counts_ents.most_common():
    print(f"{tag}: {count}")

for tag, count in tag_counts_verbs.most_common():
    print(f"{tag}: {count}")


# Changing key name for verbs
for ex in converted_verbs:
    ex['pos_tags'] = ex.pop('ner_tags')

ds_ner = Dataset.from_list(converted_entities)
ds_pos = Dataset.from_list(converted_verbs)

# Create label mapping for NER
ner_label_list = sorted(set(tag for ex in ds_ner for tag in ex["ner_tags"]))
ner_label2id = {label: i for i, label in enumerate(ner_label_list)}
ner_id2label = {i: label for label, i in ner_label2id.items()}

# Save NER
with open("/work/Ccp-HERMOD/data/ner_label2id.json", "w") as f:
    json.dump(ner_label2id, f)
with open("/work/Ccp-HERMOD/data/ner_id2label.json", "w") as f:
    json.dump(ner_id2label, f)


# Apply NER label IDs via map
ds_ner = ds_ner.map(lambda ex: {
    "ner_tags": [ner_label2id[tag] for tag in ex["ner_tags"]]
})

# Create label mapping for POS
pos_label_list = sorted(set(tag for ex in ds_pos for tag in ex["pos_tags"]))
pos_label2id = {label: i for i, label in enumerate(pos_label_list)}
pos_id2label = {i: label for label, i in pos_label2id.items()}

# Save POS
with open("/work/Ccp-HERMOD/data/pos_label2id.json", "w") as f:
    json.dump(pos_label2id, f)
with open("/work/Ccp-HERMOD/data/pos_id2label.json", "w") as f:
    json.dump(pos_id2label, f)


# Apply POS label IDs
ds_pos = ds_pos.map(lambda ex: {
    "pos_tags": [pos_label2id[tag] for tag in ex["pos_tags"]]
})

# Split datasets
ds_ner = ds_ner.train_test_split(test_size=0.2, shuffle=True)
ds_pos = ds_pos.train_test_split(test_size=0.2, shuffle=True)

# Tokenizing and aligning with labels
def tokenize_and_align_labels(examples,label_name='ner_tags', label_all_tokens = True):
    tokenized_inputs = tokenizer(examples['tokens'],
        truncation=True,max_length=512, padding='max_length', is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[label_name]):
        word_ids = \
            tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                    label_ids.append(label[word_idx] if \
                        label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

model_checkpoint = 'CALDISS-AAU/DA-BERT_Old_News_V3'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_ner = ds_ner.map(
    lambda examples: tokenize_and_align_labels(examples, label_all_tokens=True, label_name='ner_tags'),
    batched=True
)

tokenized_pos = ds_pos.map(
    lambda examples: tokenize_and_align_labels(examples, label_all_tokens=True, label_name='pos_tags'),
    batched=True
)
# Saving
tokenized_pos.save_to_disk('/work/Ccp-HERMOD/data/tokenized_pos')
tokenized_ner.save_to_disk('/work/Ccp-HERMOD/data/tokenized_ner')


