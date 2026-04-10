---
title: "\U0001F3DB️  Historical Entity Recognition Model for the Disappeared"

---

# 🏛️ Historical Entity Recognition Model for the Disappeared (HERMOD)    

This project fine-tunes a domain-adapted Transformer model to perform **token-level classification** on historical Danish texts. The model identifies:

* **Named Entities (NER)** (e.g. locations, persons)
* **Verbs / actions** relevant to historical events

The system is designed to support research into **runaway individuals during Danish absolutism**, helping automate the discovery of:

* places of interest
* actions and movements

---

## Background

The model builds on:

* **DA-BERT_Old_News_V3** (from the *CALDISS OldNews-BERT project*)
* Training data from:

  * *CALDISS “tagging coercion” project*
  * Additional **hand-labelled datasets from research collaborators**

---

## What the model does

This is a **token classification pipeline** that:

1. Preprocesses and converts raw annotated data
2. Fine-tunes Transformer models for:

   * Named Entity Recognition (NER)
   * Verb POS tagging
3. Produces trained checkpoints and inference outputs

---

## 📁 Project Structure

```
.
├── data/                # Training + evaluation datasets (JSONL, tokenized)
├── modelling/           # Model configs / pretrained checkpoints
├── py-scripts/          # Core pipeline scripts
│   ├── 01_convert_data.py # convertion from progidy format to expected classification data allignment
│   ├── 02_train_ner.py # Training on NER tags
│   ├── 03_train_pos.py # Training on POS tags
│   └── 04_BIO-tag_o_convert_data_andtrain_DIANA.py # Checkpoint training on extra data for better TAG-allignment
├── modules/             # Custom modules / utilities
├── output/              # Training checkpoints + inference outputs
├── requirements.txt
```

---

## Installation

Create environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Convert data

Prepare datasets for training:

```bash
python py-scripts/01_convert_data.py
```

---

### 2. Train NER model

```bash
python py-scripts/02_train_ner.py
```

---

### 3. Train POS / Verb classifier

```bash
python py-scripts/03_train_pos.py
```

---

### 4. Combined / Trained from checkpoint

```bash
python py-scripts/04_BIO-tag_o_convert_data_andtrain_v2.py
```

---

## Outputs

Training outputs are stored in:

```
output/
├── checkpoint-*
├── inference_output_cp-*.txt # <- results from inference test on hold-out data
```

These include:

* model checkpoints at different steps
* inference results for evaluation

---

## Data

The `data/` directory contains:

* Annotated datasets (JSONL)
* Tokenized datasets for training
* Label mappings:

  * `ner_label2id.json`
  * `pos_label2id.json`


---

## Experiments

The project includes:

* Multiple checkpoint runs

---

## ⚠️ Notes & Limitations

* Model is **domain-specific** (historical Danish texts)
* Performance depends heavily on:

  * annotation quality
  * domain similarity

---

## Acknowledgements

* **CALDISS project**
* *OldNews-BERT*
* Research collaborators providing annotated datasets

---

## Purpose

This project contributes to digital humanities research by enabling:

* automated extraction of historical events
* identification of movements and actors
* scalable analysis of archival text corpora

---
