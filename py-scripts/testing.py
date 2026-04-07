import os
from transformers import pipeline, AutoTokenizer
import json
from huggingface_hub import(
    login,
    upload_folder,
    create_repo,
    HfApi
)

# inference check
# Path to the JSONL file
file_path = "/work/Ccp-HERMOD/data/hermod_sample_test.jsonl"
texts = []  # List to store all text values

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)
            texts.append(data['text'])
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {e}")
        except KeyError:
            print(f"Key 'text' not found in line: {line}")


tokenizer = AutoTokenizer.from_pretrained(
    'CALDISS-AAU/DA-BERT_Old_News_V3',
    truncation=True,
    padding="max_length",
    max_length=512
)
tokenizer.model_max_length = 512
model_checkpoint = '/work/Ccp-HERMOD/output/checkpoint-112'
ner_pipe = pipeline("ner", model=model_checkpoint, tokenizer=tokenizer, aggregation_strategy='simple')


results = ner_pipe(
    texts,
)

with open('/work/Ccp-HERMOD/output/inference_output_cp-112.txt', 'a') as f:
    for text, entities in zip(texts, results):
        print(f"Input: {text}", file=f)
        print(entities, file=f)
        print("=" * 50, file=f)


# Push to hub
# login
login()

# # create repo
create_repo(repo_id='DA-SBERT_Old_News_V1', repo_type='model', private=False)

# Get token and upload folder to repo
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
     folder_path='/work/Ccp-HERMOD/TSDAE/py-scr/output/tsdae-example/checkpoint-49475',
     repo_id="SirMappel/DA-SBERT_Old_News_V1",
     repo_type="model",
)