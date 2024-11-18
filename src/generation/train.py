from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig
from trl import SFTTrainer
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from LlamaPreprocessData import process_data
from datetime import datetime
import yaml


# Load secrets
with open('../../secrets.yaml', 'r') as f:
    secrets = yaml.safe_load(f)

# 현재 시간 가져오기
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 실험 이름 생성 (여기서 'Llama3.1_varco_8B' 대신 적절한 이름을 지정)
experiment_name = f"[train]_Llama3.1_varco_8B_{current_time}"


model_name = "NCSOFT/Llama-VARCO-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             torch_dtype = torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True)

tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"

# 토크나이저 설정
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정

# CSV 파일 처리
messages = process_data("/data/ephemeral/home/hsk/level2-nlp-generationfornlp-nlp-07-lv3/data/train.csv")
print(len(messages))
print(messages[1])
formatted_message = []

for single_dict in messages:
    chat_message = tokenizer.apply_chat_template(single_dict, 
                                                tokenize=False, 
                                                add_generation_prompt=False)
    formatted_message.append(chat_message)

print('---비교 디버깅---')
print(formatted_message[:2]) 


# 데이터를 딕셔너리 형태로 변환
data_dict = {
    'text': formatted_message
}

# train/eval 분할
train_texts, eval_texts = train_test_split(
    formatted_message, 
    test_size=0.1
)

# Dataset 객체 생성
train_dataset = Dataset.from_dict({'text': train_texts})
eval_dataset = Dataset.from_dict({'text': eval_texts})

# 확인
print("Train dataset size:", len(train_dataset))
print("Eval dataset size:", len(eval_dataset))

# 데이터 설정 끝 이제 Peft 설정

peft_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj'],
    bias="none",
    task_type="CAUSAL_LM",
)


# 훈련 파라미터 설정
training_params = TrainingArguments(
                                    output_dir="./output_generation",
                                    do_train=True,
                                    do_eval=True,
                                    lr_scheduler_type="cosine",
                                    per_device_train_batch_size=1,
                                    per_device_eval_batch_size=1,
                                    num_train_epochs=3,
                                    learning_rate=2e-5,
                                    weight_decay=0.01,
                                    logging_steps=200,
                                    save_strategy="epoch",
                                    eval_strategy="epoch",
                                    save_total_limit=1,
                                    save_only_model=True,
                                    report_to="wandb")

# 트레이너 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset = eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
    # data_collator=data_collator 못씀. 동적패딩할 배치사이즈가 아님.
)

wandb.login(key = secrets['wandb']['api_key'])

wandb.init(
    project="CSAT-Solver",  # 프로젝트 이름
    entity="NotyNoty",      # 사용자 또는 팀 이름
    name=experiment_name,    # 실험 이름
    save_code=True,          # 코드 저장 여부
)

# finetuning
trainer.train()
wandb.finish()

