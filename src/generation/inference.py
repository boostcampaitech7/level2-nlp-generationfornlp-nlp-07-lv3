from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from LlamaPreprocessData import process_data
import pandas as pd
import csv
from tqdm import tqdm
import torch
import gc
import random

# 기본 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model_name = "NCSOFT/Llama-VARCO-8B-Instruct"

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, "/data/ephemeral/home/output_generation/checkpoint-5481")
model = model.to(device)  # 모델을 GPU로 이동
model.eval()  # 평가 모드 설정 (메모리 절약)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

tokenizer.chat_template = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}"
)

# CSV 파일 처리
csv_path = '/data/ephemeral/home/hsk/level2-nlp-generationfornlp-nlp-07-lv3/data/test.csv'

messages = process_data(csv_path)
ids = pd.read_csv(csv_path)

print(len(messages))
print(messages[1])

# 메시지 템플릿 처리
formatted_message = []
for single_dict in messages:
    chat_message = tokenizer.apply_chat_template(single_dict, tokenize=False, add_generation_prompt=False)
    formatted_message.append(chat_message[:-10] + '\n')

print('---비교 디버깅---')
print(formatted_message[:2])

# id와 메시지 엮기
result = list(zip(formatted_message, ids['id']))
print("---result[:2]---")
print(result[:2])

# 생성 파라미터 설정
generation_config = {
    "max_new_tokens": 1,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "output_attentions": False,
    "output_hidden_states": False
}

# 메모리 정리 함수
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

# 유효한 답변 생성 함수
def generate_valid_answer(model, tokenizer, inputs, generation_config):
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                **generation_config
            )
        
        # Multiple sequence handling
        if generation_config.get('num_return_sequences', 1) > 1:
            for output in outputs:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                last_char = generated_text.strip()[-1]
                if last_char in {'1', '2', '3', '4', '5'}:
                    return last_char
        else:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            last_char = generated_text.strip()[-1]
            if last_char in {'1', '2', '3', '4', '5'}:
                return last_char
        
        attempts += 1
    
    # Fallback to random choice if no valid answer
    return random.choice(['1', '2', '3', '4', '5'])

# 결과 저장 리스트
final_results = []

# 메시지 처리 루프
for formatted_msg, id_ in tqdm(result):
    clear_memory()
    inputs = tokenizer(
        formatted_msg,
        return_tensors='pt',
        max_length=1792,
        add_special_tokens=False,
        truncation=True
    )

    # 입력 데이터를 모델 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 유효한 답변 생성
    answer = generate_valid_answer(model, tokenizer, inputs, generation_config)
    final_results.append([id_, answer])
    print(final_results[-1])

    # 메모리 해제
    del inputs
    torch.cuda.empty_cache()

# CSV 파일로 저장
output_path = 'generation_results.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'answer'])
    writer.writerows(final_results)

print(f"Results saved to {output_path}")
