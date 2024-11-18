from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from LlamaPreprocessData import process_data
import pandas as pd
import csv
from tqdm import tqdm


base_model_name = "NCSOFT/Llama-VARCO-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, "output_generation")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                          trust_remote_code=True)

tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"

# CSV 파일 처리
# test.csv 파일 경로
csv_path = '/data/ephemeral/home/hsk/level2-nlp-generationfornlp-nlp-07-lv3/data/test.csv'

messages = process_data(csv_path)
print(len(messages))
print(messages[1])
formatted_message = []

for single_dict in messages:
    chat_message = tokenizer.apply_chat_template(single_dict, 
                                                tokenize=False, 
                                                add_generation_prompt=False)
    formatted_message.append(chat_message[:-57] + '\n')

print('---비교 디버깅---')
print(formatted_message[:2]) 

# id 엮어주기
ids = pd.read_csv(csv_path)

# messages 리스트와 id 칼럼의 값들을 엮어주기
result = list(zip(messages, ids['id']))

def generate_valid_answer(model, tokenizer, inputs, generation_config):
    valid_answers = {'1', '2', '3', '4', '5'}
    
    while True:
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                **generation_config
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        last_char = generated_text.strip()[-1]
        
        # 유효한 답변이면 반환
        if last_char in valid_answers:
            return last_char
        
        # 유효하지 않은 답변이면 다시 생성

# 결과를 저장할 리스트
final_results = []

# 생성 파라미터 설정
generation_config = {
    "max_new_tokens": 4,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

# 각 메시지에 대해 생성 수행
for formatted_msg, id_ in tqdm(result):
    # 토크나이징
    inputs = tokenizer(formatted_msg, 
                      return_tensors='pt',
                      add_special_tokens=False)
    
    # GPU 사용 가능시
    inputs = inputs.to('cuda')
    
    # 유효한 답변 생성
    answer = generate_valid_answer(model, tokenizer, inputs, generation_config)
    
    # 결과 저장
    final_results.append([id_, answer])

# CSV 파일로 저장
output_path = 'generation_results.csv'
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'answer'])
    writer.writerows(final_results)

print(f"Results saved to {output_path}")