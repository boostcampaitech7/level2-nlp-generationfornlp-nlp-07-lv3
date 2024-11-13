import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
from ast import literal_eval
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
torch.cuda.empty_cache()

model_name = 'CohereForAI/aya-expanse-8b'
model_name = 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct'
length = 512

parent_dir = os.path.dirname(os.getcwd())
dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_5.csv'))
dataset = dataset.sample(frac=1).reset_index(drop=True)

sample = dataset.iloc[0]

paragraph = sample['paragraph']
problems = literal_eval(sample['problems'])
question = problems['question']
choices = problems['choices']

messages = [
    {
        'role': 'system',
        'content': f'당신은 요약을 하는 기자입니다. 다음 문서를 {length}자 이하로 요약하세요. 요약본으로 정답을 맞출 수 있어야 합니다.\n'
                   f'문제 : {question}\n'
                   f'답 : {[f"{i+1}, {choice}" for i, choice in enumerate(choices)]}\n',
    },
    {
        'role': 'user',
        'content': paragraph,
    }
]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to('cuda')

input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

gen_tokens = model.generate(
    input_ids.to('cuda'),
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])

trim_start_index = gen_text.find('[|assistant|]')
trim_0 = gen_text[trim_start_index+len('[|assistant|]'):]

trim_end_index = trim_0.find('[|endofturn|]')
trim_1 = trim_0[:trim_end_index]

output = trim_1.replace('\n', ' ')
print(output)
print(len(output))
