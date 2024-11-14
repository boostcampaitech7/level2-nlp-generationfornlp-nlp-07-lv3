import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
import time
from ast import literal_eval
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

import torch
torch.cuda.empty_cache()

API_KEY = "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"
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

model_name = 'CohereForAI/aya-expanse-8b'
model_name = 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct'
#save_name = model_name.split('/')[1] + '_summurization_' + time.strftime('%Y%m%d_%H%M%S')
save_name = "perplexity" + '_summurization_' + time.strftime('%Y%m%d_%H%M%S')
length = 550

parent_dir = os.path.dirname(os.getcwd())
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_5.csv'))
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_longer_then_1024.csv'))
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train.csv'))
dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_longer_then_512_all.csv'))


def sperate_dataset(dataset : pd.DataFrame, length=length):
    # return dataset that paragraph length is longer than length
    return dataset[dataset['paragraph'].apply(lambda x: len(x) > length)], dataset[dataset['paragraph'].apply(lambda x: len(x) <= length)]


def get_llm_output(data_row, tokenizer, model, length=length):
    paragraph = data_row['paragraph']

    if len(paragraph) < length:
        return paragraph

    problems = literal_eval(data_row['problems'])
    question = problems['question']
    choices = problems['choices']

    messages = [
        {
            'role': 'system',
            'content': f'당신은 요약을 하는 기자입니다. 다음 문서를 {length}자 이하의 문서로 요약하세요. 요약한 문서로도 문제를 풀 수 있어야 합니다.'
                       f'문서 : {paragraph}'
                       f'문제 : {question}'
                       f'정답 후보 : {choices}'
        },
        {
            'role': 'user',
            'content': '요약 내용 : '
        }
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    gen_tokens = model.generate(
        input_ids.to('cuda'),
        max_new_tokens=int(length * 1.5),
        do_sample=True,
        temperature=0.7,
        )
    gen_text = tokenizer.decode(gen_tokens[0])

    trim_start_index = gen_text.find('[|assistant|]')
    trim_0 = gen_text[trim_start_index+len('[|assistant|]'):]

    trim_end_index = trim_0.find('[|endofturn|]')
    trim_1 = trim_0[:trim_end_index]

    output = trim_1.replace('\n', ' ')
    return output


def run_llm(dataset : pd.DataFrame, save_every=50):
    longer, shorter = sperate_dataset(dataset)

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(
    #     'cuda')

    for i in tqdm(range(0, len(longer))):
        before = len(longer.iloc[i]['paragraph'])
        #longer.loc[i, 'paragraph'] = get_llm_output(longer.iloc[i], tokenizer, model)
        output = run_perplexity(longer.iloc[i])
        longer.loc[i, 'paragraph'] = output

        after = len(longer.iloc[i]['paragraph'])

        print(f'Before : {before} / After : {after}')

        if i % save_every == 0:
            dataset = pd.concat([longer, shorter])
            #dataset = dataset.apply(lambda x: len(x['paragraph']) <= length, axis=1)
            dataset.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)

    # concat and sort by length desc then save
    dataset = pd.concat([longer, shorter])
    #dataset = dataset.apply(lambda x : len(x['paragraph']) <= length, axis=1)
    dataset.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)

def run_perplexity(data_row, length=length):
    paragraph = data_row['paragraph']

    if len(paragraph) < length:
        return paragraph

    problems = literal_eval(data_row['problems'])
    question = problems['question']
    choices = problems['choices']

    # 메시지 구성

    messages = [
        {
            'role': 'system',
            'content': f'당신은 요약을 하는 기자입니다. 다음 문서를 {length}글자 이하로 요약하세요. 요약한 문서로도 문제를 풀 수 있어야 합니다.'
                       f'문서 : {paragraph}'
                       f'문제 : {question}'
                       f'정답 후보 : {choices}'
        },
        {
            'role': 'user',
            'content': f'요약 : '
        }
    ]
    # API 호출 및 응답 받기
    response = client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )

    # 응답 출력
    output = response.choices[0].message.content
    output = output.replace('\n', ' ')

    return output

if __name__ == '__main__':
    run_llm(dataset)
    #run_perplexity(dataset.iloc[0])