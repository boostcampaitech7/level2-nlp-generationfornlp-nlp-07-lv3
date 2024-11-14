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
save_name = model_name.split('/')[1] + '_summurization_' + time.strftime('%Y%m%d_%H%M%S')
#save_name = "perplexity" + '_summurization_' + time.strftime('%Y%m%d_%H%M%S')
length = 550

parent_dir = os.path.dirname(os.getcwd())
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_5.csv'))
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_longer_then_1024.csv'))
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train.csv'))
dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_longer_then_512_tokens.csv'))


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
            'content': f'당신은 요약을 하는 기자입니다. 다음 문서를 주어진 문제를 풀 수 있는 수준으로 요약하세요. {length}자로 요약하세요. 문서만 출력하세요.'

        },
        {
            'role': 'user',
            'content': f'\n\n문서 : {paragraph}\n\n문제 : {question}\n\nChoice : {choices}\n\n요약 : '
        }
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    gen_tokens = model.generate(
        input_ids.to('cuda'),
        max_new_tokens=int(length * 1.5),
        do_sample=True,
        temperature=0.3,
        )
    gen_text = tokenizer.decode(gen_tokens[0])

    trim_start_index = gen_text.find('[|assistant|]')
    trim_0 = gen_text[trim_start_index+len('[|assistant|]'):]

    trim_end_index = trim_0.find('[|endofturn|]')
    trim_1 = trim_0[:trim_end_index]

    output = trim_1.split('\n\n')
    trim_2 = ''.join(output[:-1])

    if '문제:' in trim_2:
        trim_2 = trim_2.split('문제:')[0]

    return trim_2


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
            'content': f'당신은 요약을 하는 기자입니다. 다음 문서를 한글 {length} 글자로 요약하세요. 요약한 문서로도 문제를 풀 수 있어야 합니다. 설명 출력하지 마세요.'
                       f'문서 : {paragraph}'
                       f'문제 : {question}'
                       f'Choice : {choices}'
        },
        {
            'role': 'user',
            'content': '요약 : '
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


def run_llm(dataset : pd.DataFrame, save_every=25):
    longer, shorter = sperate_dataset(dataset)
    new_df = pd.DataFrame(columns=['id', 'paragraph', 'problems', 'question_plus'])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(
        'cuda')

    for i in tqdm(range(0, len(longer))):
        before = len(longer.iloc[i]['paragraph'])
        #output = run_perplexity(longer.iloc[i])
        output = get_llm_output(longer.iloc[i], tokenizer, model)
        after = len(output)

        if before < after:
            # drop this row
            longer = longer.drop(i)
            continue

        new_df = pd.concat([new_df, pd.DataFrame([[longer.iloc[i]['id'], output, longer.iloc[i]['problems'], longer.iloc[i]['question_plus']]], columns=new_df.columns)])
        print(f'Before : {before} / After : {after}')

        if i % save_every == 0:
            new_df.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)

    # concat and sort by length desc then save
    new_df.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)


if __name__ == '__main__':
    run_llm(dataset)
    #run_perplexity(dataset.iloc[0])