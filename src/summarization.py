import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
import time
from ast import literal_eval
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
torch.cuda.empty_cache()

model_name = 'CohereForAI/aya-expanse-8b'
model_name = 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct'
save_name = model_name.split('/')[1] + '_summurization_' + time.strftime('%Y%m%d_%H%M%S')
length = 512

parent_dir = os.path.dirname(os.getcwd())
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_5.csv'))
#dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train_sample_longer_then_1024.csv'))
dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'train.csv'))

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to('cuda')


def sperate_dataset(dataset : pd.DataFrame, length=length):
    # return dataset that paragraph length is longer than length
    return dataset[dataset['paragraph'].apply(lambda x: len(x) > length)], dataset[dataset['paragraph'].apply(lambda x: len(x) <= length)]


def get_llm_output(data_row, length=length):
    paragraph = data_row['paragraph']

    if len(paragraph) < length:
        return paragraph

    problems = literal_eval(data_row['problems'])
    question = problems['question']
    choices = problems['choices']

    messages = [
        {
            'role': 'system',
            'content': f'당신은 요약을 하는 기자입니다. 다음 문서를 {length}자 이하의 문서로 요약하세요'
                       f'문서 : {paragraph}'
        },
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


def run_llm(dataset : pd.DataFrame, save_every=100):
    longer, shorter = sperate_dataset(dataset)

    for i in tqdm(range(0, len(longer), 100)):
        longer.iloc[i]['paragraph'] = longer.iloc[i].apply(get_llm_output, axis=1)

        if i % save_every == 0:
            dataset = pd.concat([longer, shorter])
            dataset.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)

    # concat and sort by length desc then save
    dataset = pd.concat([longer, shorter])
    dataset.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)


if __name__ == '__main__':
    run_llm(dataset)