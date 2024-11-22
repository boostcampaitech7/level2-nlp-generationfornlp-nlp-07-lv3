import ast
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pandas as pd
import time
from ast import literal_eval
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI

import torch

torch.cuda.empty_cache()

API_KEY = "API HERE"
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

model_name = 'Qwen/Qwen2.5-14B-Instruct'
save_name = model_name.split('/')[1] + '_summurization_' + time.strftime('%Y%m%d_%H%M%S')
length = 700

parent_dir = os.path.dirname(os.getcwd())
dataset = pd.read_csv(os.path.join(parent_dir, 'data', 'chinese_data.csv'))


def sperate_dataset(dataset: pd.DataFrame, length=length):
    # return dataset that paragraph length is longer than length
    return dataset[dataset['paragraph'].apply(lambda x: len(x) > length)], dataset[
        dataset['paragraph'].apply(lambda x: len(x) <= length)]


def get_llm_output(data_row, tokenizer, model):
    paragraph = data_row['paragraph']
    problems = ast.literal_eval(data_row['problems'])

    question = problems['question']
    choices = problems['choices']
    answer = problems['answer']

    messages = [
        {
            'role': 'system',
            'content': (
                "You are a professional journalist tasked with summarizing articles. "
                "Your goal is to create a concise summary of the provided article, "
                "while ensuring no sentences relevant to the given problems are removed."
            )
        },
        {
            'role': 'user',
            'content': (
                "Here is an article and related information:\n"
                f"Article: {paragraph}\n"
                f"Problem: {question}\n"
                f"Choices: {choices}\n"
                f"Answer: {answer}\n"
                "Your task: Summarize the article based on the context of the problem. "
                "Make the summary concise and ensure no critical sentences related to the problem are omitted."
            )
        }
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    gen_tokens = model.generate(
        input_ids.to('cuda'),
        max_length=int(len(paragraph) * 2),
        do_sample=True,
        temperature=0.3,
    )
    gen_text = tokenizer.decode(gen_tokens[0])

    trim_start_index = gen_text.rfind('<|im_start|>')
    trim_end_index = gen_text.rfind('<|im_end|>')
    trim = gen_text[trim_start_index + len('<|im_start|>'):trim_end_index]
    trim = trim.replace('assistant\n', '')
    trim = trim.replace('<|im_start|>', '')
    trim = trim.replace('<|im_end|>', '')
    trim = trim.replace('\n\n', '\n')

    return trim


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


def run_llm(dataset: pd.DataFrame, save_every=10):
    longer, shorter = sperate_dataset(dataset)

    # sort longer dataset by length of paragraph column desc
    longer = longer.sort_values(by='paragraph', key=lambda x: x.str.len(), ascending=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quant = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, trust_remote_code=True)

    new_df = pd.DataFrame(columns=['id', 'paragraph', 'problems', 'question_plus'])

    for i in tqdm(range(0, len(longer))):
        before = len(longer.iloc[i]['paragraph'])
        output = get_llm_output(longer.iloc[i], tokenizer, model)
        after = len(output)

        print(f'Before : {before} / After : {after}')

        # add trim_paragraph column to dataset
        new_df = pd.concat([new_df, pd.DataFrame({'id': [longer.iloc[i]['id']], 'paragraph': [output],
                                                  'problems': [longer.iloc[i]['problems']],
                                                  'question_plus': [longer.iloc[i]['question_plus']]}), ],
                           ignore_index=True)

        if i % save_every == 0:
            new_df.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)

    # concat and sort by length desc then save
    new_df.to_csv(os.path.join(parent_dir, 'data', f'{save_name}.csv'), index=False)


if __name__ == '__main__':
    run_llm(dataset)