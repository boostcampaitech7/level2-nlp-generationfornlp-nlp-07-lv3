import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk

from retrieval import retrieval

from transformers import AutoTokenizer, AutoModel

def len_of_tokens(tokenizer, context):
        tokens = tokenizer.tokenize(context)
        return len(tokens)

def llm_summary(llm, tokenizer, retrieved_contexts, max_response_tokens):
    messages = [
        {"role": "system", 
        "content": f"주어진 지문과 문제를 바탕으로, rag된 문서에서 문제 풀이에 도움이 될 만한 내용들을 중심으로 {max_response_tokens} 길이로 요약을 해주세요. 요약한 내용만 출력하고 기타 다른 추가적인 설명 같은 건 생략하세요"},
        {"role": "user", "content": retrieved_contexts}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        truncation=True,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = llm.generate(
        inputs.to(device),
        max_new_tokens=max_response_tokens,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = generated_text.split("[|assistant|]")[-1].strip()
    if '\n' in result:
        result = result.split("\n")[0]
    del inputs, outputs, generated_text
        
    return result

def llm_check(llm, tokenizer, query):
    prompt = (
        f"다음은 수능 지문입니다. 이 지문에 추가적인 rag가 필요한지 구분해 주세요.\n\n"
        f"{query}\n\n"
        f"위의 지문과 문제를 바탕으로, 해당 지문를 가지고 문제와 보기를 확인해서 대략 10B짜리의 pre-train된 모델이 있을 때 자력으로 대답이 가능한지 혹은 불가능해서 rag가 필요한지를 추가적 설명 없이 가능한 답변 내에서만 답변하세요. 가능한 답변: '필요함' 또는 '필요하지않음'"
    )   
    messages = [
        {"role": "system", 
        "content": "당신은 수능감독관입니다."},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        truncation=True,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = llm.generate(
        inputs.to(device),
        max_new_tokens=10,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = generated_text.split("[|assistant|]")[-1].strip()
    if '\n' in result:
        result = result.split("\n")[0]
    del inputs, outputs, generated_text
        
    return result


def retrieve(retriever, llm, tokenizer, messages, max_seq_length, topk: int=5):
    prompt_tokens = len_of_tokens(tokenizer, ' '.join(messages['content']))
    max_response_tokens = max_seq_length - (prompt_tokens + 20)
    if max_response_tokens < 0: 
        return None

    query = [message['content'] for message in messages if message['role'] == 'user'][-1]
    result = llm_check(llm, tokenizer, query)
    if '필요함' in result:
        _ , contexts = retriever.retrieve(query, topk=topk)
        summary = llm_summary(llm, tokenizer, ' '.join(contexts), max_response_tokens)
        return summary
    elif '필요하지않음' in result:
        return None
    else:
        return None
