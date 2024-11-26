import argparse
import numpy as np
import pandas as pd
import torch
import logging
from datasets import Dataset, concatenate_datasets, load_from_disk
from typing import List, Optional, Tuple, Union, NoReturn

from arguments import CustomArguments
from retrieval_tasks.retrieval import retrieval
# from retrieval import retrieval
# from retrieval_hybrid import HybridSearch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig

def len_of_tokens(tokenizer, context):
    tokens = tokenizer.tokenize(context)
    return len(tokens)

def len_of_chat_template(tokenizer, custom_args: CustomArguments):
    message = [
                {"role": "system", "content": custom_args.RAG_System_prompt},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""}
            ]
    template = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                )
    return len_of_tokens(tokenizer, template)

def llm_summary(llm, tokenizer, retrieved_contexts, max_response_tokens):
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    result = generated_text.split("assistant")[-1].strip()
    if '\n' in result:
        result = result.split("\n")[0]
    del inputs, outputs, generated_text
        
    return result

def llm_check(llm, tokenizer, query):
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    prompt = (
        f"다음은 수능 지문입니다. RAG가 필요한지 자력으로 대답 가능한지 판별하시오.\n\n" 
        f"'{query}'\n\n"
        f"위의 지문과 문제를 바탕으로, RAG가 필요하면 '필요함'을 없으면 '필요하지않음'을 설명없이 출력하세요.\n\n"
    )   
    messages = [
        {"role": "system", 
        "content": "당신은 수능 감독관입니다."},
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

    result = generated_text.split("assistant")[-1].strip()
    if '\n' in result:
        result = result.split("\n")[0]
    del inputs, outputs, generated_text
        
    return result

def truncation(tokenizer, contexts: str, max_response_tokens):
    token_ids = tokenizer.encode(
        contexts,
        truncation=True,
        max_length=max_response_tokens,
        add_special_tokens=False 
    )
    truncated_context = tokenizer.decode(token_ids, skip_special_tokens=True)
    return truncated_context

def retrieve(retriever: retrieval, llm, tokenizer, messages, max_seq_length, custom_args: CustomArguments, topk: int=5):
    prompt_tokens = len_of_tokens(tokenizer, messages)
    chat_template_tokens = len_of_chat_template(tokenizer, custom_args) + 10
    max_response_tokens = max_seq_length - (prompt_tokens + chat_template_tokens)
    rag_response_threshold = prompt_tokens + chat_template_tokens
    if max_response_tokens < 0: 
        print("[max_response_tokens error] max_response_tokens를 초과함")
        return None
    if rag_response_threshold > 350:
        print("[rag_response_threshold error] rag_response_threshold를 초과함")
        return None

    query = messages
    # result = llm_check(llm, tokenizer, query)
    result = "필요함"
    print(query)
    print(f"[RAG가 필요한가?] {result}")
    if '필요함' in result:
        _ , contexts = retriever.retrieve(query, topk=topk)
        # summary = llm_summary(llm, tokenizer, ' '.join(contexts), max_response_tokens)
        summary = truncation(tokenizer, ' '.join(contexts)[:], max_response_tokens)
        print(f"[RAG & Summary] {summary}")
        return summary
    elif '필요하지않음' in result:
        return None
    else:
        return None

if __name__=="__main__":

    model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    quant_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=None,
            trust_remote_code=True,
            quantization_config=quant_config,
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    retriever = HybridSearch(
            tokenize_fn=tokenizer.tokenize,
            dense_model_name=['intfloat/multilingual-e5-large-instruct'],  #"upskyy/bge-m3-korean",
            data_path= "../data/",
            context_path = "wiki_documents_original.csv",
        )
    retriever.get_dense_embedding()
    retriever.get_sparse_embedding()

    with torch.inference_mode():
        query = ""
        messages = [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": query},
                    ]

        summary = retrieve(retriever, model, tokenizer, messages, 1024)
        print(summary)