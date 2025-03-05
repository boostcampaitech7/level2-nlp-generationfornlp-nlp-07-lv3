import torch
import logging
from typing import List, Optional, Tuple, NoReturn
from arguments import CustomArguments
from .retrieval import Retrieval

from .LLM_tasks import llm_check, llm_summary

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)

def len_of_tokens(tokenizer, context):
    tokens = tokenizer.tokenize(context)
    return len(tokens)

def len_of_chat_template(tokenizer, custom_args: CustomArguments):
    message = [
                {"role": "system", "content": custom_args.rag_system_prompt},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""}
            ]
    template = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                )
    return len_of_tokens(tokenizer, template)

def truncation(tokenizer, contexts: str, max_response_tokens):
    token_ids = tokenizer.encode(
        contexts,
        truncation=True,
        max_length=max_response_tokens,
        add_special_tokens=False 
    )
    truncated_context = tokenizer.decode(token_ids, skip_special_tokens=True)
    return truncated_context

def retrieve(retriever: Optional[Retrieval], llm, tokenizer, messages, max_seq_length, custom_args: CustomArguments, topk: int=5, device = None):
    prompt_tokens = len_of_tokens(tokenizer, messages)
    chat_template_tokens = len_of_chat_template(tokenizer, custom_args) + 10
    max_response_tokens = max_seq_length - (prompt_tokens + chat_template_tokens)
    rag_response_threshold = prompt_tokens + chat_template_tokens
    if max_response_tokens < 0: 
        logging.info("[max_response_tokens error] max_response_tokens를 초과함")
        return None
    if rag_response_threshold > custom_args.rag_response_threshold:
        logging.info("[rag_response_threshold error] rag_response_threshold를 초과함")
        return None

    query = messages
    # result = llm_check(llm, tokenizer, query, device=device)
    result = "필요함"
    logging.info(query)
    logging.info(f"[RAG가 필요한가?] {result}")
    if '필요함' in result:
        if isinstance(query, str):
            _ , contexts = retriever.retrieve(query, topk=topk)
        else:
            contexts = retriever.retrieve(query, topk=topk)
        summary = llm_summary(llm, tokenizer, ' '.join(contexts), max_response_tokens, device=device)
        # summary = truncation(tokenizer, ' '.join(contexts)[:], max_response_tokens)
        logging.info(f"[RAG & Summary] {summary}")
        return summary
    elif '필요하지않음' in result:
        return None
    else:
        return None

if __name__=="__main__":
    pass