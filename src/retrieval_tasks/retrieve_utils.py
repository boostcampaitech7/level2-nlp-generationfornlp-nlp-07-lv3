import torch

from src.arguments import CustomArguments
from retrieval import Retrieval
from retrieval_hybrid import HybridSearch
from LLM_tasks import llm_check, llm_summary

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def len_of_tokens(tokenizer, context):
    tokens = tokenizer.tokenize(context)
    return len(tokens)

def len_of_chat_template(tokenizer, custom_args: CustomArguments):
    message = [
                {"role": "system", "content": custom_args.rag_System_prompt},
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

def retrieve(retriever: Retrieval, llm, tokenizer, messages, max_seq_length, custom_args: CustomArguments, topk: int=5):
    prompt_tokens = len_of_tokens(tokenizer, messages)
    chat_template_tokens = len_of_chat_template(tokenizer, custom_args) + 10
    max_response_tokens = max_seq_length - (prompt_tokens + chat_template_tokens)
    rag_response_threshold = prompt_tokens + chat_template_tokens
    if max_response_tokens < 0: 
        print("[max_response_tokens error] max_response_tokens를 초과함")
        return None
    if rag_response_threshold > custom_args.rag_response_threshold:
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