import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def llm_summary(llm, tokenizer, retrieved_contexts, max_response_tokens, device):
    """
    주어진 문맥과 관련 내용을 바탕으로, 문제 해결에 도움이 될 핵심 포인트를 요약합니다.
    요약은 max_response_tokens 길이로 생성되며, 추가 설명 없이 요약문만 출력됩니다.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = device
    messages = [
        {"role": "system", 
         "content": (
             f"제공된 내용과 문맥을 참고하여, 문제 해결에 도움이 될 핵심 포인트를 {max_response_tokens} 토큰 길이로 요약해 주세요. "
             "요약한 내용만 출력하고, 추가 설명은 생략합니다.\n\n"
             f"{retrieved_contexts}"
         )},
        {"role": "user", "content": ""}
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

def llm_check(llm, tokenizer, query, device):
    """
    주어진 텍스트에 대해, 문제 해결을 위해 검색 보강 생성(RAG)이 필요한지 평가합니다.
    텍스트를 바탕으로 RAG가 필요하면 '필요함', 불필요하면 '필요하지 않음'을 별도의 설명 없이 출력합니다.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = device
    prompt = (
        "다음은 텍스트입니다. 해당 텍스트를 바탕으로 문제 해결을 위해 검색 보강 생성(RAG)이 필요한지 판별하세요.\n\n"
        f"텍스트: '{query}'\n\n"
        "위 텍스트를 참고하여, RAG가 필요하면 '필요함', 필요하지 않으면 '필요하지 않음'을 추가 설명 없이 출력하세요.\n\n"
    )
    messages = [
        {"role": "system", 
         "content": f"당신은 평가자입니다.\n{prompt}"},
        {"role": "user", "content": ""}
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
