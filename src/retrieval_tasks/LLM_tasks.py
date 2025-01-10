import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

def llm_summary(llm, tokenizer, retrieved_contexts, max_response_tokens):
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    messages = [
        {"role": "system", 
        "content": f"주어진 지문과 문제를 바탕으로, rag된 문서에서 문제 풀이에 도움이 될 만한 내용들을 중심으로 {max_response_tokens} 길이로 요약을 해주세요. 요약한 내용만 출력하고 기타 다른 추가적인 설명 같은 건 생략하세요\n\n{retrieved_contexts}"},
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

def llm_check(llm, tokenizer, query):
    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    prompt = (
        f"다음은 수능 지문입니다. RAG가 필요한지 자력으로 대답 가능한지 판별하시오.\n\n" 
        f"'{query}'\n\n"
        f"위의 지문과 문제를 바탕으로, RAG가 필요하면 '필요함'을 없으면 '필요하지않음'을 설명없이 출력하세요.\n\n"
    )   
    messages = [
        {"role": "system", 
        "content": f"당신은 수능 감독관입니다.\n{promt}"},
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