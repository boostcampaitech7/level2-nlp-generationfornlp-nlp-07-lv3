from transformers import AutoTokenizer
import pandas as pd
import ast  # 안전한 문자열 파싱을 위해 사용

# 모델 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("NCSOFT/Llama-VARCO-8B-Instruct")

def convert_to_dict(data_str):
    """
    문자열을 딕셔너리로 변환하는 함수
    """
    if isinstance(data_str, str):
        try:
            return ast.literal_eval(data_str)
        except (ValueError, SyntaxError) as e:
            print(f"String to dict conversion error: {e} - 문제 있는 문자열: {data_str}")
            return None
    return data_str


def format_choices(choices):
    """
    choices를 포맷팅하는 함수
    """
    if isinstance(choices, list):
        return '\n'.join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
    else:
        print(f"Invalid choices format: {choices}")
        return "Invalid choices"


def truncate_message(message, max_tokens=1024):
    """
    메시지를 토큰화하고, 최대 길이를 초과하면 자르는 함수
    """
    # 원본 메시지 토큰화
    original_tokens = tokenizer(message, add_special_tokens=False)["input_ids"]
    original_token_length = len(original_tokens)
    
    # 메시지 길이 초과 확인
    if original_token_length > max_tokens:
        # 자르기 전 디버깅 프린트
        print(f"[DEBUG] Message length exceeds max tokens ({original_token_length} > {max_tokens}). Truncating...")

        # 자른 토큰 생성
        truncated_tokens = original_tokens[:max_tokens]
        truncated_message = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # 잘린 부분 디코딩
        removed_tokens = original_tokens[max_tokens:]
        removed_message = tokenizer.decode(removed_tokens, skip_special_tokens=True)

        # 디버깅 정보 출력
        print(f"[DEBUG] Original token length: {original_token_length}")
        print(f"[DEBUG] Truncated token length: {len(truncated_tokens)}")
        print(f"[DEBUG] Removed token length: {len(removed_tokens)}")
        print(f"[DEBUG] Removed content:\n{removed_message}")
        
        return truncated_message
    else:
        # 초과하지 않는 경우
        print(f"[DEBUG] Message length within max tokens ({original_token_length} ≤ {max_tokens}). No truncation needed.")
        return message



def create_prompt(row, max_tokens=1024, debug=False):
    """
    단일 데이터에 대한 프롬프트를 생성하는 함수
    """
    try:
        problems = convert_to_dict(row.get('problems', {}))
        if not problems or 'question' not in problems or 'choices' not in problems or 'answer' not in problems:
            raise ValueError("Invalid or incomplete 'problems' field.")
        
        formatted_choices = format_choices(problems['choices'])
        system_prompt = """### Instruction:
Below is a task with context. Select the best answer among the options based on the passage."""
        
        paragraph = row.get('paragraph')
        if not paragraph:
            raise ValueError(f"Missing paragraph field in row: {row}")
        
        # 자르기 전 디버깅 정보
        paragraph_tokens = tokenizer(paragraph, add_special_tokens=False)["input_ids"]
        print(f"[DEBUG] Original paragraph token length: {len(paragraph_tokens)}")
        
        # 사용할 수 있는 최대 토큰 계산
        available_tokens = max_tokens - len(tokenizer(f"### Input:\nText:\nQ:\n{problems['question']}\n\nOptions:\n{formatted_choices}", add_special_tokens=False)["input_ids"])
        
        # 디버깅: 남은 토큰 길이 출력
        print(f"[DEBUG] Available tokens for paragraph: {available_tokens}")
        
        # 문단 자르기
        truncated_paragraph = truncate_message(paragraph, max_tokens=available_tokens)

        user_prompt = f"""### Input:
Text:
{truncated_paragraph}
Q:
{problems['question']}"""

        if pd.notna(row.get('question_plus')):
            user_prompt += f"\nNote:\n{row['question_plus']}"

        user_prompt += f"\n\nOptions:\n{formatted_choices}"
        
        assistant_prompt = f"""### Response:
{problems['answer']}"""

        full_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]

        if debug:
            print(f"[DEBUG] Created prompt:\n{full_message}")
        
        return full_message
    
    except Exception as e:
        print(f"[ERROR] Failed to create prompt: {e}")
        print(f"Row data: {row.to_dict()}")
        return None



def process_data(data, max_tokens=1024):
    """
    DataFrame 또는 CSV 경로를 받아 데이터 처리
    """
    try:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data
        
        all_messages = []
        for _, row in df.iterrows():
            processed = create_prompt(row, max_tokens=max_tokens)
            if processed:
                all_messages.append(processed)
        
        return all_messages
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None



# 예시 실행
if __name__ == "__main__":
    file_path = "/data/ephemeral/home/hsk/level2-nlp-generationfornlp-nlp-07-lv3/data/train.csv"
    processed_messages = process_data(file_path)
    if processed_messages:
        print("Processed messages:")
        for message in processed_messages[:6]:  # 첫 5개만 출력
            print(message)
