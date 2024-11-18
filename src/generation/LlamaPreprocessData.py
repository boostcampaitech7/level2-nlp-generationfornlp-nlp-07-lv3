import pandas as pd
import ast  # 안전한 문자열 파싱을 위해 사용


def convert_to_dict(data_str):
    """
    문자열을 딕셔너리로 변환하는 함수
    - ast.literal_eval 사용: 안전한 문자열 파싱
    - 실패 시 None 반환
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
    - 리스트의 각 항목에 번호 추가
    """
    if isinstance(choices, list):
        return '\n'.join([f"{i + 1}. {choice}" for i, choice in enumerate(choices)])
    else:
        print(f"Invalid choices format: {choices}")
        return "Invalid choices"


def create_prompt(row):
    """
    단일 데이터에 대한 프롬프트를 생성하는 함수
    - 각 필드를 robust하게 처리
    """
    try:
        # problems 열 파싱
        problems = convert_to_dict(row.get('problems', {}))
        if not problems or 'question' not in problems or 'choices' not in problems or 'answer' not in problems:
            raise ValueError("Invalid or incomplete 'problems' field.")
        
        # choices 포맷팅
        formatted_choices = format_choices(problems.get('choices', []))
        
        # 기본 system 메시지
        system_prompt = """### Instruction:
Below is a task with context. Select the best answer among the options based on the passage."""
        
        # 사용자 프롬프트 구성
        user_prompt = f"""### Input:
Text:
{row.get('paragraph', 'No paragraph provided')}
Q:
{problems['question']}"""

        # question_plus 추가
        if pd.notna(row.get('question_plus')):
            user_prompt += f"\nNote:\n{row['question_plus']}"

        # options 추가
        user_prompt += f"\n\nOptions:\n{formatted_choices}"
        
        # 정답 처리
        answer = problems.get('answer', 'No answer provided')
        if isinstance(answer, int):
            answer_text = f"{answer}"  # 숫자를 문자열로 변환
        else:
            answer_text = str(answer)  # 기타 형식 문자열 변환
        
        # assistant 응답 구성
        assistant_prompt = f"""### Response:
{answer_text}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prompt}
        ]
    except Exception as e:
        print(f"Error processing row: {e}")
        print(f"Problematic row: {row.to_dict()}")
        return None


def process_data(data):
    """
    DataFrame 또는 CSV 경로를 받아 데이터 처리
    - 데이터 전체를 처리하며 문제 있는 행 무시
    """
    try:
        # CSV 경로 또는 DataFrame 처리
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data
        
        all_messages = []
        for _, row in df.iterrows():
            processed = create_prompt(row)
            if processed:
                all_messages.append(processed)
        
        return all_messages
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


# 예시 데이터 로드 및 실행
if __name__ == "__main__":
    # 예시 데이터 파일 경로
    file_path = "example.csv"
    
    # 데이터 처리 실행
    processed_messages = process_data(file_path)
    
    # 결과 출력
    if processed_messages:
        print("Processed messages:")
        for message in processed_messages:
            print(message)
