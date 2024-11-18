# BT

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from googletrans import Translator
import time
#import deepl


class BackTranslator():
    """
    텍스트 데이터 증강을 위한 역번역(Back Translation) 클래스
    
    한국어 텍스트를 다른 언어로 번역한 후 다시 한국어로 번역하여 
    원본과 유사하지만 다른 표현의 텍스트를 생성.
    
    Attributes:
        device (str): 사용할 디바이스 ('cuda' 또는 'cpu')
        type (str): 사용할 번역기 종류 ('google' 또는 'deepl')
        loop (int): 역번역 반복 횟수
        lang (str): 중간 번역 언어 코드
        batch_size (int): 배치 처리 크기
        deepl_api_key (str): DeepL API 키 (선택적)
    """
    def __init__(self, type="google", loop=1, lang="en", batch_size=16, deepl_api_key=None):
        """
        BackTranslator 초기화
        Args:
            type (str): 번역기 종류 ("google" 또는 "deepl")
            loop (int): 역번역 반복 횟수
            lang (str): 중간 번역 언어 코드
            batch_size (int): 배치 크기
            deepl_api_key (str): DeepL API 키
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type = type
        self.loop = loop
        self.lang = lang
        self.batch_size = batch_size
        self.deepl_api_key = deepl_api_key
        
        
        # 번역기 초기화
        if self.type == "google":
            self.translator = Translator()
        elif self.type == "deepl":
            if not deepl_api_key:
                raise ValueError("DeepL API를 사용하려면 API 키가 필요합니다.")
            self.translator = deepl.Translator(deepl_api_key)
        
        print(f"Using device: {self.device}")
        print(f"Translator: {self.type}")
        print(f"Target language: {self.lang}")
        print(f"Loop count: {self.loop}")

    def _chunk_batch(self, iterable):
        """
        데이터를 배치 크기로 분할하는 메서드
        
        Args:
            iterable: 분할할 데이터 이터러블
            
        Returns:
            list: 배치 크기로 분할된 데이터 리스트
        """
        length = len(iterable)
        result = []
        for ndx in range(0, length, self.batch_size):
            result.append(iterable[ndx : min(ndx + self.batch_size, length)])
        return result

    def translate_text(self, text, src_lang, dest_lang):
        """
        단일 텍스트를 번역하는 메서드
        
        Args:
            text (str): 번역할 텍스트
            src_lang (str): 원본 언어 코드
            dest_lang (str): 목표 언어 코드
            
        Returns:
            str: 번역된 텍스트
        """
        if self.type == "google":
            translated = self.translator.translate(text, src=src_lang, dest=dest_lang).text
            time.sleep(0.5)
            return translated
        elif self.type == "deepl":
            try:
                # DeepL의 언어 코드 매핑
                source_lang_map = {
                    'ko': 'ko',      # source는 소문자 사용
                    'en': 'en',      
                    'ja': 'ja'       
                }
                
                target_lang_map = {
                    'ko': 'KO',      # target은 대문자 사용
                    'en': 'EN-US',   # 영어는 EN-US 형식 사용
                    'ja': 'JA'       # target은 대문자 사용
                }
                
                result = self.translator.translate_text(
                    text,
                    source_lang=source_lang_map[src_lang],
                    target_lang=target_lang_map[dest_lang]
                )
                time.sleep(0.5)
                return result.text
            except Exception as e:
                print(f"DeepL 번역 중 오류 발생: {e}")
                return text

    def _back_translate(self, batch):
        """
        배치 데이터에 대한 역번역을 수행하는 메서드
        
        Args:
            batch (list): 번역할 텍스트 배치
            
        Returns:
            list: 역번역된 텍스트 리스트
        """
        translated_texts = []
        for text in batch:
            try:
                current_text = text
                for _ in range(self.loop):
                    # 한국어 => 대상 언어
                    intermediate_text = self.translate_text(current_text, 'ko', self.lang)
                    # 대상 언어 => 한국어
                    current_text = self.translate_text(intermediate_text, self.lang, 'ko')
                translated_texts.append(current_text)
            except Exception as e:
                print(f"번역 중 오류 발생: {e}")
                translated_texts.append(text)
        return translated_texts

    def augment(self, df: pd.DataFrame, augment_columns: list, id_column: str, increment_id: bool = True) -> pd.DataFrame:
        """
        데이터프레임의 선택된 컬럼에 대해 역번역 기반 데이터 증강을 수행
        
        Args:
            df (pd.DataFrame): 증강할 데이터프레임
            augment_columns (list): 증강할 컬럼명 리스트
            id_column (str): ID 역할을 할 컬럼명
            increment_id (bool): ID 자동 증가 여부 (기본값: True)
            
        Returns:
            pd.DataFrame: 원본과 증강된 데이터가 결합된 데이터프레임          
        """
        augmented_data = {}
        
        # ID 처리 개선
        try:
            last_id = df[id_column].iloc[-1]
            
            if increment_id:  # ID 증가가 선택된 경우
                # 순수 숫자인 경우
                if str(last_id).isdigit():
                    last_id_num = int(last_id)
                    new_ids = [str(i) for i in range(last_id_num + 1, last_id_num + len(df) + 1)]
                else:
                    # 문자열에서 마지막 숫자 부분 추출
                    import re
                    number_groups = re.findall(r'\d+', str(last_id))
                    if number_groups:
                        last_number = number_groups[-1]
                        prefix = str(last_id).rsplit(last_number, 1)[0]# 숫자 앞부분을 prefix로
                        suffix = str(last_id).rsplit(last_number, 1)[1]# 숫자 뒷부분을 suffix로
                        
                        new_ids = [
                            f"{prefix}{str(int(last_number) + i)}{suffix}"
                            for i in range(1, len(df) + 1)
                        ]
                    else:
                        print(f"경고: ID 컬럼 '{id_column}'에 숫자가 없어 원본 ID를 복제합니다.")
                        new_ids = [last_id] * len(df)
            else:  # ID 증가가 선택되지 않은 경우
                # 원본 ID 복제
                new_ids = [last_id] * len(df)
            
            augmented_data[id_column] = new_ids
            
        except Exception as e:
            print(f"ID 생성 중 오류 발생: {e}. 원본 ID를 복제합니다.")
            new_ids = [df[id_column].iloc[-1]] * len(df)
            augmented_data[id_column] = new_ids
        
        # 선택된 컬럼 증강
        for col in df.columns:
            if col == 'problems' and col in augment_columns:
                # problems 컬럼 특별 처리
                problems_data = df[col].apply(eval)
                target_questions = problems_data.apply(lambda x: x['question'])
                
                # question 증강
                print(f"{col} - Question Back Translation 시작...")
                question_batches = self._chunk_batch(target_questions)
                augmented_questions = []
                for batch in tqdm(question_batches):
                    augmented_questions.extend(self._back_translate(batch))
                
                # 증강된 problems 데이터 생성
                augmented_problems = []
                for q, p_data in zip(augmented_questions, problems_data):
                    problem_dict = {
                        'question': q,
                        'choices': p_data['choices'],    # 원본 유지
                        'answer': p_data['answer']       # 원본 유지
                    }
                    augmented_problems.append(str(problem_dict))
                augmented_data[col] = augmented_problems
                
            elif col in augment_columns and col != id_column:
                # 일반 컬럼 증강
                print(f"{col} Back Translation 시작...")
                batches = self._chunk_batch(df[col])
                augmented_values = []
                for batch in tqdm(batches):
                    augmented_values.extend(self._back_translate(batch))
                augmented_data[col] = augmented_values
            else:
                # 증강하지 않는 컬럼은 원본 값 유지
                augmented_data[col] = df[col].tolist()
        
        # 증강된 데이터프레임 생성
        augmented_df = pd.DataFrame(augmented_data)
        
        # 원본 데이터와 병합하여 반환
        return pd.concat([df, augmented_df], ignore_index=True)

def test_augmentation():
    """
    BackTranslator 클래스의 기능을 테스트하는 함수
    
    테스트 데이터셋을 로드하고 다양한 설정으로 데이터 증강을 수행하여
    결과를 파일로 저장
    """
    print("Back Translation 테스트 시작")

    # 테스트 데이터셋 로드
    test_df = pd.read_csv('/data/ephemeral/home/ksw/forAdditionalBT/train_augmented_google_ja_1loop.csv')
    

    # 테스트 케이스 1: Google 번역, 영어
    print("\n테스트 1: Google 번역 (한국어 -> 영어 -> 한국어)")
    bt_google = BackTranslator(
        type="google",
        loop=1,
        lang="en",
        batch_size=16
    )
    augmented_df = bt_google.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/additional_BT_Data_1044.csv', 
                        index=False)
    
    

    """
    
    # 테스트 케이스 2: Google 번역, 일본어
    print("\n테스트 2: Google 번역 (한국어 -> 일본어 -> 한국어)")
    bt_google_ja = BackTranslator(
        type="google",
        loop=1,
        lang="ja",
        batch_size=16
    )
    augmented_df2 = bt_google_ja.augment(augmented_df)
    augmented_df2.to_csv('/data/ephemeral/home/ksw/train_augmented_google_ja_1loop.csv', 
                        index=False)
    
    

    DEEPL_API_KEY = ""  # DeepL API 키 입력
    # 테스트 케이스 3: DeepL 번역, 영어
    print("\n테스트 3: DeepL 번역 (한국어 -> 영어 -> 한국어)")
    bt_deepl = BackTranslator(
        type="deepl",
        loop=1,
        lang="en",
        batch_size=16,
        deepl_api_key=DEEPL_API_KEY  # API 키 전달
    )
    augmented_df = bt_deepl.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train_augmented_deepl_en_1loop.csv', 
                        index=False)
                      

    # 테스트 케이스 4: DeepL 번역, 일본어
    print("\n테스트 4: DeepL 번역 (한국어 -> 일본어 -> 한국어)")
    bt_deepl = BackTranslator(
        type="deepl",
        loop=1,
        lang="ja",
        batch_size=16,
        deepl_api_key=DEEPL_API_KEY
    )
    augmented_df = bt_deepl.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train_augmented_deepl_ja_1loop.csv', 
                        index=False)
    """
    

if __name__ == "__main__":
    test_augmentation()
