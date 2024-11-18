from dataclasses import dataclass, field
from typing import Optional

import os
parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, 'data')
output_dir = os.path.join(parent_dir, 'output')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='beomi/gemma-ko-2b',
    )
    train_test_split: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "test_size"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        #default=os.path.join(data_dir, 'train.csv'),
        metadata={
            "help": "The name of the dataset to use."
        },
    )
    test_dataset_name: Optional[str] = field(
        default=os.path.join(data_dir, 'test.csv'),
        metadata={
            "help": "The name of the dataset to use."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )

@dataclass
class CustomArguments:
    quantization : Optional[int] = field(
        default=None,
        metadata={
            "help": "Quantization level"
        },
    )
    chat_template : Optional[str] = field(
        default="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}",
        metadata={
            "help": "Chat template"
        },
    )
    prompt_question_plus : str = field(
        default="지문:\n{paragraph}\n\n질문:\n{question}\n\n<보기>:\n{question_plus}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:",
        metadata={
            "help": "Prompt question plus"
        },
    )
    prompt_no_question_plus : str = field(
        default="지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:",
        metadata={
            "help": "Prompt no question plus"
        },
    )
    response_template : str = field(
        default="<start_of_turn>model",
        metadata={
            "help": "Response template"
        },
    )