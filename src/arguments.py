import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, 'data')
output_dir = os.path.join(parent_dir, 'output')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        #default='beomi/gemma-ko-2b',
        default='Qwen/Qwen2.5-32B-Instruct',
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
        default=os.path.join(data_dir, 'train.csv'),
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
        default=768,
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
        # default="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\n' + content + '<|im_end|>\n<s>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}",
        default="{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}",
        metadata={
            "help": "Chat template"
        },
    )
    prompt_question_plus : Optional[str] = field(
        default="지문:\n{paragraph}\n\n질문:\n{question}\n\n<보기>:\n{question_plus}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:",
        metadata={
            "help": "Prompt question plus"
        },
    )
    prompt_no_question_plus : Optional[str] = field(
        default="지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:",
        metadata={
            "help": "Prompt no question plus"
        },
    )
    prompt_question_plus_rag : Optional[str] = field(
        default="참고문서:\n{plus_doc}\n\n지문:\n{paragraph}\n\n질문:\n{question}\n\n<보기>:\n{question_plus}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:",
        metadata={
            "help": "Prompt question plus"
        },
    )
    prompt_no_question_plus_rag : Optional[str] = field(
        default="참고문서:\n{plus_doc}\n\n지문:\n{paragraph}\n\n질문:\n{question}\n\n선택지:\n{choices}\n\n1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.\n정답:",
        metadata={
            "help": "Prompt no question plus"
        },
    )
    response_template : Optional[str] = field(
        # default="<s>model",
        default="<start_of_turn>model",
        metadata={
            "help": "Response template"
        },
    )
    peft_config : Optional[LoraConfig] = field(
        default=LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        metadata={
            "help": "PEFT Config"
        },
    )
    quant_4_bit_config : Optional[BitsAndBytesConfig] = field(
        default=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        metadata={
            "help": "Quant 4 bit config"
        },
    )
    quant_8_bit_config : Optional[BitsAndBytesConfig] = field(
        default=BitsAndBytesConfig(
            load_in_8bit=True,
        ),
        metadata={
            "help": "Quant 8 bit config"
        },
    )
    do_rag : Optional[bool] = field(
        default=True,
        metadata={
            "help": "RAG for pred"
        },
    )
    peft_base : Optional[str] = field(
        default='beomi/gemma-ko-2b',
        metadata={
            "help": "peft base model"
        },
    )
    peft_base_chat_template : Optional[str] = field(
        default=None,
        metadata={
            "help": "peft base model"
        },
    )
    dense_model_name : Optional[str] = field(
        default='BAAI/bge-m3', #intfloat/multilingual-e5-large-instruct',
        metadata={
            "help": "dense embedding models"
        }
    )
    rag_dataset_path : Optional[str] = field(
        default="../data",
        metadata={
            "help": "The path of directory that stores contexts for RAG"
        }
    )
    rag_context_path : Optional[str] = field(
        default="wiki_docs.csv", #"wiki_documents_original.csv", 
        metadata={
            "help": "contexts for RAG"
        }
    )
    faiss_index_output_path : Optional[str] = field(
        default="2050iter_flat", 
        metadata={
            "help": "index of dpr for faiss"
        }
    )
    faiss_chunk_path : Optional[str] = field(
        default="../data/processed_passages", 
        metadata={
            "help": "chunked contexts for RAG"
        }
    )
    rag_system_prompt : Optional[str] = field(
        default="지문을 읽고 참고문서를 참고하여 질문의 답을 구하세요.",
        metadata={
            "help": "system prompt for RAG"
        }
    )
    optimize_flag : Optional[bool] = field(
        default=True,
        metadata={
            "help": "Optimize flag"
        },
    )
    gc_flag : Optional[bool] = field(
        default=True,
        metadata={
            "help": "Gradient Checkpoint flag"
        },
    )
    num_hidden_layers_ratio : Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Number of hidden layers"
        },
    )
    rag_response_threshold : Optional[int] = field(
        default=350,
        metadata={
            "help": "length of rag response threshold"
        },
    )