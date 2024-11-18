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
        #default='beomi/gemma-ko-2b',
        #default='LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct',
        #default='Qwen/Qwen2.5-0.5B-Instruct',
        default='Qwen/Qwen2.5-14B-Instruct',
        #default='CohereForAI/aya-expanse-32b',
        #default='bartowski/aya-expanse-32b-GGUF',
        #default='bartowski/aya-expanse-32b-GGUF',
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
