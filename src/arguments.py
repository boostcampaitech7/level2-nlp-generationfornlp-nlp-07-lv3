from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='klue/bert-base'
    )
    train_test_split: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "test_size"
        },
    )
    learning_rate: Optional[float] = field(
        default=2e-5,
        metadata={
            "help": "The initial learning rate for training."
        },
    )
    per_device_train_batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": "The batch size per GPU/TPU core/CPU for training."
        },
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": "The batch size per GPU/TPU core/CPU for evaluation."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="../data/train.csv",
        metadata={
            "help": "The name of the dataset to use."
        },
    )
    test_dataset_name: Optional[str] = field(
        default="../data/test.csv",
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
    data_path: str = field(
        default="../data/",
        metadata={
            "help": "The path of the data directory"
        },
    )
    train_learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for training"}
    )
##################################################################################################################
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
