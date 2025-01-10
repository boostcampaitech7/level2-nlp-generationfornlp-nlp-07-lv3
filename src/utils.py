import random
from ast import literal_eval

import pandas as pd
import numpy as np
import torch
import transformers
from datasets import Dataset
from tqdm import tqdm
from peft import PeftModel

from arguments import DataTrainingArguments, CustomArguments
from retrieval_tasks.retrieve import retrieve

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def apply_lora(model, adaptor_path):
    lora_model = PeftModel.from_pretrained(model, adaptor_path)
    return lora_model

def remove_lora(model):
    vanilla_model = model.merge_and_unload()
    return vanilla_model

def record_to_df(dataset : pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)

    return pd.DataFrame(records)

def train_df_to_process_df(dataset : pd.DataFrame, q_plus, no_q_plus) -> Dataset:
    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

        # <보기>가 있을 때
        if dataset[i]["question_plus"]:
            user_message = q_plus.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = no_q_plus.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )

        # chat message 형식으로 변환
        processed_dataset.append(
            {
                "id": dataset[i]["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                ],
                "label": dataset[i]["answer"],
            }
        )

    return Dataset.from_pandas(pd.DataFrame(processed_dataset))

def test_df_to_process_df(dataset : pd.DataFrame, q_plus, no_q_plus) -> list:
    test_dataset = []
    for i, row in dataset.iterrows():
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = q_plus.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = no_q_plus.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )

    return test_dataset

def optimize_model(config : transformers.AutoConfig, data_args, custom_args):
    config.use_cache = False
    config.max_position_embeddings = data_args.max_seq_length
    config.num_hidden_layers = int(custom_args.num_hidden_layers_ratio * config.num_hidden_layers)

    return config

def train_df_to_process_df_with_rag(
        dataset, 
        q_plus, 
        no_q_plus, 
        retriever, 
        model, 
        tokenizer, 
        adaptor_path, 
        custom_args: CustomArguments, 
        data_args: DataTrainingArguments
    ):
    processed_dataset = []

    def rag_process(retriever, model, tokenizer, message, max_seq_length, custom_args: CustomArguments):
        # model = remove_lora(model)
        # tokenizer.chat_template = default_chat_template 
        retrieved_contexts_summary = retrieve(retriever, model, tokenizer, message, max_seq_length, custom_args, topk=2)
        # model = apply_lora(model, adaptor_path)
        # tokenizer.chat_template = custom_args.chat_template
        return retrieved_contexts_summary

    for i in tqdm(range(len(dataset)),desc="[Ragging..]" ,total=len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

        # <보기>가 있을 때
        if dataset[i]["question_plus"]:
            user_message = q_plus.format(
                plus_doc="None",
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )

            retrieved_contexts_summary = rag_process(retriever, model, tokenizer, user_message, data_args.max_seq_length, custom_args)
            
            if retrieved_contexts_summary is not None:
                user_message = q_plus.format(
                    plus_doc=retrieved_contexts_summary,
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )
        # <보기>가 없을 때
        else:
            user_message = no_q_plus.format(
                plus_doc="None",
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )

            retrieved_contexts_summary = rag_process(retriever, model, tokenizer, user_message, data_args.max_seq_length, custom_args)
            
            if retrieved_contexts_summary is not None:
                user_message = q_plus.format(
                    plus_doc=retrieved_contexts_summary,
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )

        # chat message 형식으로 변환
        processed_dataset.append(
            {
                "id": dataset[i]["id"],
                "messages": [
                    {"role": "system", "content": custom_args.rag_System_prompt},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{dataset[i]['answer']}"}
                ],
                "label": dataset[i]["answer"],
            }
        )

    return Dataset.from_pandas(pd.DataFrame(processed_dataset))

def test_df_to_process_df_with_rag(
        dataset, 
        q_plus, 
        no_q_plus, 
        retriever, 
        model, 
        tokenizer, 
        adaptor_path, 
        custom_args: CustomArguments, 
        data_args: DataTrainingArguments
    ):
    test_dataset = []

    def rag_process(retriever, model, tokenizer, message, max_seq_length, custom_args: CustomArguments):
        # model = remove_lora(model)
        # tokenizer.chat_template = default_chat_template 
        retrieved_contexts_summary = retrieve(retriever, model, tokenizer, message, max_seq_length, custom_args, topk=2)
        # model = apply_lora(model, adaptor_path)
        # tokenizer.chat_template = custom_args.chat_template
        return retrieved_contexts_summary

    for i, row in tqdm(dataset.iterrows(), desc="[Ragging..]", total=len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
        len_choices = len(row["choices"])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = q_plus.format(
                plus_doc="None",
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )

            retrieved_contexts_summary = rag_process(retriever, model, tokenizer, user_message, data_args.max_seq_length, custom_args)
            
            if retrieved_contexts_summary is not None:
                user_message = q_plus.format(
                    plus_doc=retrieved_contexts_summary,
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )

        # <보기>가 없을 때
        else:
            user_message = no_q_plus.format(
                plus_doc="None",
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

            retrieved_contexts_summary = rag_process(retriever, model, tokenizer, user_message, data_args.max_seq_length, custom_args)
            
            if retrieved_contexts_summary is not None:
                user_message = q_plus.format(
                    plus_doc=retrieved_contexts_summary,
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": custom_args.rag_System_prompt},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )

    return test_dataset

