import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging
import torch
import wandb
import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, Trainer
from datasets import Dataset
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig
from arguments import ModelArguments, DataTrainingArguments

pd.set_option('display.max_columns', None)

os.environ["WANDB_MODE"] = "offline"
logger = logging.getLogger(__name__)

def set_seed(seed: int = 456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


SEED = 2024
set_seed(SEED)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def main(run_name):
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    model_name = None

    project_prefix = "[train]" if train_args.do_train else "[eval]" if train_args.do_eval else "[pred]"
    # wandb.init(
    #     project="data_centric",
    #     entity="nlp15",
    #     name=f"{project_prefix}_{run_name}",
    #     save_code=True,
    # )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"model is from {model_args.model_name_or_path}")
    logging.info(f"data is from {data_args.dataset_name}")

    # Load model
    if train_args.do_train:
        model_name = model_args.model_name_or_path

    if not train_args.do_train:
        latest_ckpt = sorted(os.listdir(model_args.model_name_or_path))[-1]
        model_name = os.path.join(model_args.model_name_or_path, latest_ckpt)

    # Load data
    dataset = pd.read_csv(data_args.dataset_name)

    # Flatten the JSON dataset
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

    # Convert to DataFrame
    df = pd.DataFrame(records)
    #print(df.head())
    #print("\nMissing values in each column:")
    #print(df.isnull().sum())
    #print("\nDataset Information:")
    #print(df.info())

    # Combine 'question' and 'question_plus' if available
    df['question_plus'] = df['question_plus'].fillna('')
    df['full_question'] = df.apply(
        lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'], axis=1)

    # Calculate the length of each question
    df['question_length'] = df['full_question'].apply(len)

    # plt.figure(figsize=(5, 3))
    # plt.hist(df['question_length'], bins=30, edgecolor='black', alpha=0.7)
    # plt.title('Distribution of Question Lengths')
    # plt.xlabel('Question Length')
    # plt.ylabel('Frequency')
    # plt.show()

    # dataset_train, dataset_valid = train_test_split(data, test_size=model_args.train_test_split, random_state=SEED)
    #
    # data_train = BERTDataset(dataset_train, tokenizer, data_args)
    # data_valid = BERTDataset(dataset_valid, tokenizer, data_args)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #
    # if train_args.do_train:
    #     model = train(model, data_train, data_valid, data_collator, train_args)
    # if train_args.do_predict:
    #     predict(model, tokenizer, train_args, data_args)

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['full_question'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # print("\nTF-IDF Features:")
    # print(tfidf_df.head(20))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ) if train_args.do_train else (
        AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"

    peft_config = LoraConfig(
        r=6,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_pandas(df)

    PROMPT_NO_QUESTION_PLUS = """지문:
    {paragraph}

    질문:
    {question}

    선택지:
    {choices}

    1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
    정답:"""

    PROMPT_QUESTION_PLUS = """지문:
    {paragraph}

    질문:
    {question}

    <보기>:
    {question_plus}

    선택지:
    {choices}

    1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
    정답:"""

    processed_dataset = []
    for i in range(len(dataset)):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

        # <보기>가 있을 때
        if dataset[i]["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
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

    #print(processed_dataset[0])
    processed_dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
    #print(processed_dataset)

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # 데이터 토큰화
    tokenized_dataset = processed_dataset.map(
        tokenize,
        remove_columns=list(processed_dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

    # 데이터 분리
    # vram memory 제약으로 인해 인풋 데이터의 길이가 1024 초과인 데이터는 제외하였습니다. *힌트: 1024보다 길이가 더 긴 데이터를 포함하면 더 높은 점수를 달성할 수 있을 것 같습니다!
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']

    # 데이터 확인
    # print(tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=True))
    # print(tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False))
    #
    # train_dataset_token_lengths = [len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))]
    # print(f"max token length: {max(train_dataset_token_lengths)}")
    # print(f"min token length: {min(train_dataset_token_lengths)}")
    # print(f"avg token length: {np.mean(train_dataset_token_lengths)}")
    #
    # print(tokenizer.chat_template)

    response_template = "<start_of_turn>model"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"],
                     tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
        return logits

    # metric 로드
    acc_metric = evaluate.load("accuracy")

    # 정답 토큰 매핑
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    # metric 계산 함수
    def compute_metrics(evaluation_result):
        logits, labels = evaluation_result

        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        labels = list(map(lambda x: int_output_map[x], labels))

        # 소프트맥스 함수를 사용하여 로그트 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 정확도 계산
        acc = acc_metric.compute(predictions=predictions, references=labels)
        return acc

    # 모델 설정, pad token 설정
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    #print(tokenizer.special_tokens_map)

    tokenizer.padding_side = 'right'

    if train_args.do_train:
        sft_config = SFTConfig(
            output_dir=train_args.output_dir,
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine",
            max_seq_length=1024,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=200,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            save_only_model=True,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            args=sft_config,
        )

        trainer.train()

    if train_args.do_predict:
        test_df = pd.read_csv(data_args.test_dataset_name)

        # Flatten the JSON dataset
        records = []
        for _, row in test_df.iterrows():
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

        # Convert to DataFrame
        test_df = pd.DataFrame(records)

        test_dataset = []
        for i, row in test_df.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])
            len_choices = len(row["choices"])

            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = PROMPT_NO_QUESTION_PLUS.format(
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

        infer_results = []
        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

        model.eval()
        with torch.inference_mode():
            for data in tqdm(test_dataset):
                _id = data["id"]
                messages = data["messages"]
                len_choices = data["len_choices"]

                outputs = model(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to("cuda")
                )

                logits = outputs.logits[:, -1].flatten().cpu()

                target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32)
                    ).detach().cpu().numpy()
                )

                predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                infer_results.append({"id": _id, "answer": predict_value})

        pd.DataFrame(infer_results).to_csv(train_args.output_dir, index=False)
        print(pd.DataFrame(infer_results))


def train(model, data_train, data_valid, data_collator, train_args: TrainingArguments):
    # output_path = os.path.join(MODEL_DIR, run_name)

    training_args = TrainingArguments(
        output_dir=train_args.output_dir,
        overwrite_output_dir=True,
        do_train=train_args.do_train,
        do_eval=train_args.do_eval,
        do_predict=train_args.do_predict,
        logging_strategy='steps',
        eval_strategy='steps',
        save_strategy='steps',
        logging_steps=200,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,

        # 수정 안됨
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        num_train_epochs=2,

        # 수정 가능
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )

    f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average='macro')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model


def predict(model, tokenizer, train_args: TrainingArguments, data_args: DataTrainingArguments = None):
    dataset_test = pd.read_csv(data_args.test_dataset_name)
    model.eval()
    preds = []

    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    dataset_test['target'] = preds

    os.makedirs(train_args.output_dir, exist_ok=True)
    dataset_test.to_csv(os.path.join(train_args.output_dir, 'predictions.csv'), index=False)


if __name__ == '__main__':
    try:
        argv_run_index = sys.argv.index('--run_name') + 1
        argv_run_name = sys.argv[argv_run_index]

    except ValueError:
        argv_run_name = ''
        while argv_run_name == '':
            argv_run_name = input("run name is missing, please add run name : ")

    main(argv_run_name)