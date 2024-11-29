import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import logging

from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig, \
    AutoConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from tqdm import tqdm
from arguments import ModelArguments, DataTrainingArguments, CustomArguments

from retrieval_tasks.retrieval_hybrid import HybridSearch
from retrieval_tasks.retrieval_rerank import Reranker
from retrieval_tasks.retrieve_utils import retrieve

import wandb

from utils import record_to_df, train_df_to_process_df, test_df_to_process_df, set_seed, optimize_model, apply_lora, remove_lora, train_df_to_process_df_with_rag, test_df_to_process_df_with_rag

SEED = 42
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parent_dir = os.path.dirname(os.getcwd())
now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

pd.set_option('display.max_columns', None)
os.environ["WANDB_MODE"] = "offline"
logger = logging.getLogger(__name__)

def main(run_name, debug=False):
    set_seed(SEED)

    model_name = None
    adaptor = None
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    model_args, data_args, train_args, custom_args = parser.parse_args_into_dataclasses()
    plus_prompt, no_plus_prompt = custom_args.prompt_question_plus, custom_args.prompt_no_question_plus
    plus_prompt_rag, no_plus_prompt_rag = custom_args.prompt_question_plus_rag, custom_args.prompt_no_question_plus_rag

    project_prefix = "[train]" if train_args.do_train else "[eval]" if train_args.do_eval else "[pred]"
    wandb.init(
        project="CSAT-Solver",
        entity="NotyNoty",
        name=f"{project_prefix}_{run_name}",
        save_code=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"model is from {model_args.model_name_or_path}")
    logging.info(f"data is from {data_args.dataset_name}")

    # Load data
    dataset = pd.read_csv(data_args.dataset_name)
    dataset = dataset.sample(10, random_state=SEED).reset_index(drop=True) if debug else dataset
    dataset.to_csv(parent_dir, 'data', f"sampled_dataset_{now}.csv") if debug else None

    df = record_to_df(dataset)

    quant_config = None
    if custom_args.quantization == 4:
        quant_config = custom_args.quant_4_bit_config

    elif custom_args.quantization == 8:
        quant_config = custom_args.quant_8_bit_config

    # Load model
    if train_args.do_train:
        model_name = model_args.model_name_or_path
        config = AutoConfig.from_pretrained(model_name)
        config = optimize_model(config, data_args, custom_args) if custom_args.optimize_flag else config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float32 if not isinstance(quant_config, BitsAndBytesConfig) else None,
            trust_remote_code=True,
            quantization_config=quant_config if isinstance(quant_config, BitsAndBytesConfig) else None,
            device_map="auto",
        )
        
        # 그래디언트 체크포인팅 활성화
        if custom_args.gc_flag:
            model.gradient_checkpointing_enable()

    if not train_args.do_train and not custom_args.do_RAG:
        latest_ckpt = sorted(os.listdir(model_args.model_name_or_path))[-1]
        model_name = os.path.join(model_args.model_name_or_path, latest_ckpt)
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if not isinstance(quant_config, BitsAndBytesConfig) else None,
            trust_remote_code=True,
            quantization_config=quant_config if isinstance(quant_config, BitsAndBytesConfig) else None,
        )

    if not train_args.do_train and custom_args.do_RAG:
        latest_ckpt = sorted(os.listdir(model_args.model_name_or_path))[-1]
        adaptor = os.path.join(model_args.model_name_or_path, latest_ckpt)
        model_name = custom_args.peft_base
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto" if not isinstance(quant_config, BitsAndBytesConfig) else None,
            trust_remote_code=True,
            quantization_config=quant_config if isinstance(quant_config, BitsAndBytesConfig) else None,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    custom_args.peft_base_chat_template = tokenizer.chat_template
    tokenizer.chat_template = custom_args.chat_template
    peft_config = custom_args.peft_config
    if not train_args.do_train and custom_args.do_RAG:
        model = apply_lora(model, adaptor)

    if custom_args.do_RAG:
            dense_model_name = []
            dense_model_name.append(custom_args.dense_model_name)
            retriever = HybridSearch(
                        tokenize_fn=tokenizer.tokenize,
                        dense_model_name=dense_model_name,
                        data_path=custom_args.RAG_dataset_path,
                        context_path=custom_args.RAG_context_path,
                    )
            retriever.get_dense_embedding()
            retriever.get_sparse_embedding()

    dataset = Dataset.from_pandas(df)
    if not custom_args.do_RAG:
        processed_dataset = train_df_to_process_df(dataset, plus_prompt, no_plus_prompt)
    if custom_args.do_RAG:
        processed_dataset = train_df_to_process_df_with_rag(dataset, plus_prompt_rag, no_plus_prompt_rag, retriever, model, tokenizer, adaptor, custom_args, data_args)

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

    mex_seq_len = data_args.max_seq_length

    # 모델의 logits 를 조정하여 정답 토큰 부분만 출력하도록 설정
    def preprocess_logits_for_metrics(logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [tokenizer.vocab["1"], tokenizer.vocab["2"], tokenizer.vocab["3"], tokenizer.vocab["4"],
                     tokenizer.vocab["5"]]
        logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
        return logits

    # metric 로드
    acc_metric = evaluate.load("accuracy")
    # metric 계산 함수
    def compute_metrics(evaluation_result):
        logits, labels = evaluation_result

        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
        # 실제 지문은 1에서 시작, 인덱스는 0부터 시작하므로
        labels = list(map(lambda x: int(x) - 1, labels))

        # 소프트맥스 함수를 사용하여 로그트 변환
        probs = torch.nn.functional.softmax(torch.FloatTensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        # 정확도 계산
        acc = acc_metric.compute(predictions=predictions, references=labels)
        return acc

    try:
        if train_args.do_train:
            dataset = Dataset.from_pandas(df)
            processed_dataset = train_df_to_process_df(dataset, plus_prompt, no_plus_prompt)

            tokenizer.chat_template = custom_args.chat_template
            # 데이터 토큰화
            tokenized_dataset = processed_dataset.map(
                tokenize,
                remove_columns=list(processed_dataset.features),
                batched=True,
                num_proc=4,
                load_from_cache_file=True,
                desc="Tokenizing",
            )

            # vram memory 제약으로 인해 인풋 데이터의 길이가 1024 초과인 데이터는 제외하였습니다. 1024보다 길이가 더 긴 데이터를 포함하면 더 높은 점수를 달성할 수 있을 것 같습니다.
            tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= data_args.max_seq_length)
            tokenized_dataset = tokenized_dataset.train_test_split(test_size=model_args.train_test_split, seed=SEED)

            train_dataset = tokenized_dataset['train']
            eval_dataset = tokenized_dataset['test']

            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=custom_args.response_template,
                tokenizer=tokenizer,
            )

            # 모델 설정, pad token 설정
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = 'right'

            sft_config = SFTConfig(
                output_dir=train_args.output_dir,
                do_train=True,
                do_eval=True,
                lr_scheduler_type="cosine",
                max_seq_length=data_args.max_seq_length,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                num_train_epochs=10,
                learning_rate=2e-5,
                weight_decay=0.01,
                logging_steps=200,
                save_strategy="epoch",
                #eval_strategy="epoch",
                save_total_limit=1,
                save_only_model=True,
                report_to="wandb",
                gradient_checkpointing=custom_args.gc_flag,  # 그래디언트 체크포인팅 활성화
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                peft_config=custom_args.peft_config,
                args=sft_config,
            )
            trainer.train()

        if train_args.do_predict:
            test_df = pd.read_csv(data_args.test_dataset_name)
            test_df = record_to_df(test_df)
            if not custom_args.do_RAG:
                test_dataset = test_df_to_process_df(test_df, plus_prompt, no_plus_prompt)
            if custom_args.do_RAG:
                test_dataset = test_df_to_process_df_with_rag(test_df, plus_prompt_rag, no_plus_prompt_rag, retriever, model, tokenizer, adaptor, custom_args, data_args)
            
            infer_results = []
            pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

            model.to(DEVICE)
            model.eval()
            with torch.inference_mode():
                for data in tqdm(test_dataset):
                    _id = data["id"]
                    messages = data["messages"]
                    len_choices = data["len_choices"]

                    input = tokenizer.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(DEVICE)

                    outputs = model(input)

                    logits = outputs.logits[:, -1].flatten().cpu()

                    target_logit_list = [logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

                    probs = (
                        torch.nn.functional.softmax(torch.tensor(target_logit_list, dtype=torch.float32),
                                                    dim=-1).detach().cpu().numpy()
                    )

                    predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                    infer_results.append({"id": _id, "answer": predict_value})

            os.makedirs(train_args.output_dir, exist_ok=True)
            pd.DataFrame(infer_results).to_csv(os.path.join(train_args.output_dir, 'predictions.csv'), index=False)
            print(pd.DataFrame(infer_results))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e

    finally:
        torch.cuda.empty_cache()

    return


if __name__ == '__main__':
    try:
        argv_run_index = sys.argv.index('--run_name') + 1
        argv_run_name = sys.argv[argv_run_index]

    except ValueError:
        argv_run_name = ''
        while argv_run_name == '':
            argv_run_name = input("run name is missing, please add run name : ")

    main(argv_run_name, debug=True)
    # main(argv_run_name)
