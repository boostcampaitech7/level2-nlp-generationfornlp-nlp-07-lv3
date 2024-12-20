import os
import time

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer

from src.utils import record_to_df, test_df_to_process_df
from src.arguments import CustomArguments

parent_dir = os.path.dirname(os.getcwd())
output_root_dir = os.path.join(parent_dir, 'ensemble')

now = time.strftime('%Y%m%d_%H%M%S')
SOTA_WEIGHT = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def ensemble_by_voting():
    output_root_dir_ = os.listdir(os.path.join(output_root_dir, 'voting'))
    output_root_dir_.remove('.gitkeep')

    mapper = {}

    for output_dir in output_root_dir_:
        path = os.path.join(output_root_dir, output_dir, 'predictions.csv')

        if os.path.exists(path) and path.endswith('.csv'):
            dataset = pd.read_csv(path)

            weight = SOTA_WEIGHT if 'SOTA' in path else 1

            for i, row in dataset.iterrows():
                if row['id'] in mapper:
                    mapper[row['id']][row['answer']] += weight

                else:
                    mapper[row['id']] = {1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0}
                    mapper[row['id']][row['answer']] += weight

    new_df = pd.DataFrame(columns=['id', 'answer'])

    for key in mapper:
        answer = max(mapper[key], key=mapper[key].get)
        new_df = pd.concat([new_df, pd.DataFrame({'id': [key], 'answer': [answer]})], ignore_index=True)

    new_df.to_csv(os.path.join(parent_dir, 'output', 'ensemble', 'ensemble_results', f'voting_{now}.csv'), index=False)


def ensemble_by_softmax():
    output_root_dir_ = os.listdir(os.path.join(output_root_dir, 'softmax'))
    output_root_dir_.remove('.gitkeep')

    test_df = pd.read_csv(os.path.join(parent_dir, 'data', 'test.csv'))

    test_df = record_to_df(test_df)
    test_dataset = test_df_to_process_df(test_df, CustomArguments.prompt_question_plus, CustomArguments.prompt_no_question_plus)

    infer_results = pd.DataFrame(columns=['id', 'answer'])
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    quant_config = CustomArguments.quant_4_bit_config
    model_list = []
    pred_choice_mapper = {}

    prob_weight_list = []

    for output_dir in output_root_dir_:
        latest_ckpt = os.listdir(os.path.join(output_root_dir, 'softmax', output_dir))[-1]
        model_list.append(os.path.join(output_root_dir, 'softmax', output_dir, latest_ckpt))

    if len(prob_weight_list) == 0:
        prob_weight_list = [1 for _ in range(len(model_list))]

    elif len(prob_weight_list) != len(model_list):
        raise ValueError("The length of prob_weight_list must be the same as the length of model_list.")

    for model_name in model_list:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=None,
            trust_remote_code=True,
            quantization_config=quant_config if isinstance(quant_config, BitsAndBytesConfig) else None,
        )

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

                if _id in pred_choice_mapper:
                    pred_choice_mapper[_id] += probs * prob_weight_list[model_list.index(model_name)]

                else:
                    pred_choice_mapper[_id] = probs * prob_weight_list[model_list.index(model_name)]

        torch.cuda.empty_cache()

    for _id, probs in pred_choice_mapper.items():
        predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
        infer_results = pd.concat([infer_results, pd.DataFrame({'id': [_id], 'answer': [predict_value]})], ignore_index=True)

    infer_results.to_csv(os.path.join(parent_dir, 'ensemble', 'ensemble_results', f'softmax_{now}.csv'), index=False)


if __name__ == '__main__':
    #ensemble_by_voting()
    ensemble_by_softmax()