import json
import os
import pickle
import time
import torch
import logging
import scipy
import scipy.sparse
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, NoReturn
from tqdm.auto import tqdm

import argparse
import random
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.nn.functional import normalize

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# from retrieval_tasks.utils import set_seed
from utils import set_seed

set_seed(2024)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")


class Reranker:
    def __init__(
        self,
        model_name: Optional[str] = "Dongjin-kr/ko-reranker",
        data_path: Optional[str] = "../../data/",
        context_path: Optional[str] = "wiki_documents_original.csv",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = pd.read_csv(f)

        self.contexts = list(dict.fromkeys(wiki['content']))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")   
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def retrieve(self, query, topk=1):
        result = []
        pairs = [[query, context] for context in self.contexts]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.DEVICE)
        scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        sorted_scores_idx = np.argsort(scores)[::-1]
        retrieved_docs = [self.contexts[idx] for idx in sorted_scores_idx[:topk]]
        return retrieved_docs

if __name__ == "__main__":
    reranker = Reranker() 
    query = "인플레이션은 화폐의 가치가 하락하면서 상품과 서비스의 가격이 전반적으로 상승하는 경제 현상입니다. 이는 수요 증가, 생산 비용 상승, 과도한 통화 공급 등 다양한 요인에서 비롯됩니다. 적정 수준의 인플레이션은 경제 성장에 긍정적 영향을 줄 수 있지만, 과도할 경우 구매력 저하, 경제 불안정, 소득 불균형을 초래합니다. 반대로 인플레이션이 지나치게 낮거나 디플레이션이 발생하면 경기 침체로 이어질 수 있어 중앙은행은 이를 적절히 조율하는 정책을 시행합니다."
    docs = reranker.retrieve(query)

    print(docs[0])
