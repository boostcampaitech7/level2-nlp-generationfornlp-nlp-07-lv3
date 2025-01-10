import logging
import os
import time

import pandas as pd
import torch
from contextlib import contextmanager
from datasets import Dataset
from typing import  Optional, NoReturn
from tqdm.auto import tqdm

from .retrieval_hybrid import HybridSearch
from .retrieval import Retrieval
from utils import set_seed

set_seed(2024)
torch.use_deterministic_algorithms(False)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")


class Reranker(Retrieval):
    def __init__(
        self,
        tokenize_fn,
        dense_model_name: list = ['intfloat/multilingual-e5-large-instruct','upskyy/bge-m3-korean'],
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wiki_documents_original.csv",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = pd.read_csv(f)

        self.contexts = list(dict.fromkeys(wiki['content']))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.embeder = HybridSearch(
            tokenize_fn=tokenize_fn,
            dense_model_name=dense_model_name[0],
            data_path=data_path,
            context_path=context_path,
        )
        self.embeder.get_dense_embedding()
        self.embeder.get_sparse_embedding()

        if len(dense_model_name) == 2:
            self.embeder2 = HybridSearch(
                tokenize_fn=tokenize_fn,
                dense_model_name=dense_model_name[1],
                data_path=data_path,
                context_path=context_path,
            )
        elif len(dense_model_name) == 1:
            self.embeder2 = HybridSearch(
                tokenize_fn=tokenize_fn,
                dense_model_name=dense_model_name[0],
                data_path=data_path,
                context_path=context_path,
            )

    def retrieve_first(self, queries, topk: Optional[int] = 1, alpha: Optional[int]=0):
        f_df = self.embeder.retrieve(queries, topk=topk, alpha=alpha)
        return f_df
    
    def retireve_second(self, queries, topk: Optional[int] = 1, contexts=None, alpha: Optional[int]=0):
        self.embeder2.get_dense_embedding(contexts=contexts)
        self.embeder2.get_sparse_embedding(contexts=contexts)
        s_df = self.embeder2.retrieve(queries, topk=topk, alpha=alpha)
        return s_df

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1, alpha_1: Optional[int] = 0, alpha_2: Optional[int] = 0):
        retrieved_contexts = []
        if isinstance(query_or_dataset, str):
            _, doc_contexts = self.retrieve_first(query_or_dataset, topk, alpha=alpha_1)
            retrieved_contexts = doc_contexts
        elif isinstance(query_or_dataset, Dataset):
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Rerank first retrieval]: ")):
                _, doc_contexts = self.retrieve_first(example['question'], topk, alpha=alpha_1)
                retrieved_contexts.append(doc_contexts)

        half_topk = 5 

        if isinstance(query_or_dataset, str):
            second_df = self.retireve_second(query_or_dataset, half_topk, contexts=retrieved_contexts, alpha=alpha_2)
            return second_df
        elif isinstance(query_or_dataset, Dataset):
            second_df = []
            for i, example in enumerate(tqdm(query_or_dataset, desc="[Rerank second retrieval] ")):
                context = retrieved_contexts[i]
                doc_scores, doc_contexts = self.retireve_second(example['question'], half_topk, contexts=context, alpha=alpha_2)
                template = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(doc_contexts),
                }
                second_df.append(template)
            second_df = pd.DataFrame(second_df)
            return second_df

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train.csv", type=str)
    parser.add_argument("--model_name_or_path", default="yanolja/EEVE-Korean-Instruct-10.8B-v1.0", type=str)
    parser.add_argument("--data_path", default="../data", type=str)
    parser.add_argument("--context_path", default="wiki_documents_original.csv", type=str)
    parser.add_argument("--use_faiss", default=False, type=bool)

    args = parser.parse_args()
    logging.info(args.__dict__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    retriever = Reranker(
        tokenize_fn=tokenizer.tokenize,
        dense_model_name=['intfloat/multilingual-e5-large-instruct', "upskyy/bge-m3-korean"],
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = ""
   
    with timer("single query by exhaustive search using reranker"):
        scores, contexts = retriever.retrieve(query, topk=200)
   
    for i, context in enumerate(contexts):
        logging.info(f"Top-{i + 1} 의 문서")
        logging.info("---------------------------------------------")
        logging.info(context)


