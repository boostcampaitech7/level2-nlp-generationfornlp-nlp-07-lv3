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
from .retrieval_syntactic import Syntactic
from .retrieval_semantic import Semantic
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
        step1_model: Retrieval,
        step2_model: Retrieval,
    ) -> NoReturn:
        self.step1_model = step1_model
        self.step2_model = step2_model

    def _step1_retrieve(self, queries, topk: Optional[int] = 1, alpha: Optional[int]=0.5):
        if isinstance(self.step1_model, HybridSearch):
            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step1_model.retrieve(queries, topk=topk, alpha=alpha)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step1_model.retrieve(queries, topk=topk, alpha=alpha)
                return datas
        else:
            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step1_model.retrieve(queries, topk=topk)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step1_model.retrieve(queries, topk=topk)
                return datas
    
    def _step2_retrieve(self, queries, contexts, topk: Optional[int] = 1, alpha: Optional[int]=0.5):
        if isinstance(self.step2_model, HybridSearch):
            if isinstance(self.step2_model.step1_model, Syntactic):
                self.step2_model.step1_model.contexts = contexts
                self.step2_model.step1_model.get_sparse_embedding(vectorizer_path=f"data/rerank_syntactic_{self.step2_model.step2_model.vectorizer_type}.bin")
            elif isinstance(self.step2_model.step1_model, Semantic):
                self.step2_model.step1_model.get_dense_embedding_with_faiss(contexts)

            if isinstance(self.step2_model.step2_model, Syntactic):
                self.step2_model.step2_model.contexts = contexts
                self.step2_model.step2_model.get_sparse_embedding(vectorizer_path=f"data/rerank_syntactic_{self.step2_model.step2_model.vectorizer_type}.bin")
            elif isinstance(self.step2_model.step2_model, Semantic):
                self.step2_model.step2_model.get_dense_embedding_with_faiss(contexts)

            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step2_model.retrieve(queries, topk=topk, alpha=alpha)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step2_model.retrieve(queries, topk=topk, alpha=alpha)
                return datas
        else:
            if isinstance(self.step2_model, Syntactic):
                self.step2_model.contexts = contexts
                self.step2_model.get_sparse_embedding(vectorizer_path=f"data/rerank_syntactic_{self.step2_model.vectorizer_type}.bin")
            elif isinstance(self.step2_model, Semantic):
                self.step2_model.get_dense_embedding_with_faiss(contexts)

            if isinstance(queries, str):
                doc_scores, doc_contexts = self.step2_model.retrieve(queries, topk=topk)
                return doc_scores, doc_contexts
            elif isinstance(queries, Dataset):
                datas = self.step2_model.retrieve(queries, topk=topk)
                return datas

    def retrieve(self, query_or_dataset, topk: Optional[int] = 5, alpha_1: Optional[int] = 0, alpha_2: Optional[int] = 0):
        retrieved_contexts = []
        if isinstance(query_or_dataset, str):
            _, doc_contexts = self._step1_retrieve(query_or_dataset, topk, alpha=alpha_1)
            retrieved_contexts = doc_contexts
        elif isinstance(query_or_dataset, Dataset):
            for _, example in enumerate(tqdm(query_or_dataset, desc="[Rerank first retrieval]: ")):
                doc_contexts = self._step1_retrieve(example['question'], topk, alpha=alpha_1)
                retrieved_contexts.append(doc_contexts)

        half_topk = int(topk / 2)

        if isinstance(query_or_dataset, str):
            second_retrieved_scores, second_retrieved_contexts = self._step2_retrieve(query_or_dataset, retrieved_contexts, half_topk, alpha=alpha_2)
            return (second_retrieved_scores, second_retrieved_contexts)
        elif isinstance(query_or_dataset, Dataset):
            second_retrieved_contexts = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Rerank second retrieval] ")):
                context = retrieved_contexts[idx]
                doc_contexts = self._step2_retrieve(example['question'], context, half_topk, alpha=alpha_2)
                template = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(doc_contexts),
                }
                second_retrieved_contexts.append(template)
            second_retrieved_contexts = pd.DataFrame(second_retrieved_contexts)
            return second_retrieved_contexts

if __name__ == "__main__":
    pass
