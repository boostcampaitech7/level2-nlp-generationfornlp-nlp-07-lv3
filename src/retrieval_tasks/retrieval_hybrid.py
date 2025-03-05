import os
import time
import logging

import pickle
import torch
import scipy
import scipy.sparse
from contextlib import contextmanager
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.nn.functional import normalize
from typing import List, Optional, Tuple, NoReturn
from tqdm.auto import tqdm

from .retrieval import Retrieval
from .retrieval_syntactic import Syntactic
from .retrieval_semantic import Semantic
from utils import set_seed

set_seed(2024)
torch.use_deterministic_algorithms(False)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")

class HybridSearch(Retrieval):
    def __init__(
        self,
        tokenize_fn,
        step1_model: Retrieval,
        step2_model: Retrieval,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wiki_docs.csv",
    ) -> NoReturn:
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenize_fn = tokenize_fn

        self.step1_model = step1_model
        self.step2_model = step2_model

        self.data_path = data_path
        if isinstance(self.step1_model, Syntactic):
            self.contexts = self.step1_model.contexts
        elif isinstance(self.step2_model, Syntactic):
            self.contexts = self.step2_model.contexts
        else:
            with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
                wiki = pd.read_csv(f)
            self.contexts = list(dict.fromkeys(wiki['content']))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
    
    def get_step1_embedding(self, query):
        if isinstance(self.step1_model, Syntactic):
            return self.step1_model.transform(query)
        elif isinstance(self.step1_model, Semantic):
            if not isinstance(query, List):
                query = [query]
            sentence_embeddings = self.step1_model.transform(query)
            sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()

    def get_step2_embedding(self, query):
        if isinstance(self.step2_model, Syntactic):
            return self.step2_model.transform(query)
        elif isinstance(self.step2_model, Semantic):
            if not isinstance(query, List):
                query = [query]
            sentence_embeddings = self.step2_model.transform(query)
            sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()
        
    def hybrid_scale(self, step1_model_score, step2_model_score, alpha: float):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if isinstance(step1_model_score, torch.Tensor):
            step1_model_score = step1_model_score.detach().numpy() 
        if isinstance(step2_model_score, torch.Tensor):
            step2_model_score = step2_model_score.detach().numpy()  

        step1_model_score = np.array(step1_model_score)
        step2_model_score = np.array(step2_model_score)

        result = (1 - alpha) * step1_model_score + alpha * step2_model_score
        return result

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1, alpha: Optional[float] = 0.5):
        assert self.step1_model is not None, "You should first init `step1_model`"
        assert self.step2_model is not None, "You should first init `step2_model`"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, alpha, k=topk)
            logging.info(f"[Search query] {query_or_dataset}")

            for i in range(topk):
                logging.info(f"Top-{i+1} passage with score {doc_scores[i]}")
                logging.info(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], alpha, k=topk
                )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Hybrid retrieval] ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, alpha: float, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("query ex search"):
            if isinstance(self.step1_model, Syntactic):
                if self.step1_model.vectorizer_type.lower() == "bm25":
                    tokenized_query = [self.tokenize_fn(query)]
                    step1_score = np.array([self.step1_model.get_scores(query) for query in tokenized_query])
                elif  self.step1_model.vectorizer_type.lower() == "tfidf":
                    step1_score = self.step1_model.get_scores(query)
                    if not isinstance(step1_score, np.ndarray):
                        step1_score = step1_score.toarray()
            elif isinstance(self.step1_model, Semantic):
                dense_qvec = self.get_step1_embedding(query=query)
                step1_score = self.step1_model.get_scores(dense_qvec).numpy()

            if isinstance(self.step2_model, Syntactic):
                if self.step2_model.vectorizer_type.lower() == "bm25":
                    tokenized_query = [self.tokenize_fn(query)]
                    step2_score = np.array([self.step2_model.get_scores(query) for query in tokenized_query])
                elif  self.step2_model.vectorizer_type.lower() == "tfidf":
                    step2_score = self.step2_model.get_scores(query)
                    if not isinstance(step2_score, np.ndarray):
                        step2_score = step2_score.toarray()
            elif isinstance(self.step2_model, Semantic):
                dense_qvec = self.get_step2_embedding(query=query)
                step2_score = self.step2_model.get_scores(dense_qvec).numpy()
            result = self.hybrid_scale(step1_score, step2_score, alpha)
        sorted_result = np.argsort(result.squeeze())[::-1]
        print(sorted_result)
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indice = sorted_result.tolist()[:k]
        return doc_score, doc_indice

    def get_relevant_doc_bulk(
        self, queries: List[str], alpha: float, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        tokenized_queries = [self.tokenize_fn(query) for query in queries]
        if isinstance(self.step1_model, Syntactic):
            step1_score = np.array([self.step1_model.get_scores(query) for query in tokenized_queries])
        elif isinstance(self.step1_model, Semantic):
            dense_qvec = self.get_step1_embedding(query=queries)
            step1_score = self.step1_model.get_scores(dense_qvec).numpy()
        if isinstance(self.step2_model, Syntactic):
            step2_score = np.array([self.step2_model.get_scores(query) for query in tokenized_queries])
        elif isinstance(self.step2_model, Semantic):
            dense_qvec = self.get_step2_embedding(query=queries)
            step2_score = self.step2_model.get_scores(dense_qvec).numpy()
        result = self.hybrid_scale(step1_score, step2_score, alpha)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


if __name__ == "__main__":
    pass