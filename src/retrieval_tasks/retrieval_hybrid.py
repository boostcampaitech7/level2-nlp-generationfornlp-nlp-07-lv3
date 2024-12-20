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

from retrieval import Retrieval
from retrieval_syntactic import Syntactic
from retrieval_semantic import Semantic
from src.utils import set_seed

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
        dense_model_name: str = 'intfloat/multilingual-e5-large-instruct',
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wiki_documents_original.csv",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = pd.read_csv(f)

        self.contexts = list(dict.fromkeys(wiki['content']))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenize_fn = tokenize_fn

        self.dense_model_name = dense_model_name

        self.sparse_embeder = None
        self.dense_embeder = Semantic(self.dense_model_name)
        self.sparse_embeds = None
        self.dense_embeds = None

    def get_sparse_embedding(self, question=None, contexts=None):
        vectorizer_path = os.path.join(self.data_path, "sparse_vectorizer.bin")
        if contexts is not None:
            self.contexts = contexts
            self.sparse_embeder = Syntactic(self.tokenize_fn, contexts=self.contexts)

        if question is not None:
            if not hasattr(self.sparse_embeder, 'vocabulary_'):
                self.sparse_embeder = Syntactic(self.tokenize_fn, contexts=self.contexts, vectorizer_path=vectorizer_path, save_embedding=True)
            return self.sparse_embeder.transform(question)

        if question is None and contexts is None:
            self.sparse_embeder = Syntactic(self.tokenize_fn, contexts=self.contexts, vectorizer_path=vectorizer_path, save_embedding=True)

    def get_dense_embedding(self, question=None, contexts=None, batch_size=64):
        if contexts is not None:
            self.contexts = contexts
            self.dense_embeds = self.dense_embeder.output(self.contexts).cpu()

        if question is not None:
            sentence_embeddings = self.dense_embeder.output(question)
            sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()

        if question is None and contexts is None:
            model_n = self.dense_model_name.split('/')[1]
            pickle_name = f"{model_n}_dense_embedding.bin"
            emd_path = os.path.join(self.data_path, pickle_name)

            if os.path.isfile(emd_path):
                self.dense_embeds = torch.load(emd_path)
                print("Dense embedding loaded.")
            else:
                print("Building passage dense embeddings in batches.")
                self.dense_embeds = torch.zeros(len(self.contexts), self.dense_embeder.config.hidden_size)

                for i in tqdm(range(0, len(self.contexts), batch_size), desc="Encoding passages"):
                    batch_contexts = self.contexts[i:i+batch_size]
                    sentence_embeddings = self.dense_embeder.output(batch_contexts)
                    sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
                    self.dense_embeds[i] = sentence_embeddings.cpu()
                    del encoded_input, model_output, sentence_embeddings 
                    torch.cuda.empty_cache()  

                torch.save(self.dense_embeds, emd_path)
                print("Dense embeddings saved.")

    def hybrid_scale(self, dense_score, sparse_score, alpha: float):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if isinstance(dense_score, torch.Tensor):
            dense_score = dense_score.detach().numpy() 
        if isinstance(sparse_score, torch.Tensor):
            sparse_score = sparse_score.detach().numpy()  

        result = (1 - alpha) * dense_score + alpha * sparse_score
        return result

    def get_similarity_score(self, q_vec, c_vec):
        if isinstance(q_vec, scipy.sparse.spmatrix):
            q_vec = q_vec.toarray()  
        if isinstance(c_vec, scipy.sparse.spmatrix):
            c_vec = c_vec.toarray()
        
        q_vec = torch.tensor(q_vec, dtype=torch.float32)
        c_vec = torch.tensor(c_vec, dtype=torch.float32)

        if q_vec.ndim == 1:
            q_vec = q_vec.unsqueeze(0) 
        if c_vec.ndim == 1:
            c_vec = c_vec.unsqueeze(0) 

        similarity_score = torch.matmul(q_vec, c_vec.T)
        
        return similarity_score  

    def get_cosine_score(self, q_vec, c_vec):
        q_vec = q_vec / q_vec.norm(dim=1, keepdim=True)
        c_vec = c_vec / c_vec.norm(dim=1, keepdim=True)
        return torch.mm(q_vec, c_vec.T)

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1, alpha: Optional[float] = 0, no_sparse: bool = True):
        assert self.sparse_embeder is not None, "You should first execute `get_sparse_embedding()`"
        assert self.dense_embeds is not None, "You should first execute `get_dense_embedding()`"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, alpha, k=topk, no_sparse=no_sparse)
            logging.info(f"[Search query] {query_or_dataset}")

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], alpha, k=topk, no_sparse=no_sparse
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

    def get_relevant_doc(self, query: str, alpha: float, k: Optional[int] = 1, no_sparse: bool = False) -> Tuple[List, List]:
        with timer("transform"):
            dense_qvec = self.get_dense_embedding([query])

        with timer("query ex search"):
            tokenized_query = [self.tokenize_fn(query)]
            if no_sparse is False:
                sparse_score = np.array([self.sparse_embeder.get_scores(query) for query in tokenized_query])
            else:
                sparse_score = 0
            dense_score = self.get_similarity_score(dense_qvec, self.dense_embeds)
            result = self.hybrid_scale(dense_score.numpy(), sparse_score, alpha)
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List[str], alpha: float, k: Optional[int] = 1, no_sparse: bool = False
    ) -> Tuple[List, List]:
        dense_qvec = self.get_dense_embedding(queries)
        
        tokenized_queries = [self.tokenize_fn(query) for query in queries]
        if no_sparse is False:
            sparse_score = np.array([self.sparse_embeder.get_scores(query) for query in tokenized_queries])
        else:
            sparse_score = 0
        dense_score = self.get_similarity_score(dense_qvec, self.dense_embeds)
        result = self.hybrid_scale(dense_score.numpy(), sparse_score, alpha)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../../data/train.csv", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-14B-Instruct", type=str)
    parser.add_argument("--data_path", default="../../data", type=str)
    parser.add_argument("--context_path", default="rag_aug_docs_mini.csv", type=str)
    parser.add_argument("--use_faiss", default=False, type=bool)

    args = parser.parse_args()
    logging.info(args.__dict__)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    retriever = HybridSearch(
        tokenize_fn=tokenizer.tokenize,
        dense_model_name=['intfloat/multilingual-e5-large-instruct'],  #"upskyy/bge-m3-korean",
        data_path=args.data_path,
        context_path=args.context_path,
    )
    retriever.get_dense_embedding()
    retriever.get_sparse_embedding()

    query = "명량 대첩은 1597년 임진왜란 당시 이순신 장군이 12척의 배로 약 330척의 왜군 함대를 상대로 승리를 거둔 전투입니다. 이순신은 명량해협의 좁은 물길과 빠른 물살을 활용하여 왜군의 대규모 함선을 효과적으로 제압했습니다. 이는 병력 열세 속에서도 뛰어난 전략과 지휘력을 통해 이뤄낸 전승으로, 조선의 사기를 크게 북돋우고 전쟁의 전환점을 마련한 역사적 전투입니다."

    with timer("single query by exhaustive search using hybrid search"):
         scores, contexts = retriever.retrieve(query, topk=5, alpha=0)
   
    for i, context in enumerate(contexts):
        logging.info(f"Top-{i + 1} 의 문서")
        logging.info("---------------------------------------------")
        logging.info(context)

    