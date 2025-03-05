import os
import pickle
import logging
from typing import List, Optional
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

import torch
import numpy as np
from rank_bm25 import BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .retrieval import Retrieval

class Syntactic(Retrieval):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wiki_docs.csv",
        vectorizer_type: str = "bm25",  
        k1: Optional[float] = 1.5,
        b: Optional[float] = 0.75,
        delta: Optional[float] = 0.5,
        vectorizer_path: str = "../data/sparse_vectorizer.bin",
    ):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = pd.read_csv(f)

        self.contexts = list(dict.fromkeys(wiki['content']))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.tokenize_fn = tokenize_fn
        self.vectorizer_type = vectorizer_type
        self.vectorizer_path = vectorizer_path
        self.k1 = k1
        self.b = b
        self.delta = delta

        self.syntactic_embeder = None
        self.syntactic_embeds = None

    def custom_analyzer(self, doc):
        tokens = self.tokenize_fn(doc)
        if hasattr(tokens, 'tokens'):
            tokens_attr = tokens.tokens
            if callable(tokens_attr):
                tokens = tokens_attr()
            else:
                tokens = tokens_attr
        return tokens
    
    def get_sparse_embedding(self, vectorizer_path=""):
        if not vectorizer_path:
            vectorizer_path = self.vectorizer_path
        if os.path.isfile(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                self.syntactic_embeder = pickle.load(f)
            if isinstance(self.syntactic_embeder, TfidfVectorizer):
                self.syntatic_embeds = self.syntactic_embeder.transform(self.contexts)
            print("Sparse vectorizer and embeddings loaded.")
        else:
            print("Fitting sparse vectorizer and building embeddings.")
            if self.vectorizer_type.lower() == "bm25":
                tokenized_corpus = [self.tokenize_fn(doc)['input_ids'] for doc in self.contexts]
                self.syntactic_embeder = BM25Plus(tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
            elif self.vectorizer_type.lower() == "tfidf":
                self.syntactic_embeder = TfidfVectorizer(
                    analyzer=self.custom_analyzer, ngram_range=(1, 2), max_features=50000,
                )
                self.syntactic_embeds = self.syntactic_embeder.fit_transform(self.contexts)
            else:
                raise ValueError(f"Unsupported vectorizer_type: {self.vectorizer_type}")

            if vectorizer_path:
                with open(vectorizer_path, "wb") as f:
                    pickle.dump(self.syntactic_embeder, f)
                print("Sparse vectorizer and embeddings saved.")
            else:
                print("Sparse vectorizer built (not saved).")

    def transform(self, context):
        return self.syntactic_embeder.transform(context)
    
    def get_scores(self, query):
        if isinstance(self.syntactic_embeder, TfidfVectorizer):
            if isinstance(query, str):
                query = [query]
            query_vector = self.syntactic_embeder.transform(query)
            return cosine_similarity(query_vector, self.syntactic_embeds)
        elif isinstance(self.syntactic_embeder, BM25Plus):
            return self.syntactic_embeder.get_scores(query)
        else:
            assert self.syntactic_embeder is not None, "Error: self.syntactic_embeder is NoneType"
        
    def retrieve(self, query_or_dataset, topk: Optional[int] = 1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            logging.info(f"[Search query] {query_or_dataset}")

            if isinstance(doc_scores[0], list):
                doc_scores = doc_scores[0]
            if isinstance(doc_indices[0], list):
                doc_indices = doc_indices[0]

            for i in range(topk):
                logging.info(f"Top-{i+1} passage with score {doc_scores[i]}")
                logging.info(self.contexts[doc_indices[i]])
            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Sparse retrieval] ")):
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

    def get_relevant_doc(self, query: str, k: Optional[int] = 1):
        if self.vectorizer_type == 'tfidf':
            result = self.get_scores(query)
            if not isinstance(result, np.ndarray):
                result = result.toarray()

            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_score = result.squeeze()[sorted_result].tolist()[:k]
            doc_indices = sorted_result.tolist()[:k]
            return doc_score, doc_indices
        elif self.vectorizer_type == 'bm25':
            tokenized_query = [self.tokenize_fn(query)['input_ids']]
            result = np.array([self.get_scores(query) for query in tokenized_query])
            doc_scores = []
            doc_indices = []
            
            for scores in result:
                sorted_result = np.argsort(scores)[-k:][::-1]
                doc_scores.append(scores[sorted_result].tolist())
                doc_indices.append(sorted_result.tolist())
            
            return doc_scores, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1):
        if self.vectorizer_type == 'tfidf':
            result = self.get_scores(queries)
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
            return doc_scores, doc_indices
        elif self.vectorizer_type == 'bm25':
            tokenized_queries = [self.tokenize_fn(query)['input_ids'] for query in queries]
            result = np.array([self.get_scores(query) for query in tokenized_queries])
            doc_scores = []
            doc_indices = []
            
            for scores in result:
                sorted_result = np.argsort(scores)[-k:][::-1]
                doc_scores.append(scores[sorted_result].tolist())
                doc_indices.append(sorted_result.tolist())
            
            return doc_scores, doc_indices

if __name__ == "__main__":
    pass
