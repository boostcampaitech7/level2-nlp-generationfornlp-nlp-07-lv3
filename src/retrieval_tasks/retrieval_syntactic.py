import os

import pickle
import torch
from rank_bm25 import BM25Plus
from typing import List, Optional, Tuple, NoReturn

from retrieval import Retrieval

class Syntactic:
    def __init__(
        self,
        tokenize_fn,
        contexts,
        k1: Optional[float] = 1.837782128608009,
        b: Optional[float] = 0.587622663072072,
        delta: Optional[float] = 1.1490,
        vectorizer_path: str = "./sparse_vectorizer.bin",
        save_embedding: bool = False,
        syntactic_model: Optional[Retrieval] = None
    ):
    self.contexts = contexts
    self.tokenize_fn = tokenize_fn
    self.k1 = k1
    self.b = b
    self.delta = delta

    if syntaictic_model is not None:
        self.syntactic_embeder = syntactic_model
    else:
        self.syntactic_embeder = self.fit_vectorizer(vectorizer_path, save_embedding)

    def transform(self, context):
        return self.syntactic_embeder.transform(context)

    def fit_vectorizer(self, vectorizer_path, save_embedding=False):
        if os.path.isfile(vectorizer_path):
            with open(vectorizer_path, "rb") as f:
                self.sparse_embeder = pickle.load(f)
            print("Sparse vectorizer and embeddings loaded.")
        else:
            print("Fitting sparse vectorizer and building embeddings.")
            self.sparse_embeder = BM25Plus([self.tokenize_fn(doc) for doc in self.contexts], k1=self.k1, b=self.b, delta=self.delta)
            if save_embedding:
                with open(vectorizer_path, "wb") as f:
                        pickle.dump(self.sparse_embeder, f)
                print("Sparse vectorizer and embeddings saved.")
            else:
                print("Sparse vectorizer saved.")
