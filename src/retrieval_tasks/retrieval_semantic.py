import torch
import logging
import time
import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager 
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple, NoReturn
import scipy

from torch.utils.data import Dataset
from torch.nn.functional import normalize

from .retrieval import Retrieval
from .indexer import IndexRunner, indexers, get_passage_file

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similarity_score(q_vec, c_vec):
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

def get_cosine_score(q_vec, c_vec):
    q_vec = q_vec / q_vec.norm(dim=1, keepdim=True)
    c_vec = c_vec / c_vec.norm(dim=1, keepdim=True)
    return torch.mm(q_vec, c_vec.T)

class Semantic(Retrieval):
    def __init__(
        self,
        dense_model_name: str, 
        indexer_type: str = "DenseFlatIndexer",
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wiki_docs.csv", #"wiki_documents_original.csv",
        index_output_path: Optional[str] = "2050iter_flat",
        chunked_path: Optional[str] = "../../data/processed_passages",
        FAISS: bool = False,
        device = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.contexts = None
        if not FAISS:
            with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
                wiki = pd.read_csv(f)
            self.contexts = list(dict.fromkeys(wiki['content']))

        self.data_path = data_path
        self.context_path = context_path
        self.index_output_path = index_output_path
        self.chunked_path = chunked_path
        self.indexer_type = indexer_type
        self.indexer = None
        self.dense_model_name = dense_model_name
        self.tokenize_fn = AutoTokenizer.from_pretrained(
                self.dense_model_name
            )
        self.dense_embeder = AutoModel.from_pretrained(
                self.dense_model_name
            ).to(self.device)
        self.dense_embeds = None
        
        
    def get_dense_embedding(self, query=None, contexts=None):
        if contexts is not None:
            self.contexts = contexts
            self.dense_embeds = self.output(self.contexts).cpu()

        if query is not None:
            sentence_embeddings = self.output(query)
            sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()

        if query is None and contexts is None:
            model_n = self.dense_model_name.split('/')[1]
            pickle_name = f"{model_n}_dense_embedding.bin"
            emd_path = os.path.join(self.data_path, pickle_name)

            if os.path.isfile(emd_path):
                self.dense_embeds = torch.load(emd_path)
                print("Dense embedding loaded.")
            else:
                print("Building passage dense embeddings in batches.")
                self.dense_embeds = torch.zeros(len(self.contexts), self.dense_embeder.config.hidden_size)

                for i in tqdm(range(0, len(self.contexts)), desc="Encoding passages"):
                    batch_contexts = self.contexts[i:i+1]
                    encoded_input = self.tokenize_fn(
                        batch_contexts, padding=True, truncation=True, return_tensors='pt'
                    ).to(self.device)
                    with torch.no_grad():
                        model_output = self.dense_embeder(**encoded_input)
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                    self.dense_embeds[i] = sentence_embeddings.cpu()
                    del encoded_input, model_output, sentence_embeddings 
                    torch.cuda.empty_cache()  

                torch.save(self.dense_embeds, emd_path)
                print("Dense embeddings saved.")

    def get_dense_embedding_with_faiss(self, contexts=None):
        self.indexer = getattr(indexers, self.indexer_type)()
        if self.indexer.index_exists(self.index_output_path):
            self.indexer.deserialize(self.index_output_path)
        else:
            IndexRunner(
                encoder=self.dense_embeder,
                tokenizer=self.tokenize_fn,
                data_dir=os.path.join(self.data_path, self.context_path),
                indexer_type=self.indexer_type,
                index_output_path=os.path.join(self.data_path, self.index_output_path),
                chunked_path=os.path.join(self.data_path, self.chunked_path),
                indexer=self.indexer,
                use_faiss=True,
                contexts=contexts
            ).run()

    def transform(self, contexts):
        encoded_input = self.tokenize_fn(
                contexts, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
        with torch.no_grad():
            model_output = self.dense_embeder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        del encoded_input
        return sentence_embeddings

    def get_scores(self, query_embeds, contexts_embeds=None):
        if self.dense_embeds is None:
            if contexts_embeds is None:
                raise ValueError("Contexts Embeds not existed!")
            return get_similarity_score(query_embeds, contexts_embeds)
        else:
            contexts_embeds = self.dense_embeds 
            return get_similarity_score(query_embeds, contexts_embeds)

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1):
        assert self.dense_embeder is not None, "You should first execute `get_dense_embedding()`"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_contexts = self.get_relevant_doc_with_faiss(query_or_dataset, k=topk)
            logging.info(f"[Search query] {query_or_dataset}")

            for i in range(topk):
                logging.info(f"Top-{i+1} passage with score {doc_scores[i]}")
                logging.info(doc_contexts[i])
            return (doc_scores, doc_contexts)

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_contexts = self.get_relevant_doc_bulk_with_faiss(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Semantic retrieval] ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([context for context in doc_contexts[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
        
    def get_relevant_doc_with_faiss(
            self, query: str, k: Optional[int] = 1
        ) -> Tuple[List, List]:
        encoded_input = self.tokenize_fn(
                query, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
        with torch.no_grad():
            model_output = self.dense_embeder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        if self.indexer is not None:
            result = self.indexer.search_knn(query_vectors=sentence_embeddings.cpu().numpy(), top_docs=k)
            passages = []
            scores = []
            for idx, sim in zip(*result[0]): # idx, sim
                path = get_passage_file(os.path.join(self.data_path, self.chunked_path), [idx])
                if not path:
                    logging.debug(f"올바른 경로에 피클화된 위키피디아가 있는지 확인하세요.No single passage path for {idx}")
                    continue
                with open(path, "rb") as f:
                    passage_dict = pickle.load(f)
                passages.append(passage_dict[idx])
                scores.append(sim)
            return scores, passages
        else:
            raise ValueError("Indexer is None.")

    def get_relevant_doc_bulk_with_faiss(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List, List]:
        encoded_input = self.tokenize_fn(
                queries, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
        with torch.no_grad():
            model_output = self.dense_embeder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        if self.indexer is not None:
            result = self.indexer.search_knn(query_vectors=sentence_embeddings.cpu().numpy(), top_docs=k)
            doc_scores = []
            doc_indices = []
            for index in enumerate(result):
                doc_score = []
                doc_indice = []
                for idx, sim in zip(*result[index]): # idx, sim
                    path = get_passage_file(os.path.join(self.data_path, self.chunked_path), [idx])
                    if not path:
                        logging.debug(f"올바른 경로에 피클화된 위키피디아가 있는지 확인하세요. No single passage path for {idx}")
                        continue
                    with open(path, "rb") as f:
                        passage_dict = pickle.load(f)
                    doc_indice.append(passage_dict[idx])
                    doc_score.append(sim)
                doc_scores.append(doc_score)
                doc_indices.append(doc_indice)
            return doc_scores, doc_indices
        else:
            raise ValueError("Indexer is None.")

if __name__ == "__main__":
    pass
