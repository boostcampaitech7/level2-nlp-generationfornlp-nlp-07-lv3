import torch
import logging
import os
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple, NoReturn

import retrieval_tasks.indexers
from .retrieval import Retrieval
from .index_runner import IndexRunner
from .utils import get_passage_file

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Semantic(Retrieval):
    def __init__(
        self,
        dense_model_name: str, 
        indexer_type: str = "DenseFlatIndexer",
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wiki_docs.csv", #"wiki_documents_original.csv",
        index_output: Optional[str] = "../data/2050iter_flat"
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = data_path
        self.context_path = context_path
        self.index_output = index_output
        self.indexer_type = indexer_type
        self.indexer = None
        self.dense_model_name = dense_model_name
        self.dense_tokenize_fn = AutoTokenizer.from_pretrained(
                self.dense_model_name
            )
        self.dense_embeder = AutoModel.from_pretrained(
                self.dense_model_name
            ).to(self.device)

    def get_dense_embedding_with_faiss(self, question=None, contexts=None, batch_size=64):
        self.indexer = getattr(retrieval_tasks.indexers, self.indexer_type)()
        if self.indexer.index_exists(self.index_output):
            self.indexer.deserialize(self.index_output)
        else:
            IndexRunner(
                encoder=self.dense_embeder,
                tokenizer=self.dense_tokenize_fn,
                data_dir=os.path.join(self.data_path, self.context_path),
                indexer_type=self.indexer_type,
                index_output=self.index_output,
            ).run()

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1, alpha: Optional[float] = 0, no_sparse: bool = True):
        assert self.dense_embeder is not None, "You should first execute `get_dense_embedding()`"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_with_faiss(query_or_dataset, alpha, k=topk)
            logging.info(f"[Search query] {query_or_dataset}")

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_with_faiss(
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

    def get_relevant_doc_with_faiss(
            self, query: str, alpha: float, k: Optional[int] = 1
        ) -> Tuple[List, List]:
        encoded_input = self.dense_tokenize_fn(
                query, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
        with torch.no_grad():
            model_output = self.dense_embeder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        if self.indexer is not None:
            result = self.indexer.search_knn(query_vectors=sentence_embeddings.cpu().numpy(), top_docs=k)
            passages = []
            for idx, _ in zip(*result[0]): # idx, sim
                path = get_passage_file([idx])
                if not path:
                    logging.debug(f"올바른 경로에 피클화된 위키피디아가 있는지 확인하세요.No single passage path for {idx}")
                    continue
                with open(path, "rb") as f:
                    passage_dict = pickle.load(f)
                passages.append(passage_dict[idx])
            return passages
        else:
            raise ValueError("Indexer is None.")

    def get_relevant_doc_bulk_with_faiss(
        self, queries: List[str], alpha: float, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        encoded_input = self.dense_tokenize_fn(
                query, padding=True, truncation=True, return_tensors='pt'
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
                    path = get_passage_file([idx])
                    if not path:
                        logging.debug(f"올바른 경로에 피클화된 위키피디아가 있는지 확인하세요.No single passage path for {idx}")
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

    def output(self, contexts):
        encoded_input = self.dense_tokenize_fn(
                contexts, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
        with torch.no_grad():
            model_output = self.dense_embeder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        del encoded_input
        return sentence_embeddings

    