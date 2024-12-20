import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple, NoReturn

from retrieval import Retrieval

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Semantic:
    def __init__(
        self,
        dense_model_name: str
    ):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.dense_model_name = dense_model_name
    self.dense_tokenize_fn = AutoTokenizer.from_pretrained(
            self.dense_model_name
        )
    self.dense_embeder = AutoModel.from_pretrained(
            self.dense_model_name
        ).to(self.device)

    def output(self, contexts):
        encoded_input = self.dense_tokenize_fn(
                contexts, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
        with torch.no_grad():
            model_output = self.dense_embeder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        del encoded_input
        return sentence_embeddings

    