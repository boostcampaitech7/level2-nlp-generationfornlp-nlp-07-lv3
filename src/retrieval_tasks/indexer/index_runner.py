import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
import transformers
from typing import List, Optional, Tuple, NoReturn

transformers.logging.set_verbosity_error()  # 토크나이저 초기화 관련 warning suppress
from tqdm import tqdm
import os
import sys
import logging
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import indexer.indexers as indexers
from .indexers import DenseIndexer
from .chunk_data import DataChunk
from .utils import get_wiki_filepath, wiki_worker_init

# logger basic config
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger()


def wiki_collator(batch: List, padding_value: int) -> Tuple[torch.Tensor]:
    batch_p = pad_sequence(
        [torch.tensor(e) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_p, batch_p_attn_mask)

class WikiCollator:
    def __init__(self, pad_value: int):
        self.pad_value = pad_value

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        return wiki_collator(batch, self.pad_value)


class WikiArticleStream(torch.utils.data.IterableDataset):
    def __init__(self, wiki_path, wiki_contexts, chunker):
        # self.chunk_size = chunk_size
        super(WikiArticleStream, self).__init__()
        self.chunker = chunker
        self.pad_token_id = self.chunker.tokenizer.get_vocab()["<pad>"]
        self.wiki_path = wiki_path
        self.wiki_contexts = wiki_contexts
        self.max_length = 168 

    def __iter__(self):
        _, passages = self.chunker.chunk_and_save_orig_passage(input_file_path=self.wiki_path, input_file=self.wiki_contexts)
        logger.debug(f"chunked file path {self.wiki_path}")
        for passage in passages:
            yield passage


class IndexRunner:
    def __init__(
        self,
        encoder,
        tokenizer,
        data_dir: str,
        indexer_type: str = "DenseFlatIndexer",
        chunk_size: int = 100,
        batch_size: int = 64,
        buffer_size: int = 50000,
        index_output_path: str = "",
        chunked_path: str = "",
        device: str = "cuda",
        indexer: Optional[DenseIndexer] = None,
        use_faiss = False,
        index_name: str = 'documents',
        contexts = None
    ):
        """
        data_dir : 인덱싱할 한국어 wiki 데이터가 들어있는 디렉토리입니다.
        indexer_type : 사용할 FAISS indexer 종류로 DPR 리포에 있는 대로 Flat, HNSWFlat, HNSWSQ 세 종류
        chunk_size : indexing과 searching의 단위가 되는 passage의 길이입니다. DPR 논문에서는 100개 token 길이 + title로 하나의 passage를 정의하였습니다.
        """
        if "=" in data_dir:
            self.data_dir, self.to_this_page = data_dir.split("=")
            self.to_this_page = int(self.to_this_page)
            self.wiki_files = [self.data_dir] # get_wiki_filepath(self.data_dir)
        else:
            self.data_dir = data_dir
            self.wiki_files = [self.data_dir] # get_wiki_filepath(self.data_dir)
            self.to_this_page = len(self.wiki_files)

        self.contexts = contexts
        self.device = torch.device(device)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.encoder_emb_sz = self.encoder.pooler.dense.out_features # get cls token dim
        self.indexer = getattr(indexers, indexer_type)() if indexer is None else indexer
        self.chunked_path = chunked_path
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.use_faiss = use_faiss
        if self.use_faiss:
            self.loader = self.get_loader_for_Faiss(
                self.tokenizer,
                #self.wiki_files[: self.to_this_page],
                self.wiki_files,
                self.contexts,
                chunk_size,
                batch_size,
                worker_init_fn=None,
                chunked_path=self.chunked_path
            )
        
        self.indexer.init_index(self.encoder_emb_sz)
        self.index_output_path = index_output_path if index_output_path else indexer_type

    @staticmethod
    def get_loader_for_Faiss(tokenizer, wiki_files, contexts, chunk_size, batch_size, worker_init_fn=None, chunked_path: str = ""):
        chunker = DataChunk(chunk_size=chunk_size, tokenizer=tokenizer, chunked_path=chunked_path)
        ds = torch.utils.data.ChainDataset(
            tuple(WikiArticleStream(path, contexts, chunker) for path in wiki_files)
        )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=WikiCollator(pad_value=tokenizer.get_vocab().get("<pad>", 0)),
            num_workers=1,
            worker_init_fn=worker_init_fn,
        ) 
        return loader
    
    def run(self):
        _to_index = []
        cur = 0

        if self.use_faiss:
            for batch in tqdm(self.loader, desc="indexing"):
                p, p_mask = batch
                p, p_mask = p.to(self.device), p_mask.to(self.device)
                with torch.no_grad():
                    p_emb = self.encoder(p, p_mask)
                try:
                    _to_index += [(cur + i, _emb) for i, _emb in enumerate(p_emb.cpu().numpy())]
                    cur += p_emb.size(0)
                except Exception as e:
                    logger.info("p_emb Object doesn't have .cpu()!")
                    _to_index += [(cur + i, _emb) for i, _emb in enumerate(p_emb.pooler_output.cpu().numpy())]
                    cur += p_emb.pooler_output.size(0)
                # if len(_to_index) > self.buffer_size - self.batch_size:
                logger.info(f"perform indexing... {len(_to_index)} added")
                self.indexer.index_data(_to_index)
                _to_index = []
            if _to_index:
                logger.info(f"perform indexing... {len(_to_index)} added")
                self.indexer.index_data(_to_index)
                _to_index = []
            os.makedirs(self.index_output_path, exist_ok=True)
            self.indexer.serialize(self.index_output_path) 
             
                        


if __name__ == "__main__":
    pass