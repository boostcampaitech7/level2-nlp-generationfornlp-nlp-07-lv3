from abc import abstractmethod
from typing import Optional, NoReturn

class Retrieval:
    @abstractmethod
    def __init__(
        self,
        tokenize_fn,
        dense_model_name,
        data_path: Optional[str],
        context_path: Optional[str],
    ) -> NoReturn:
        pass  
    
    @abstractmethod
    def retrieve(self, query_or_dataset, topk: Optional[int], alpha: Optional[float] = None, no_sparse: bool = False):
        pass