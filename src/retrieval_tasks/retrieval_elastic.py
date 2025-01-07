import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple, NoReturn

from retrieval import Retrieval
import indexers
from index_runner import IndexRunner
from utils import get_passage_file

class Elastic(Retrieval):
    def __init__(self):

    def retrieve(self):