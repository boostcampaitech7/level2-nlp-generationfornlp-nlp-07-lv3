from collections import defaultdict
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import pickle
import logging

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger()

class DataChunk:
    """인풋 text를 tokenizing한 뒤에 주어진 길이로 chunking 해서 반환합니다. 이때 하나의 chunk(context, index 단위)는 하나의 article에만 속해있어야 합니다."""

    def __init__(self, tokenizer, chunk_size=100, chunked_path: str = ""):
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.chunked_path = chunked_path

    def chunk(self, input_file_path, input_file):
        chunk_list = []
        orig_text = []
        if input_file is not None:
            input_txt = input_file
            for txt in tqdm(input_txt, desc="[chunking]", total=len(input_txt)):
                encoded_txt = self.tokenizer.encode(txt)
                if len(encoded_txt) < 5:  # 본문 길이가 subword 5개 미만인 경우 패스
                    logger.debug(f"title has <5 subwords in its article, passing")
                    continue

                # article마다 chunk_size 길이의 chunk를 만들어 list에 append. 각 chunk에는 title을 prepend합니다.
                # ref : DPR paper
                for start_idx in range(0, len(encoded_txt), self.chunk_size):
                    end_idx = min(len(encoded_txt), start_idx + self.chunk_size)
                    chunk = encoded_txt[start_idx:end_idx]
                    orig_text.append(self.tokenizer.decode(chunk))
                    chunk_list.append(chunk)
        else:
            with open(input_file_path, "r", encoding="utf8") as f:
                input_txt = pd.read_csv(input_file_path)
            for _ , art in tqdm(input_txt.iterrows(), desc="[chunking]", total=len(input_txt)):
                title = art['title']
                text = art['content']

                encoded_title = self.tokenizer.encode(title)
                encoded_txt = self.tokenizer.encode(text)
                if len(encoded_txt) < 5:  # 본문 길이가 subword 5개 미만인 경우 패스
                    logger.debug(f"title {title} has <5 subwords in its article, passing")
                    continue

                # article마다 chunk_size 길이의 chunk를 만들어 list에 append. 각 chunk에는 title을 prepend합니다.
                # ref : DPR paper
                for start_idx in range(0, len(encoded_txt), self.chunk_size):
                    end_idx = min(len(encoded_txt), start_idx + self.chunk_size)
                    chunk = encoded_title + encoded_txt[start_idx:end_idx]
                    orig_text.append(self.tokenizer.decode(chunk))
                    chunk_list.append(chunk)
        return orig_text, chunk_list

    def chunk_and_save_orig_passage(
        self, 
        input_file_path,
        input_file
    ):
        passage_path = self.chunked_path
        resource_doc_name = input_file_path.split("/")[-1].replace(".", "_")
        os.makedirs(passage_path, exist_ok=True)
        idx = 0
        orig_text, chunk_list = self.chunk(input_file_path, input_file)
        if passage_path:
            os.makedirs(passage_path, exist_ok=True)
            to_save = {idx + i: orig_text[i] for i in range(len(orig_text))}
            with open(f"{passage_path}/{resource_doc_name}-{idx}-{idx+len(orig_text)-1}.p", "wb") as f:
                pickle.dump(to_save, f)
        return orig_text, chunk_list

def save_orig_passage_bulk(
    input_path="text", passage_path="processed_passages", chunk_size=100
):
    """store original passages with unique id"""
    os.makedirs(passage_path, exist_ok=True)
    app = DataChunk(chunk_size=chunk_size)
    idx = 0
    for path in tqdm(glob(f"{input_path}/*/wiki_*")):
        ret, _ = app.chunk(path)
        to_save = {idx + i: ret[i] for i in range(len(ret))}
        with open(f"{passage_path}/{idx}-{idx+len(ret)-1}.p", "wb") as f:
            pickle.dump(to_save, f)
        idx += len(ret)

def save_title_index_map(
    index_path="title_passage_map.p", source_passage_path="processed_passages"
):
    logging.getLogger()

    files = glob(f"{source_passage_path}/*")
    title_id_map = defaultdict(list)
    for f in tqdm(files):
        with open(f, "rb") as _f:
            id_passage_map = pickle.load(_f)
        for id, passage in id_passage_map.items():
            title = passage.split("[SEP]")[0].split("[CLS]")[1].strip()
            title_id_map[title].append(id)
        logger.info(f"processed {len(id_passage_map)} passages from {f}...")
    with open(index_path, "wb") as f:
        pickle.dump(title_id_map, f)
    logger.info(f"Finished saving title_index_mapping at {index_path}!")


if __name__ == "__main__":
    pass