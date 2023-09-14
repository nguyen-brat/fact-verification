from torch.utils.data import Dataset
import collections
import glob
from glob import glob
import logging
import os
import random
from typing import Dict, List, Tuple
import json
import torch
import re
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize
import multiprocessing


class BiEncoderSample(object): # hom nay toi di hoc -> hom_nay toi di_hoc n_gram
    query: str
    positive_passages: List[str]
    negative_passages: List[str] # 10 cai negative
    hard_negative_passages: List[str] # top 10 cau tra loi tu bm25

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "questions", # list of question
        "contexts", # list of answer (include positive answer, negative answer, hard negative)
        "is_positive", # torch.tensor (number_question, number of context) (tensor include 0 and 1)
    ],
    # ['bao lau nua thi ban duoc mot ty goi me', 'khong dong hoc phi thi co bi duoi hoc khong']
    # ['gop 6 cau tra loi cua cau hoi']
    #   [ # torch.tensor()
    #     [0, 1, 0 , 0 , 0, 0],
    #     [0, 0, 1, 0, 0, 0],
    # ]
)

class dataloader(Dataset):
    def __init__(
            self,
            data_path,
            batch_size: int = 32,
            num_hard_negatives: int = 10,
            num_other_negatives: int = 10,
            tokenize: bool = True,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    ):
        global read_contexts
        def read_contexts(diction):
          return sent_tokenize(diction['context'])
        
        self.data_paths = glob(data_path + '/*/*.json')
        self.raw_data = self.read_files(self.data_paths)
        
        with multiprocessing.Pool() as pool:
            raw_context = pool.map(read_contexts, self.raw_data)
        self.raw_context = []
        for i in range(len(raw_context)):
          self.raw_context.extend(raw_context[i])
        self.raw_context = pd.Series(self.raw_context)

        if(shuffle):
          random.shuffle(self.data_paths)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_hard_negatives = num_hard_negatives
        self.num_other_negatives = num_other_negatives
        self.tokenize = tokenize
        self.fit_context()

    @staticmethod
    def read_files(paths):
        global read_file
        def read_file(file):
            with open(file, 'r') as f:
                data = json.load(f)
            return data
    
        with multiprocessing.Pool() as pool:
            results = pool.map(read_file, paths)
        return results
    
    def __len__(self):
      return len(self.data_paths) // self.batch_size

    def __getitem__(self, idx):
      return self.create_one_biencoder_input(
          idx = idx
      )

    @staticmethod
    def _preprocess(text:str)->str:
        text = text.lower()
        reg_pattern = '[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễếệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ\s]'
        output = re.sub(reg_pattern, '', text)
        return output

    def preprocess(self, texts:List)->List:
        return [self._preprocess(text) for text in texts]

    @staticmethod
    def generate_random_idx(upper:int, lower:int=0):
        shuffle = list(range(lower, upper))
        random.shuffle(shuffle)
        shuffle = np.array(shuffle)
        return shuffle

    @staticmethod
    def tokenizer(texts: List)->List:
        return [word_tokenize(text, format='text') for text in texts]

    def fit_context(self):
        self.clean_context = self.preprocess(self.raw_context)
        if self.tokenize:
            self.clean_context = self.tokenizer(self.clean_context)
        self.bm25 = BM25Okapi([text.split() for text in self.clean_context])

    def retrieval(self,
            text:str,
            top_k:int=10,
            return_context:bool=False,
    )->Tuple[List, List]:
        '''
        take a raw input text as query and
        return index of accesding order of
        relevant of context
        '''
        text = self._preprocess(text)
        if self.tokenize:
            text = word_tokenize(text, format='text')
        doc_scores = np.array(self.bm25.get_scores(text.split()))
        sort_idx = np.flip(np.argsort(doc_scores))
        if return_context:
            return [self.raw_context[idx] for idx in sort_idx[:top_k]]
        return sort_idx, doc_scores

    def create_one_biencoder_input(
          self,
          idx: int = 0
    ):
        global read_claims
        global read_evidients
        def read_claims(diction):
          return diction['claim']
        def read_evidients(diction):
          return diction['evidient']

        id = idx * self.batch_size
        raw_data = self.read_files(self.data_paths[id:(id+self.batch_size)])
        with multiprocessing.Pool() as pool:
            claim_list = pool.map(read_claims, raw_data)
            positive_context_list = pool.map(read_evidients, raw_data)

        ctx = []
        for claim, positive_context in zip(claim_list, positive_context_list):
              relevant_ctx_ids, _ = self.retrieval(claim)

              # add positive context
              ctx_0 = []
              ctx_0.append(positive_context)

              # add hard negatives
              hard_negatives = relevant_ctx_ids[:self.num_hard_negatives]
              ctx_0.extend(self.raw_context[hard_negatives])

              # add other negatives
              other = relevant_ctx_ids[self.num_hard_negatives:]
              shuffle_other = self.generate_random_idx(len(other))
              shuffle_other_negatives = shuffle_other[:self.num_other_negatives]
              ctx_0.extend(self.raw_context[shuffle_other_negatives])

              ctx.extend(ctx_0)

        pid = torch.zeros(len(claim_list), len(ctx), dtype=torch.int64)
        for i in range(len(claim_list)):
            idx = ctx.index(positive_context_list[i])
            pid[i, idx] = 1

        return BiEncoderBatch(claim_list, ctx, pid)
