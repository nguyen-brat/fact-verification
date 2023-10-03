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

class BiEncoderSample(object): # hom nay toi di hoc -> hom_nay toi di_hoc n_gram
    query: str
    positive_passages: List[str]
    negative_passages: List[str] # 10 cai negative
    hard_negative_passages: List[str] # top 10 cau tra loi tu bm25

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "claims", # list of question
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
        self.data_paths = glob(data_path + '/*/*.json')
        self.raw_data = self.read_files(self.data_paths)
        self.raw_data = pd.DataFrame(self.raw_data)

        if(shuffle):
          random.shuffle(self.data_paths)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_hard_negatives = num_hard_negatives
        self.num_other_negatives = num_other_negatives
        self.tokenize = tokenize

    @staticmethod
    def split_doc(graphs):
        graphs = re.sub(r'\n+', r'. ', graphs)
        graphs = re.sub(r'\.+', r'.', graphs)
        graphs = re.sub(r'\.', r'|.', graphs)
        outputs = sent_tokenize(graphs)
        return [output.rstrip('.').replace('|', '') for output in outputs]

    def read_files(self, paths):
        results = list(map(self.read_file, paths))
        return results

    @staticmethod
    def read_file(file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

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

    def create_one_biencoder_input(
          self,
          idx: int = 0
    ):
        id = idx * self.batch_size
        raw_data = self.read_files(self.data_paths[id:(id+self.batch_size)])
        raw_data = pd.DataFrame(raw_data)
        claim_list = raw_data['claim'].to_list()
        positive_context_list = raw_data['evidient'].to_list()
        context_list = raw_data['context'].map(self.split_doc)
        raw_context = []
        for i in range(len(context_list)):
          raw_context.extend(context_list[i])
        raw_context = pd.Series(raw_context)
        bm25 = BM25Okapi([text.split() for text in raw_context])

        ctx = []
        for claim, positive_context in zip(claim_list, positive_context_list):
              doc_scores = np.array(bm25.get_scores(claim.split()))
              relevant_ctx_ids = np.flip(np.argsort(doc_scores))

              # add positive context
              ctx_0 = []
              ctx_0.append(positive_context)

              # add hard negatives
              hard_negatives = relevant_ctx_ids[:self.num_hard_negatives]
              ctx_0.extend(raw_context[hard_negatives])

              # add other negatives
              other = relevant_ctx_ids[self.num_hard_negatives:]
              shuffle_other = self.generate_random_idx(len(other))
              shuffle_other_negatives = shuffle_other[:self.num_other_negatives]
              ctx_0.extend(raw_context[shuffle_other_negatives])

              ctx.extend(ctx_0)

        pid = torch.zeros(len(claim_list), len(ctx), dtype=torch.int64)
        for i in range(len(claim_list)):
            idx = ctx.index(positive_context_list[i])
            pid[i, idx] = 1

        return BiEncoderBatch(claim_list, ctx, pid)
