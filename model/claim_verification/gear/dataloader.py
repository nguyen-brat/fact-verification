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
 
relation = {
    "SUPPORTED":0,
    "REFUTED":1,
    "NEI":2,
}

class FactVerificationSample(object):
    claim: str
    context: str
    label:int # 0, 1, 2

class FactVerificationBatch(object): 
    claims:List[str] # [claim1, claim2, claim3] # batch_size
    facts:List[str] # [evidient 1, evidient2, evidien3, evident5, enviden]
    label:torch.Tensor # 1-d tensor for label of if claim has the same len of claims
    fact_per_claim:int # 5

class dataloader(Dataset):
    def __init__(
            self,
            data_path,
            num_hard_negatives:int=2,
            num_other_negatives:int=2,
            tokenize:bool=True,
            shuffle:bool=True,
            shuffle_positives:bool=True,
            batch_size:int=32,
            fact_per_claim:int = 5
    ):
        self.data_paths = glob(data_path + '/*/*.json')
        self.raw_data = self.read_files(self.data_paths)
        self.raw_data = pd.DataFrame(self.raw_data)
        raw_context = self.raw_data['context'].map(self.split_doc)
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
        self.fact_per_claim = fact_per_claim
        self.fit_context()

    def __len__(self):
        return len(self.data_paths)//self.batch_size

    def __getitem__(self, idx):
        return self.create_fact_verification_input(
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
    def tokenizer(texts: List)->List:
        return [word_tokenize(text, format='text') for text in texts]

    def fit_context(self):
        self.clean_context = self.preprocess(self.raw_context)
        if self.tokenize:
            self.clean_context = self.tokenizer(self.clean_context)
        self.bm25 = BM25Okapi([text.split() for text in self.clean_context])

    @staticmethod
    def generate_random_idx(upper:int, lower:int=0):
        shuffle = list(range(lower, upper))
        random.shuffle(shuffle)
        shuffle = np.array(shuffle)
        return shuffle

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
    
    def create_fact_verification_input(self, idx)->FactVerificationBatch:
        id = idx * self.batch_size
        raw_data = self.read_files(self.data_paths[id:(id+self.batch_size)])
        raw_data = pd.DataFrame(raw_data)
        claim_list = raw_data['claim'].to_list()
        positive_context_list = raw_data['evidient'].to_list()
        raw_data['verdict'] = raw_data['verdict'].map(lambda x: relation[x])

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

        lb = torch.zeros(len(claim_list))
        for i in range(len(claim_list)):
          lb[i] = raw_data['verdict'][i]
        
        batch = FactVerificationBatch
        batch.claim = raw_data['claim'].to_list()
        batch.facts = ctx
        batch.label = lb
        batch.fact_per_claim = self.fact_per_claim

        return batch

    @staticmethod
    def split_doc(graphs):
        graphs = re.sub(r'\n+', r'. ', graphs)
        graphs = re.sub(r'\.+', r'.', graphs)
        graphs = re.sub(r'\.', r'|.', graphs)
        outputs = sent_tokenize(graphs)
        return [output.rstrip('.').replace('|', '') for output in outputs]
    
    @staticmethod
    def list_sentence_tokenize(inputs:List[str])->List[List[str]]:
        result = []
        for sentence in inputs:
            result.append(word_tokenize(sentence, format='text'))
        return result

    def read_files(self, paths):
        results = list(map(self.read_file, paths))
        return results

    @staticmethod
    def read_file(file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data
