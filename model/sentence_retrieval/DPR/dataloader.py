from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
import json
import torch
import re
import numpy as np
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi

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

FormatData = collections.namedtuple(
    "FormatData",
    [
        "context",
        "claim",
        "verdict",
        "evidient"
    ]
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
        self.data_path = data_path
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        self.format_data = FormatData([self.raw_data[f'{i}']['context'] for i in range(len(self.raw_data))],
                                      [self.raw_data[f'{i}']['claim'] for i in range(len(self.raw_data))],
                                      [self.raw_data[f'{i}']['verdict'] for i in range(len(self.raw_data))],
                                      [self.raw_data[f'{i}']['evidient'] for i in range(len(self.raw_data))])
        self.context = [self.raw_data[f'{i}']['context'] for i in range(len(self.raw_data))]
        raw_context = []
        for i in range(len(self.context)):
            raw_context.append(self.context[i].split(". "))
        self.raw_context = []
        for i in range(len(self.context)):
            for u in raw_context[i]:
              self.raw_context.append(u)
        self.claim = [self.raw_data[f'{i}']['claim'] for i in range(len(self.raw_data))]
        self.evidient = [self.raw_data[f'{i}']['evidient'] for i in range(len(self.raw_data))]
        for i in range(len(self.evidient)):
          if(self.evidient[i] == ''):
            self.evidient[i] = 'vào ngày 1/10/2023 ông phạm nhật vượng tuyên bố sẽ bán được một tỷ gói mè vào cuối năm 2024'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_hard_negatives = num_hard_negatives
        self.tokenize = tokenize
        self.fit_context()
        
        self.batch_claim_and_context = self.create_biencoder_input(
            batch_size = batch_size,
            num_hard_negatives = num_hard_negatives,
            num_other_negatives = num_other_negatives
        )
        

    def __len__(self):
      return self.batch_size * len(self.batch_claim_and_context)

    def __getitem__(self, idx):
      batch_number = idx//self.batch_size
      idx_in_batch = idx%self.batch_size

      batch_candidate = self.batch_claim_and_context[batch_number]
      claim = batch_candidate.claims[idx_in_batch]
      context = batch_candidate.contexts[idx_in_batch]
      pid = batch_candidate.is_positive[idx_in_batch]
      return claim, context, pid

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
        self.clean_context = self.preprocess(self.context)
        if self.tokenize:
            self.clean_context = self.tokenizer(self.clean_context)
        self.bm25 = BM25Okapi([text.split() for text in self.clean_context])

    def remove_answer_not_match_bm25_retrieval(self, question_list, positive_context_idx_list):
        clean_question_list = []
        clean_positive_text_idx_list = []
        for question, contexts_idx in zip(question_list, positive_context_idx_list):
            relevant_ctx_ids, _ = self.retrieval(question)
            relevant_answer_idx = [answer_idx for answer_idx in contexts_idx if answer_idx in relevant_ctx_ids[:100]]
            if len(relevant_answer_idx) > 0:
                clean_question_list.append(question)
                clean_positive_text_idx_list.append(relevant_answer_idx)
        return clean_question_list, clean_positive_text_idx_list

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

    def create_biencoder_input(
            self,
            batch_size: int = 32,
            num_hard_negatives: int = 10,
            num_other_negatives: int = 10
    ):
        shuffle = self.generate_random_idx(len(self.claim))
        claim_list = list(np.array(self.claim)[shuffle])
        positive_context_list = list(np.array(self.evidient)[shuffle])
        batch_claims = [claim_list[i:i+batch_size] for i in range(0, len(claim_list)//batch_size*batch_size, batch_size)]
        batch_positive_contexts = [positive_context_list[i:i+batch_size] for i in range(0, len(positive_context_list)//batch_size*batch_size, batch_size)]
        dataset = []
        for batch_claim, batch_positive_context in zip(batch_claims, batch_positive_contexts):
            clm = [claim for claim in batch_claim]

            ctx = []
            for claim, positive_context in zip(batch_claim, batch_positive_context):
                relevant_ctx_ids, _ = self.retrieval(claim)

                # add positive context
                ctx_0 = []
                ctx_0.append(positive_context)

                # add hard negatives
                hard_negatives = relevant_ctx_ids[:num_hard_negatives]
                for i in range(len(hard_negatives)):
                    ctx_0.append(self.raw_context[hard_negatives[i]])

                # add other negatives
                other = relevant_ctx_ids[num_hard_negatives:]
                shuffle_other = self.generate_random_idx(len(other))
                shuffle_other_negatives = shuffle_other[:num_other_negatives]
                for i in range(num_other_negatives):
                    ctx_0.append(self.raw_context[shuffle_other_negatives[i]])

                for u in ctx_0:
                    ctx.append(u)
            
            pid = torch.zeros(len(clm), len(ctx), dtype=torch.int64)
            pl = []
            for i in range(len(batch_claims)):
                idx = ctx.index(self.evidient[i])
            pid[i, idx] = 1
            dataset.append(BiEncoderBatch(clm, ctx, pid))
        return dataset