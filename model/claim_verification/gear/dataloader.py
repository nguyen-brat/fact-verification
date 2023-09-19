from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
import numpy as np
from enum import Enum
import torch
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize, word_tokenize
from ...reranking.dataloader import RerankDataloader, RerankDataloaderConfig, relation, inverse_relation


class CrossEncoderSamples(object):
    query: List[str] = []
    positive_passages: List[str] = []
    contexts: List[str] = []
    labels: List[int]


class FactVerificationBatch(object):
    claims:List[str] # [claim1, claim2, claim3] # batch_size
    facts:List[str] # [evidient 1, evidient2, evidien3, evident5, enviden]
    label:torch.Tensor # 1-d tensor for label of if claim has the same len of claims
    fact_per_claim:int # 5


class FactVerifyDataloader(RerankDataloader):
    def __init__(
            self,
            config:RerankDataloaderConfig,
            data_path,
    ):
        super().__init__(data_path=data_path, config=config)

    def __getitem__(self, idx):
        return self.create_fact_verification_input(idx=idx)
    
    def create_fact_verification_input(self, idx)->List[FactVerificationBatch]:
        raw_batch = self.create_crossencoder_samples(idx=idx)
        
        batch = FactVerificationBatch
        batch.claims = raw_batch.query
        batch.label = torch.tensor(raw_batch.labels)
        batch.fact_per_claim = self.config.num_hard_negatives + self.config.num_other_negatives + 1 # 1 is the positive fact
        batch.facts = []

        tokenize_batch_context = self.list_sentence_tokenize(raw_batch.contexts)
        bm25 = BM25Okapi(tokenize_batch_context)
        result = []
        for i, query in enumerate(raw_batch.query):
            positive_id = -1 # positive_id = -1 mean there if no positive id and label is NEI
            sample = []
            if inverse_relation[raw_batch.labels[i]] != "NEI":
                sample.append(raw_batch.positive_passages[i])
                positive_id = raw_batch.contexts.index(raw_batch.positive_passages[i])
            all_negative_index = self.retrieval(query,
                                                bm25,
                                                positive_id,
                                                hard=self.config.num_hard_negatives,
                                                easy=0,)
            sample += np.array(raw_batch.contexts)[np.array(all_negative_index)].tolist()
            if self.config.shuffle_positives:
                random.shuffle(sample)
            result.append(sample)
        if self.config.shuffle:
            temp = list(zip(raw_batch.query, raw_batch.labels, result))
            random.shuffle(temp)
            claims, label, result = zip(*temp)
            claims, label, result = list(claims), list(label), list(result)
            batch.label = torch.tensor(label)
            batch.claims = claims
        batch.facts = np.array(result).flatten().tolist()

        return batch