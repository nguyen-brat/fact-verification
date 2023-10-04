from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple, Union
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
    claims:List[str]
    claims_facts:List[List[str]] # list of list of fact and list of claims ([claims1, ...,claimsn], [facts1,...,factsn])
    label:torch.Tensor # 1-d tensor for label of if claim has the same len of claims
    is_positive:torch.Tensor # 1-d tensor
    is_positive_ohot:torch.Tensor # 1-d tensor
    fact_per_claim:int # 5


class FactVerifyDataloader(RerankDataloader):
    def __init__(
            self,
            data_path='data/ise-dsc01-warmup.json',
            config:RerankDataloaderConfig=RerankDataloaderConfig(4,0),
    ):
        super().__init__(data_path=data_path, config=config)

    def __getitem__(self, idx):
        return self.create_fact_verification_input(idx=idx)
    
    def create_fact_verification_input(self, idx)->FactVerificationBatch:
        raw_batch = self.create_crossencoder_samples(idx=idx)
        
        batch = FactVerificationBatch
        batch.claims = raw_batch.query
        batch.label = torch.tensor(raw_batch.labels)
        batch.fact_per_claim = self.config.num_hard_negatives + self.config.num_other_negatives + 1 # 1 is the positive fact
        batch.claims_facts = []
        batch.is_positive = []

        tokenize_batch_context = self.list_sentence_tokenize(raw_batch.contexts)
        bm25 = BM25Okapi(tokenize_batch_context)
        facts = []
        for i, query in enumerate(raw_batch.query):
            positive_id = -1 # positive_id = -1 mean there if no positive id and label is NEI
            sample = []
            if inverse_relation[raw_batch.labels[i]] != "NEI":
                sample.append(raw_batch.positive_passages[i])
                try:
                    positive_id = raw_batch.contexts.index(raw_batch.positive_passages[i])
                except:
                    positive_id = -1
            ohot_positive_id = F.one_hot(torch.tensor(0), num_classes=batch.fact_per_claim).tolist() if positive_id != -1 else [0]*batch.fact_per_claim
            all_negative_index = self.retrieval(query,
                                                bm25,
                                                positive_id,
                                                hard=self.config.num_hard_negatives,
                                                easy=0,)
            sample += np.array(raw_batch.contexts)[np.array(all_negative_index)].tolist()

            if self.config.shuffle_positives:
                temp = list(zip(sample, ohot_positive_id))
                random.shuffle(temp)
                sample, ohot_positive_id = zip(*temp)
                sample, ohot_positive_id = list(sample), list(ohot_positive_id)
            batch.is_positive.append(ohot_positive_id)
            facts.append(sample)

        if self.config.shuffle:
            temp = list(zip(raw_batch.query, raw_batch.labels, facts, batch.is_positive))
            random.shuffle(temp)
            claims, label, facts, batch.is_positive = zip(*temp)
            claims, label, facts, batch.is_positive = list(claims), list(label), list(facts), list(batch.is_positive)
            batch.is_positive_ohot = torch.tensor(batch.is_positive, dtype=torch.float32)
            batch.is_positive = torch.argmax(batch.is_positive_ohot, dim=1).type(torch.float32)
            batch.label = torch.tensor(label)
            batch.claims = claims

        concat_claims = np.array(claims*batch.fact_per_claim)
        concat_claims = concat_claims.reshape(batch.fact_per_claim, self.config.batch_size)
        concat_claims = concat_claims.transpose().flatten().tolist()
        batch.claims_facts = [concat_claims, np.array(facts).flatten().tolist()]

        return batch