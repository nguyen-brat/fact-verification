from torch.utils.data import Dataset
import torch.nn.functional as F
import sys
import glob
import os
import random
from typing import Dict, List, Tuple, Union
from underthesea import word_tokenize
import numpy as np
from enum import Enum
import torch
from ...reranking.cross_encoder.dataloader import RerankDataloader, RerankDataloaderConfig, relation, inverse_relation


class CrossEncoderSamples(object):
    query: List[str] = [] # claim
    positive_passages: List[str] = [] # positive passage for each query
    labels: List[int] # verdict
    fact_list: List[List[str]] # list of list of candidate fact not contain evident the


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
        batch.fact_per_claim = self.config.num_hard_negatives + 1 # 1 is the positive fact
        batch.claims_facts = []
        batch.is_positive_ohot = []

        facts = []
        for i, claim in enumerate(raw_batch.query):
            if inverse_relation[raw_batch.labels[i]] != "NEI":
                sample = [raw_batch.positive_passages[i]] + raw_batch.fact_list[i]*batch.fact_per_claim # duplicate the fact here in case fact per claim not long enough
                sample = sample[:batch.fact_per_claim]
                ohot_positive_id = F.one_hot(torch.tensor(0), num_classes=batch.fact_per_claim).to(torch.float32).tolist()
            else:
                ohot_positive_id = torch.zeros(size=(batch.fact_per_claim, )).tolist()
                sample = raw_batch.fact_list[i]*batch.fact_per_claim # duplicate the fact here in case fact per claim not long enough
                sample = sample[:batch.fact_per_claim]

            temp = list(zip(sample, ohot_positive_id))
            random.shuffle(temp)
            sample, ohot_positive_id = zip(*temp)
            sample, ohot_positive_id = list(sample), list(ohot_positive_id)
            batch.is_positive_ohot.append(ohot_positive_id)
            facts.append(sample)

        temp = list(zip(raw_batch.query, raw_batch.labels, facts, batch.is_positive_ohot))
        random.shuffle(temp)
        claims, label, facts, batch.is_positive_ohot = zip(*temp)
        claims, label, facts, batch.is_positive_ohot = list(claims), list(label), list(facts), list(batch.is_positive_ohot)
        try:
            batch.is_positive_ohot = torch.tensor(batch.is_positive_ohot, dtype=torch.float32)
        except:
            print(claims)
            print(batch.is_positive_ohot)
        batch.is_positive = torch.argmax(batch.is_positive_ohot, dim=1).type(torch.float32)
        batch.label = torch.tensor(label)
        batch.claims = claims

        concat_claims = np.array(claims*batch.fact_per_claim)
        concat_claims = concat_claims.reshape(batch.fact_per_claim, self.config.batch_size)
        concat_claims = concat_claims.transpose().flatten().tolist()
        batch.claims_facts = [concat_claims, np.array(facts).flatten().tolist()]

        return batch