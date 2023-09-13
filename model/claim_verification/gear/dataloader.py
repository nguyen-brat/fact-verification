from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
from enum import Enum
from transformers import AutoModelForCausalLM
import torch
 
class relation(Enum):
    SUPPORTED = 0
    REFUTED = 1
    NEI = 2

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
    ):
        model = AutoModelForCausalLM('bert-based-uncased')
        self.data_path = data_path

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
    
    def create_fact_verification_input(
            batch_size,
            samples: List[FactVerificationSample],
            shuffle: bool = True,
    )->List[FactVerificationBatch]:
        '''
        TODO
        '''
        pass
    
    @staticmethod
    def preprocess(path):
        '''
        take data path and return FactVerificationSample
        '''
        pass