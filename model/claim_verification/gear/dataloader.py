from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM


class FactVerificationSample(object):
    claim: str
    context: str
    result:int # 0, 1, 2

class FactVerificationBatch(object): 
    claims:List[str] # [claim1, claim2, claim3]
    facts:List[List[str]] # [[evidient 1, evidient2, evidien3, evident5], [enviden]]
    label:List[int]

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