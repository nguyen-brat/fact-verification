from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

class Verdict(object):
    SUPPORT=0
    REFUTED=1
    NEI=2

class FactVerificationSample(object):
    claim: str
    context: str
    result:Verdict

class FactVerificationBatch(object):
    claims:List[str]
    facts:List[List[str]] 

class dataloader(Dataset):
    def __init__(
            self,
            data_path,
    ):
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