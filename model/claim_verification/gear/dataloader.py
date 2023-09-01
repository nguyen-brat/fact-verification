from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

FactVerificationResultType = {
    "SUPPORT":0,
    "REFUSE":1,
    "NEI":2 # not enough information
}

class FactVerificationSample(object):
    claim: str
    facts: List[str]
    result:FactVerificationResultType 

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
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    )->List[FactVerificationSample]:
        '''
        TODO
        '''
        pass