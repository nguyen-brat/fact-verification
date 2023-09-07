from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

class CrossEncoderSample(object):
    query: str
    positive_passages: List[str]
    negative_passages: List[str]
    hard_negative_passages: List[str]

CrossEncoderBatch = List[InputExample] # [[claim, answer], [claim, answer]]

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
    
    def create_biencoder_input(
            samples: List[CrossEncoderSample],
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    )->CrossEncoderBatch:
        pass