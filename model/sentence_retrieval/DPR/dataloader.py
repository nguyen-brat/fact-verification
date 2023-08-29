from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "questions",
        "contexts",
        "is_positive",
    ],
)

def dataloader(Dataset):
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
            samples: List[BiEncoderSample],
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    )->BiEncoderBatch:
        pass