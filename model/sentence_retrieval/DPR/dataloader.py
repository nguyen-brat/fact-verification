from torch.utils.data import Dataset
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])

# {
#     'context':,
#     'claim',
#     'verdict',
#     'evidient'
# }

class BiEncoderSample(object): # hom nay toi di hoc -> hom_nay toi di_hoc n_gram
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage] # 10 cai negative
    hard_negative_passages: List[BiEncoderPassage] # top 10 cau tra loi tu bm25

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "questions", # list of question
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
            samples: List[BiEncoderSample],
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    )->BiEncoderBatch:
        pass