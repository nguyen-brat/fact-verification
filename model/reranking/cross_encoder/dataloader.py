from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample
import glob
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from glob import glob

relation = {
    "SUPPORTED":0,
    "REFUTED":1,
    "NEI":2,
}

inverse_relation = {
    0:"SUPPORTED",
    1:"REFUTED",
    2:"NEI",
}

class CrossEncoderSamples(object):
    query: List[str] = [] # claim
    positive_passages: List[str] = [] # positive passage for each query
    labels: List[int] # verdict
    fact_list: List[List[str]] # list of list of candidate fact not contain evident the

CrossEncoderBatch = List[InputExample] # [[claim, answer], [claim, answer], ...]

class RerankDataloaderConfig:
    def __init__(
            self,
            num_hard_negatives:int=4,
            shuffle:bool=True,
            shuffle_positives:bool=True,
            batch_size:int=16,
            remove_duplicate_context=False,
            word_tokenize = False
    ):
        self.num_hard_negatives = num_hard_negatives
        self.shuffle = shuffle
        self.shuffle_positives = shuffle_positives
        self.batch_size = batch_size
        self.remove_duplicate_context = remove_duplicate_context
        self.word_tokenize = word_tokenize

class RerankDataloader(Dataset):
    def __init__(
            self,
            data_path='data/ise-dsc01-warmup.json',
            config:RerankDataloaderConfig=RerankDataloaderConfig(),
    ):
        self.config = config
        self.data_path = data_path
        self.raw_datas = self.read_file(data_path)
        if config.shuffle:
            random.shuffle(self.raw_datas)

    def __len__(self):
        return len(self.raw_datas)//self.config.batch_size


    def __getitem__(self, idx):
        return self.create_biencoder_input(idx=idx)


    def create_biencoder_input(self, idx)->CrossEncoderBatch:
        raw_batch = self.create_crossencoder_samples(idx)
        result = []
        for claim, evidence, facts, label in zip(raw_batch.query, raw_batch.positive_passages, raw_batch.fact_list, raw_batch.labels):
            if label != 2:
                result.append(self.create_pos_input(claim, evidence))
                for fact in facts:
                    result.append(self.create_neg_input(claim, fact))
            else:
                for fact in facts:
                    result.append(self.create_neg_input(claim, fact))
        if self.config.shuffle:
            random.shuffle(result)
        
        return result

    def create_crossencoder_samples(self, idx)->CrossEncoderSamples:
        id = idx*self.config.batch_size
        samples = CrossEncoderSamples
        raw_data = self.raw_datas[id:id+self.config.batch_size]

        data = pd.DataFrame(raw_data)
        data['verdict'] = data['verdict'].map(lambda x: relation[x])

        samples.query = data['claim'].to_list()
        samples.positive_passages = data['evidence'].to_list()
        samples.labels = data['verdict'].to_list()
        fact_list = []
        for i, label in enumerate(samples.labels):
            if label == 2:
                fact_list.append(data['facts_list'][i])
            else:
                fact_list.append([data['evidence'][i]]+data['facts_list'][i])
        samples.fact_list = fact_list

        return samples

    @staticmethod
    def create_neg_input(query, context):
        return InputExample(texts=[query, context], label=0)
    @staticmethod
    def create_pos_input(query, context):
        return InputExample(texts=[query, context], label=1)


    def read_files(self, paths):
        results = []
        for path in paths:
            results += self.read_file(path)
        return results


    def read_file(self, file):
        with open(file, 'r') as f:
            data = list(json.load(f).values())
        return data