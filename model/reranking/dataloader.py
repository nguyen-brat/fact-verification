from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample
import collections
import glob
import logging
import os
import random
from typing import Dict, List, Tuple
import multiprocessing
from glob import glob
from underthesea import sent_tokenize

class CrossEncoderSamples(object):
    query: List[str] = []
    positive_passages: List[str] = []
    contexts: List[str] = []

CrossEncoderBatch = List[InputExample] # [[claim, answer], [claim, answer]]

class DataloaderConfig(object):
    num_hard_negatives:int=1,
    num_other_negatives:int=7,
    shuffle:bool=True,
    shuffle_positives:bool=True,
    batch_size:int=16,

class dataloader(Dataset):
    def __init__(
            self,
            data_path,
            config:DataloaderConfig,
    ):
        self.config = config
        self.data_paths = glob('dump_data/train' + '/*/*.json')
        if config.shuffle:
            random.shuffle(self.data_path)

    def __len__(self):
        return len(self.data_path)//self.batch_size

    def __getitem__(self, idx):
        return self.create_biencoder_input(idx=idx)
    
    def create_biencoder_input(self, idx)->CrossEncoderBatch: 
        raw_batch = self.create_crossencoder_samples(idx)
    
    def create_crossencoder_samples(self, idx)->CrossEncoderSamples:
        def read_contexts(diction):
            return sent_tokenize(diction['context'])
        def read_claims(diction):
            return diction['claim']
        def read_evidients(diction):
            return diction['evident']

        id = idx*self.config.batch_size
        samples = CrossEncoderSamples
        raw_data = self.read_files(self.data_paths[id:(id+self.config.batch_size)])
        with multiprocessing.Pool() as pool:
            contexts = pool.map(read_contexts, raw_data)
            claims = pool.map(read_claims, raw_data)
            evidents = pool.map(read_evidients, raw_data)
        contexts_set = set()
        for context in contexts:
            contexts_set.update(context)
        samples.contexts = list(contexts_set)
        samples.query = claims
        samples.positive_passages = evidents
        
        return samples
        

    @staticmethod
    def read_files(paths):
        def read_file(file):
            with open(file, 'r') as f:
                data = f.read()
            return data
    
        with multiprocessing.Pool() as pool:
            results = pool.map(read_file, paths)
        return results

    @staticmethod
    def read_file(file):
        with open(file, 'r') as f:
            data = f.read()
        return data
        