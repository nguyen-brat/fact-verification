from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
import collections
import glob
import logging
import os
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import multiprocessing
from glob import glob
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize, word_tokenize

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
    query: List[str] = []
    positive_passages: List[str] = []
    contexts: List[str] = []
    labels: List[int]

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
        self.data_paths = glob(data_path + '/*/*.json')
        if config.shuffle:
            random.shuffle(self.data_paths)

    def __len__(self):
        return len(self.data_paths)//self.config.batch_size


    def __getitem__(self, idx):
        return self.create_biencoder_input(idx=idx)
    

    def create_biencoder_input(self, idx)->CrossEncoderBatch: 
        raw_batch = self.create_crossencoder_samples(idx)
        tokenize_batch_context = self.list_sentence_tokenize(raw_batch.contexts)
        bm25 = BM25Okapi(tokenize_batch_context)
        result = []
        for i, query in enumerate(raw_batch.query):
            positive_id = -1
            neg_sample = []
            if inverse_relation[raw_batch.labels[i]] != "NEI":
                positive_id = raw_batch.contexts.index(raw_batch.positive_passages[i])
                neg_sample.append(InputExample(texts=[query, raw_batch.positive_passages[i]], label=1))
            all_negative_index = self.retrieval(query,
                                                bm25,
                                                positive_id, 
                                                hard=self.config.num_hard_negatives, 
                                                easy=self.config.num_other_negatives,)
            

            neg_sample = list(map(lambda x, y: self.create_neg_input(x, y), [query]*all_negative_index.shape[0], np.array(raw_batch.contexts)[all_negative_index].tolist()))
            if self.config.shuffle_positives:
                random.shuffle(neg_sample)
            result += neg_sample
        if self.config.shuffle:
            random.shuffle(result)
        return result

    def create_crossencoder_samples(self, idx)->CrossEncoderSamples:
        id = idx*self.config.batch_size
        samples = CrossEncoderSamples
        raw_data = self.read_files(self.data_paths[id:(id+self.config.batch_size)])

        data = pd.DataFrame(raw_data)
        data['context'] = data['context'].map(self.split_doc)
        data['verdict'] = data['verdict'].map(lambda x: relation[x])

        contexts_set = set()
        for context in data['context'].to_list():
            contexts_set.update(context)
        samples.contexts = list(contexts_set)
        samples.query = data['claim'].to_list()
        samples.positive_passages = data['evidient'].to_list()
        samples.labels = data['verdict'].to_list()
        
        return samples
    
    @staticmethod
    def create_neg_input(query, context):
        return InputExample(texts=[query, context], label=0)
    @staticmethod
    def split_doc(graphs):
        '''
        because then use underthesea sent_token it still have . in the end of sentence so
        we have to remove it
        '''
        output = sent_tokenize(graphs)
        in_element = list(map(lambda x:x[:-1].strip(), output[:-1]))
        last_element = output[-1] if (output[-1][-1] != '.') else output[-1][-1].strip()
        return in_element + [last_element]
        

    def retrieval(self,
            query:str,
            bm25:BM25Okapi,
            positive_id:int,# id of positive sample in batch
            hard:int=5, # number of hard negative sample
            easy:int=10, # number of easy negative sample
    )->np.ndarray:
        '''
        take query and bm25 object of batch context
        return index of top hard negative sample and easy negative sample in batch
        '''
        scores = bm25.get_scores(word_tokenize(query))
        sorted_index = np.argsort(scores)
        # remove positive id in the sorted id list because this create negative id sample for training
        extra_neg_sample = 1 # it will add a extra easy negative sample if there is no positive answer
        if positive_id != -1:
            extra_neg_sample = 0
            ids_of_positive_id = np.where(sorted_index == positive_id)
            sorted_index = np.delete(sorted_index, ids_of_positive_id)
        easy_sample_index = sorted_index[20:20+easy+extra_neg_sample]
        hard_sample_index = sorted_index[:hard]
        return np.concatenate([easy_sample_index, hard_sample_index])


    @staticmethod
    def list_sentence_tokenize(inputs:List[str])->List[List[str]]:
        result = []
        for sentence in inputs:
            result.append(word_tokenize(sentence, format='text'))
        return result


    def read_files(self, paths):
        results = list(map(self.read_file, paths))
        return results

    @staticmethod
    def read_file(file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data