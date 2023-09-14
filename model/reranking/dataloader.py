from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample
import collections
import glob
import logging
import os
import random
import numpy as np
from typing import Dict, List, Tuple
import multiprocessing
from glob import glob
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize, word_tokenize

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
        tokenize_batch_context = self.list_sentence_tokenize(raw_batch.contexts)
        bm25 = BM25Okapi(tokenize_batch_context)
        result = []
        for i, query in enumerate(raw_batch.query):
            positive_id = raw_batch.contexts.index(raw_batch.positive_passages[i])
            all_negative_index = self.retrieval(query,
                                                bm25,
                                                positive_id, 
                                                hard=self.config.num_hard_negatives, 
                                                easy=self.config.num_other_negatives,)
            
            def create_neg_input(query, context):
                return InputExample(texts=[query, context], label=0)
            with multiprocessing.Pool() as pool:
                neg_sample = pool.starmap(create_neg_input, zip([query]*all_negative_index.shape[0], all_negative_index))
            neg_sample.append(InputExample(texts=[query, raw_batch.positive_passages[i]], label=1))
            if self.config.shuffle_positives:
                random.shuffle(neg_sample)
            result += neg_sample
        if self.config.shuffle:
            random.shuffle(result)
        return result
        
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
        ids_of_positive_id = np.where(sorted_index == positive_id)
        sorted_index = np.delete(sorted_index, ids_of_positive_id)
        easy_sample_index = sorted_index[20:20+easy]
        hard_sample_index = sorted_index[:hard]
        return np.concatenate([easy_sample_index, hard_sample_index])


    @staticmethod
    def list_sentence_tokenize(inputs:List[str])->List[List[str]]:
        result = []
        for sentence in inputs:
            result.append(word_tokenize(sentence, format='text'))
        return result

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
        