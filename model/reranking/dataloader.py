from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
import glob
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from glob import glob
from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize, word_tokenize
from nltk import ngrams

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

class RerankDataloaderConfig:
    def __init__(
            self,
            num_hard_negatives:int=1,
            num_other_negatives:int=7,
            shuffle:bool=True,
            shuffle_positives:bool=True,
            batch_size:int=16,
            remove_duplicate_context=False,
            word_tokenize = False
    ):
        self.num_hard_negatives = num_hard_negatives
        self.num_other_negatives = num_other_negatives
        self.shuffle = shuffle
        self.shuffle_positives = shuffle_positives
        self.batch_size = batch_size
        self.remove_duplicate_context = remove_duplicate_context
        self.word_tokenize = word_tokenize

class RerankDataloader(Dataset):
    def __init__(
            self,
            data_path,
            config:RerankDataloaderConfig,
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
            positive_id = -1 # positive_id = -1 mean there if no positive id and label is NEI
            sample = []
            if inverse_relation[raw_batch.labels[i]] != "NEI":
                positive_id = raw_batch.contexts.index(raw_batch.positive_passages[i])
                sample.append(InputExample(texts=[query, raw_batch.positive_passages[i]], label=1))
            all_negative_index = self.retrieval(query,
                                                bm25,
                                                positive_id,
                                                hard=self.config.num_hard_negatives,
                                                easy=self.config.num_other_negatives,)

            #print(f'num negative sample {len(all_negative_index)}')
            sample += list(map(lambda x, y: self.create_neg_input(x, y), [query]*all_negative_index.shape[0], np.array(raw_batch.contexts)[all_negative_index].tolist()))
            #print(f'one sample len is: {len(sample)}')
            if self.config.shuffle_positives:
                random.shuffle(sample)
            result += sample
        if self.config.shuffle:
            random.shuffle(result)
        return result

    def create_crossencoder_samples(self, idx)->CrossEncoderSamples:
        id = idx*self.config.batch_size
        samples = CrossEncoderSamples
        raw_data = self.read_files(self.data_paths[id:id+self.config.batch_size])

        data = pd.DataFrame(raw_data)
        #print(data['context'])
        data['context'] = data['context'].map(self.split_doc)
        data['verdict'] = data['verdict'].map(lambda x: relation[x])

        if self.config.remove_duplicate_context:
            contexts_set = set()
            for context in data['context'].to_list():
                contexts_set.update(context)
            contexts_set = list(contexts_set)
        else:
            contexts_set = np.array(data['context'].to_list()).flatten().tolist()

        samples.contexts = contexts_set
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
            easy_sample_pivot:int=20,
    )->np.ndarray:
        '''
        take query and bm25 object of batch context
        return index of top hard negative sample and easy negative sample in batch
        '''
        if not self.config.word_tokenize:
            query = word_tokenize(query, format='text')
        scores = bm25.get_scores(self.n_gram(query))
        sorted_index = np.argsort(scores)

        extra_neg_sample = 1 # it will add a extra easy negative sample if there is no positive answer
        len_context = len(scores)
        easy_sample_pivot = easy_sample_pivot if (len_context - easy_sample_pivot) > (easy + extra_neg_sample) else (len_context - easy - extra_neg_sample)
        # remove positive id in the sorted id list because this create negative id sample for training
        if positive_id != -1:
            extra_neg_sample = 0
            ids_of_positive_id = np.where(sorted_index == positive_id)
            sorted_index = np.delete(sorted_index, ids_of_positive_id)
        easy_sample_index = sorted_index[easy_sample_pivot:easy_sample_pivot+easy+extra_neg_sample]
        hard_sample_index = sorted_index[:hard]
        return np.concatenate([easy_sample_index, hard_sample_index])


    def list_sentence_tokenize(self, inputs:List[str])->List[List[str]]:
        '''
        tokenize list of sentence for feeding to bm25
        '''
        result = []
        for sentence in inputs:
            if not self.config.word_tokenize:
                sentence = word_tokenize(sentence=sentence, format='text')
            result.append(self.n_gram(sentence))
        return result


    def read_files(self, paths):
        results = list(map(self.read_file, paths))
        return results


    def read_file(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
            if self.config.word_tokenize:
                for key in ['context', 'claim', 'evidient']:
                    data[key] = word_tokenize(data[key], format='text')
        return data
    
    @staticmethod
    def n_gram(sentence, n=3):
        result = [*sentence.split()]
        for gram in range(2, n+1):
            ngram = ngrams(sentence.split(), gram)
            result += map(lambda x: '_'.join(x), ngram)
        return result