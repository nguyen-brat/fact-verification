from rank_bm25 import BM25Okapi
from glob import glob
import re
from underthesea import sent_tokenize, word_tokenize
from typing import List, Dict
import json
from nltk import ngrams
import numpy as np

class CleanData:
    def __init__(
            self,
            data_paths,
    ):
        self.data_path = data_paths

    
    def split_doc(self, graphs):
        graphs = re.sub(r'\n+', r'. ', graphs)
        graphs = re.sub(r'\.+', r'.', graphs)
        graphs = re.sub(r'\.', r'|.', graphs)
        outputs = sent_tokenize(graphs)
        outputs = [word_tokenize(output.rstrip('.').replace('|', ''), format='text') for output in outputs] if self.config.word_tokenize else [output.rstrip('.').replace('|', '') for output in outputs]
        return np.array(outputs)


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
        results = []
        for path in paths:
            results += self.read_file(path)
        return results


    def read_file(self, file):
        with open(file, 'r') as f:
            data = list(json.load(f).values())
            data = list(map(self.preprocess, data))
        return data
    

    def preprocess(self, item:Dict):
        for key in ['claim', 'evidence']:
            item[key] = item[key].rstrip('.') if item[key] != None else item[key]
            if self.config.word_tokenize:
                item[key] = word_tokenize(item[key], format='text')
        return item

    
    @staticmethod
    def n_gram(sentence, n=3):
        result = [*sentence.split()]
        for gram in range(2, n+1):
            ngram = ngrams(sentence.split(), gram)
            result += map(lambda x: '_'.join(x), ngram)
        return result