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
            data_path='ise-dsc01-train.json',
            tokenize=False,
    ):
        self.data_path = data_path
        self.tokenize = tokenize
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)

    
    def __call__(self, k=5, output_path=r'data/clean_data/train.json'):
        result = {}
        for key in self.raw_data.keys():
            clean_sample = self.clean(self.raw_data[key])
            if clean_sample != {}:
                result[key] = clean_sample
        with open(output_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


    def clean(self, sample, k=5):
        '''
        if the evident not in top 5 result return from the bm25
        we will remove that sample from training data
        every sample have evident not in top k will be removed
        '''
        result = {}
        fact_list, _ = self.bm25(sample['claim'], sample['context'], k=k)
        if sample['evidence']:
            evident = self.preprocess_text(sample['evidence'])
            if evident in fact_list:
                result["context"] = sample['context']
                result["claim"] = self.preprocess_text(sample['claim'])
                result["verdict"] = sample["verdict"]
                result["evidence"] = evident
                result["domain"] = sample["domain"]
                result["facts_list"] = fact_list
        else:
            result["verdict"] = sample["verdict"] # equal null
            result["claim"] = self.preprocess_text(sample['claim'])
            result["context"] = sample['context']
            result["domain"] = sample["domain"]
            result["facts_list"] = fact_list
            result["evidence"] = sample["evidence"]
        return result
    
    
    def split_doc(self, graphs):
        '''
        graphs not need to be cleaned
        '''
        graphs = re.sub(r'\.{3}\,', r' ', graphs)
        for match in re.finditer(r"(\d\.\d|)(\w\.\w)", graphs):
            graphs = graphs[:match.span()[0]+1] + '|' + graphs[match.span()[1]-1:]
        outputs = graphs.split('.')
        return [self.preprocess_text(output.replace('|', '.')) for output in outputs if output != '']

    
    def bm25(self, claim:str, document:str, k=5):
        '''
        take claim and a str document return the most relevant sentence
        claim not need to be clean
        return the k most relevant sentence and the score of top k sentence
        '''
        claim = self.preprocess_text(claim)
        sentences = self.split_doc(document)
        bm25 = BM25Okapi([self.n_gram(sentence) for sentence in sentences])
        doc_scores = np.array(bm25.get_scores(self.n_gram(claim)))
        sort_idx = np.flip(np.argsort(doc_scores))
        fact_list = np.array(sentences)[sort_idx].tolist()[:k]
        return fact_list, doc_scores[:k]
    

    def tfidf(
            self,
            claim,
            document,
    ):
        pass


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
    
    
    @staticmethod
    def preprocess_text(text: str) -> str:    
        text = re.sub(r"['\",\?:\-!]", "", text)
        text = text.strip()
        text = " ".join(text.split())
        text = text.lower()
        return text