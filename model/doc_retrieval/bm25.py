from rank_bm25 import BM25Okapi
import pickle
import os
from typing import List
from underthesea import word_tokenize

class doc_retrieval:
    def __init__(
            self,
            file_path:str='path_to_data',
            save_object_path='model/doc_retrieval/save_model/bm25.pkl',
            data=None
    ):
        
        self.raw_data = self.read(file_path) if data != None else data
        if data != None:
            if (not os.path.exists(save_object_path)):
                self.bm25 = BM25Okapi(self.tokenizer(self.raw_data))
                with open(save_object_path, 'wb') as f:
                    pickle.dump(f, self.bm25)
            else:
                with open(save_object_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
        else:
            self.bm25 = BM25Okapi(self.tokenizer(self.raw_data))
        
    def ___call__(
            self,
            query:str,
            k:int=3,
            get_score=True,
    ):
        scores = None
        answer = self.bm25.get_top_n(self.tokenizer(query), self.raw_data, n=k)
        if get_score:
            scores = self.bm25.get_scores(self.tokenizer(query))
        return answer, scores
        
    def tokenizer(
            self,
            query:str,
            n_grams=(1, 3)
    ):
        pass

    @staticmethod
    def read(file_path):
        pass