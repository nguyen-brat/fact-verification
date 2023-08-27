from rank_bm25 import BM25Okapi
from typing import List

class doc_retrieval:
    def __init__(
            self,
            data:List,
    ):
        self.raw_data = data
        self.bm25 = BM25Okapi(self.tokenizer(data))
    
    def ___call__(
            self,
            query:str,
            k:int,
    ):
        answer = self.bm25.get_top_n(self.tokenizer(query), self.raw_data, n=k)
        return answer
        
    def tokenizer(
            self,
            query:str,
            n_grams=(1, 3)
    ):
        pass