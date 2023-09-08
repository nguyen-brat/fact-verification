from model.sentence_retrieval.DPR.model import BiEncoder
from model.reranking.model import CrossEncoder
from underthesea import word_tokenize
from model.claim_verification.gear.model import FactVerification
from model.claim_verification.gear.dataloader import FactVerificationBatch
import torch
import numpy as np
from typing import List, Union

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Pipeline:
    def __init__(
            self,
            sentence_retrieval_path='model/sentence_retrieval/saved_model',
            reranking='model/reranking/saved_model',
            fact_check='model/claim_verification/saved_model'
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dense_retrieval = BiEncoder.from_pretrained(sentence_retrieval_path)
        self.reranking = CrossEncoder(reranking)
        self.fact_verification = FactVerification.from_pretrained(fact_check)
    
    def __call__(
            self,
            claim,
            document,
    ):
        _, sparse_score = self.sparse_retrieval(claim)
        _, dense_score = self.dense_retrieval.cal_documents_score(claim=claim)
        scores = torch.tensor(sparse_score) + dense_score
        top_relvant_doc_idx = torch.argmax(scores)
        relavent_doc = [self.raw_data[idx] for idx in top_relvant_doc_idx]
        relavent_doc = self.doc_split(relavent_doc)
        relavent_facts = self.rerank_inference(claim, relavent_doc)
        result = self.fact_verify_inference(claim, relavent_facts)
        return result


    @staticmethod
    def doc_split(documents):
        '''
        this method split a list of long document into a sentences
        '''
        pass

    def fact_verify_inference(
            self,
            query:Union[str, List[str]],
            contexts:Union[List[str], List[List[str]]],
    ):
        '''
        this method performer reranking base on reranking object
        '''
        

    def rerank_inference(
            self,
            claim:str,
            facts:List[str],
            tokenize:bool=False,
    ):
        '''
        perform fact checking inference
        '''
        if tokenize:
            claim = word_tokenize(claim, format='text')
        reranking_score = []
        for fact in facts:
            if tokenize:
                fact = word_tokenize(fact, format='text')
            pair = [claim, fact]
            result = softmax(self.reranking.predict(pair))[1]
            reranking_score.append(result)
        sort_index = np.argsort(np.array(reranking_score))
        reranking_answer = list(np.array(fact)[sort_index])
        reranking_answer.reverse()
        return reranking_answer, reranking_score
    
    def get_bm25_score(
            self,
            claim,
            document,
    ):
        pass

    def get_tfifd_score(
            self,
            claim,
            document,
    ):
        pass

if __name__ == "__main__":
    pipe = Pipeline()
    print(pipe(
        claim = 'sau một ngày thì có thể bán được một tỷ gói mè',
        document = "ông Phạm Nhật Vượng là một doanh nhân thành đạt chính vì vậy ông có thể bán được một tỷ gói mè trong 2 ngày"
    ))