from model.sentence_retrieval.DPR.model import BiEncoder
from model.reranking.model import CrossEncoder
from underthesea import word_tokenize, sent_tokenize
from model.claim_verification.gear.model import FactVerification
from model.claim_verification.gear.dataloader import FactVerificationBatch
import torch
import numpy as np
from typing import List, Union, Tuple
from rank_bm25 import BM25Okapi

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
            claim:str,
            document:str,
            alpha:int=0.7, # parameter to add score of sparse retrieval and dense retrieval
            k:int=10, # top k return for rerank
            r:int=5, # top r result return from reranking
    )->torch.Tensor:
        '''
        Pipeline to check the claim
        return tensor of probability claim relation and the most relavent sentence.
        '''
        sentences = self.doc_split(document=document)
        doc_len = len(sentences)
        k = k if doc_len > k else doc_len
        r = r if doc_len > k else doc_len
        sparse_score, _ = self.get_bm25_score(claim, document=document)
        sparse_score = torch.tensor(sparse_score)

        with torch.no_grad():
            dense_score =self.dense_retrieval.predict(claim, sentences)
            score = sparse_score*alpha + dense_score*(1-alpha)
            sorted_idx_relavent_doc = torch.argsort(score)[:k]
            relavent_doc = [sentences[idx] for idx in sorted_idx_relavent_doc]
            facts = self.rerank_inference(claim=claim, facts=relavent_doc)
            result = self.fact_verify_inference(claim=claim, facts=facts)
        return result, facts[0]


    @staticmethod
    def doc_split(document:str)->List[str]:
        '''
        this method split a list of long document into a sentences
        '''
        return sent_tokenize(document)

    def fact_verify_inference(
            self,
            claim:Union[str, List[str]], # if claim is a string context must be a list of string
            facts:List[str],
            fact_per_claim:int=None
    )->torch.Tensor:
        '''
        this method performer reranking base on reranking object
        output is a torch tensor decribe probability of each logit relation (support, refuted, nei)
        '''
        inp = FactVerificationBatch
        if isinstance(claim, str):
            inp.claims = [claim]
        else:
            inp.claims = claim
        inp.facts = facts
        inp.label = None
        inp.fact_per_claim = fact_per_claim

        logit = self.fact_verification(inp)
        return torch.softmax(logit)
        

    def rerank_inference(
            self,
            claim:str,
            facts:List[str],
            tokenize:bool=False,
    )->Tuple(List[int], List[str]):
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
        return reranking_score, reranking_answer
    
    @staticmethod
    def get_bm25_score(
            self,
            claim:str,
            document:List[str], # list of sentence in raw document
            k:int=5,
    )->Tuple(np.ndarray, List[str]):
        '''
        get bm25 score and return k top relavant sentence to claim
        '''
        tokenized_corpus = []
        for sentence in document:
            tokenized_corpus.append(word_tokenize(sentence, format='text'))
        bm25 = BM25Okapi(tokenized_corpus)
        claim_tokenize = word_tokenize(claim, format='text')
        doc_scores = bm25.get_scores(claim_tokenize)
        relavent_doc = bm25.get_top_n(claim_tokenize, document, n=k)
        return doc_scores, relavent_doc

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