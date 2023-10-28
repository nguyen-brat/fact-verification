from sentence_transformers import CrossEncoder
from model.claim_verification.joint_cross_encoder.model import JointCrossEncoder
from data_preprocess.clean_data.preprocess import CleanData
from transformers import AutoTokenizer
import torch
import numpy as np
from typing import List, Union, Tuple
import json
from tqdm import tqdm
from underthesea import word_tokenize
import argparse
import os
import random


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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Pipeline(CleanData):
    def __init__(
            self,
            #reranking='nguyen-brat/rerank_crossencoder',
            fact_check='nguyen-brat/fact_verify_v3',
            device=None,
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if device == None else device
        #self.reranking_model = CrossEncoder(reranking, num_labels=2, max_length=256)
        self.fact_verification_model = JointCrossEncoder.from_pretrained(fact_check, token="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").to(self.device)
        self.fact_verification_tokenizer = AutoTokenizer.from_pretrained(fact_check, token="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")
    
    def __call__(
            self,
            claim:str,
            document:str,
            word_segmentation=False,
            mode='multiple',
    ):
        '''
        Pipeline to check the claim
        return the verdict and most relevant sentence
        '''
        return self.predict(claim, document, word_segmentation, mode=mode)

    def predict(
            self,
            claim:str,
            document:str,
            word_segmentation=False,
            mode='multiple',
    ):
        '''
        take one sample return the verdict and most relevant sentence
        '''
        fact_list, _ = self.bm25(claim=claim, document=document, k=5)
        claim = self.preprocess_text(claim)
        fact_list = [self.preprocess_text(fact) for fact in fact_list]
        evidence = fact_list[0]
        random.shuffle(fact_list)
        #evident, _, _ = self.reranking_inference(claim=claim, fact_list=fact_list)
        if word_segmentation:
            claim = word_tokenize(claim, format='text')
            fact_list = [word_tokenize(fact, format='text') for fact in fact_list]
        verdict = self.fact_verification_inference(claim=claim, fact_list=fact_list, mode=mode)
        return evidence, verdict


    def reranking_inference(self, claim:str, fact_list:List[str]):
        '''
        take claim and list of fact list
        return reranking fact list and score of them
        '''
        claim = self.preprocess_text(claim)
        reranking_score = []
        for fact in fact_list:
            pair = [claim, fact]
            with torch.no_grad():
                result = softmax(self.reranking_model.predict(pair))[1]
            reranking_score.append(result)
        sort_index = np.argsort(np.array(reranking_score))
        reranking_answer = list(np.array(fact_list)[sort_index])
        reranking_answer.reverse()
        return reranking_answer[0], reranking_answer, reranking_score
    
    def fact_verification_inference(self, claim, fact_list, mode='multiple'):
        if mode=='multiple':
            fact_input_id = self.fact_verification_tokenizer([claim]*len(fact_list), fact_list, return_tensors='pt', max_length=256, padding='max_length', pad_to_max_length=True, truncation=True).to(self.device)
            proba = self.fact_verification_model.predict(fact_input_id)
            output = torch.argmax(proba)
            return inverse_relation[output.item()]
        elif mode=='single':
            claim = self.preprocess_text(claim)
            fact_input_id = self.fact_verification_tokenizer([claim, fact_list[0]], max_length=256, padding=True, truncation=True).to(self.device)
            proba = self.fact_verification_model.predict_single(fact_input_id)
            output = torch.argmax(proba)
            return inverse_relation[output.item()]


    def output_file(
            self,
            input_path='data/test/ise-dsc01-public-test-offcial.json',
            output_path='log/output',
            word_segmentation=False,
            mode='multiple',
    ):
        '''
        input file path need to predict
        create the result file
        '''
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        result = {}
        with open(input_path, 'r') as f:
            data = json.load(f)
        for key in tqdm(data.keys()):
            evident, verdict = self.predict(data[key]['claim'], data[key]['context'], word_segmentation=word_segmentation, mode=mode)
            result[key] = {
                "verdict":verdict,
                "evidence":evident if verdict != "NEI" else ""
            }
        with open(os.path.join(output_path, 'public_result.json'), 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            
def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for pipeline inference")
    parser.add_argument("--fact_verify_model_name", default='nguyen-brat/fact_verify_v3', type=str, help='fact checking model path or huggingface path')
    parser.add_argument("--mode", default='multiple', type=str, help='only choose single or multiple')
    parser.add_argument("--rerank_model_name", default=None, type=str)
    parser.add_argument("--word_tokenize", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--input_path", default='data/test/ise-dsc01-public-test-offcial.json', type=str)
    parser.add_argument("--output_path", default='log/output', type=str)
    args = parser.parse_args()
    return args

def pipeline_run():
    args = parse_args()
    pipe = Pipeline(fact_check=args.fact_verify_model_name)
    pipe.output_file(args.input_path, args.output_path, args.word_tokenize, args.mode)

if __name__ == "__main__":
    args = parse_args()
    pipe = Pipeline(fact_check=args.fact_verify_model_name)
    pipe.output_file(args.input_path, args.output_path, args.word_tokenize, args.mode)
    