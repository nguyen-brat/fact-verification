import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from underthesea import sent_tokenize
from rank_bm25 import BM25Okapi
from nltk import ngrams
import re
import torch
import torch.nn.functional as F
from typing import List
from sentence_transformers import CrossEncoder

class Visualization:
    def __init__(self,
                data_path='data/raw_data/ise-dsc01-train.json',
                use_rerank=False):
        self.raw_data = pd.read_json(data_path).transpose().sort_index().reset_index()
        if use_rerank:
            self.model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco')


    # def split_doc(self, graphs):
    #     graphs = re.sub(r'\n+', r' ', graphs)
    #     graphs = re.sub(r'\.+', r'.', graphs)
    #     graphs = re.sub(r'\.', r'|.', graphs)
    #     outputs = sent_tokenize(graphs)
    #     #outputs = graphs.split('.')
    #     return [self.preprocess_text(output.rstrip('.').replace('|', '')) for output in outputs]

    def split_doc(self, graphs):
        #graphs = re.sub(r'(\.{3}\,|\.{3}\s[a-z])', r' ', graphs)
        graphs = re.sub(r'\.{3}\,', r' ', graphs)
        for match in re.finditer(r"(\d\.\d|)(\w\.\w)", graphs):
            graphs = graphs[:match.span()[0]+1] + '|' + graphs[match.span()[1]-1:]
        outputs = graphs.split('.')
        return [self.preprocess_text(output.replace('|', '.')) for output in outputs if output != '']


    def num_of_sentences(self):
        context = list(self.raw_data['context'])
        nos = [len(sent_tokenize(x)) for x in context]
        plt.hist(nos, bins=list(range(max(nos) + 1)))
        plt.show()
        print("Số câu trung bình của các đoạn context:", np.mean(nos))


    def num_of_words_claims(self):
        claim = list(self.raw_data['claim'])
        claim = [len(x.split()) for x in claim]
        plt.hist(claim, bins=list(range(max(claim) + 1)))
        plt.show()
        print("Số từ trung bình trong 1 claim:", np.mean(claim))


    def num_of_words_evidient(self):
        evidient = list(self.raw_data['evidence'])
        evidient = [len(x.split()) for x in evidient if x != None]
        plt.hist(evidient, bins=list(range(max(evidient) + 1)))
        plt.show()
        print("Số từ trung bình trong 1 evidient:", np.mean(evidient))


    def label(self):
        return self.raw_data['verdict'].value_counts(0)


    def bm25_result(self):
        num_match_index = {}
        error_extractor = []
        for i in range(len(self.raw_data['claim'])):
            raw_context = self.split_doc(self.raw_data['context'][i])
            bm25 = BM25Okapi([self.n_gram(txt) for txt in raw_context])
            doc_scores = np.array(bm25.get_scores(self.n_gram(self.raw_data['claim'][i])))
            sort_idx = np.flip(np.argsort(doc_scores))
            #fact_list = [raw_context[idx] for idx in sort_idx[:top_k]]
            fact_list = np.array(raw_context)[sort_idx].tolist()
            evident = self.raw_data['evidence'][i]
            if evident != None:
                #evident = evident.rstrip('.')
                evident = self.preprocess_text(evident)
                try:
                    index = fact_list.index(evident)
                    if index not in num_match_index.keys():
                        num_match_index[index] = {
                            'num_match':1,
                            'sample':[{'claim':self.raw_data['claim'][i], 'evident':fact_list[:index+1]}],
                        }
                    else:
                        num_match_index[index]['num_match'] += 1
                        num_match_index[index]['sample'].append({'claim':self.raw_data['claim'][i], 'evident':fact_list[:index+1]})
                except:
                    error_extractor.append({'claim':self.raw_data['claim'][i], 'evident':evident, 'facts':fact_list[:5]})
        return num_match_index, error_extractor
    

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
                result = F.softmax(self.reranking_model.predict(pair))[1]
            reranking_score.append(result)
        sort_index = np.argsort(np.array(reranking_score))
        reranking_answer = list(np.array(fact_list)[sort_index])
        reranking_answer.reverse()
        return reranking_answer[0], reranking_answer, reranking_score
    
    
    def bm25_result_test(self, top_k):
        result = []
        for i in range(len(self.raw_data['claim'])):
            raw_context = self.split_doc(self.raw_data['context'][i])
            bm25 = BM25Okapi([self.n_gram(txt) for txt in raw_context])
            doc_scores = np.array(bm25.get_scores(self.n_gram(self.raw_data['claim'][i].rstrip('.'))))
            sort_idx = np.flip(np.argsort(doc_scores))
            fact_list = [self.preprocess_text(raw_context[idx]) for idx in sort_idx[:top_k]]
            result.append({'claim':self.preprocess_text(self.raw_data['claim'][i]), 'facts':fact_list})

        return result
    

    def visualize_result_test(self, test_path, predict_file_path, top_k=5):
        result = {}
        with open(test_path, 'r') as f:
            raw_data = json.load(f)
        with open(predict_file_path, 'r') as f:
            predict = json.load(f)

        for key in raw_data.keys():
            raw_context = self.split_doc(raw_data[key]['context'])
            bm25 = BM25Okapi([self.n_gram(txt) for txt in raw_context])
            doc_scores = np.array(bm25.get_scores(self.n_gram(raw_data[key]['claim'].rstrip('.'))))
            sort_idx = np.flip(np.argsort(doc_scores))
            fact_list = [self.preprocess_text(raw_context[idx]) for idx in sort_idx[:top_k]]

            result[key] = raw_data[key]
            result[key]['fact_list'] = fact_list
            result[key]['verdict'] = predict[key]['verdict']

        with open('visualize.json', 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result


    @staticmethod
    def n_gram(sentence, n=3):
        result = [*sentence.split()]
        for gram in range(2, n+1):
            ngram = ngrams(sentence.split(), gram)
            result += map(lambda x: '_'.join(x), ngram)
        return result
    

    def preprocess_text(self, text: str) -> str:    
        text = re.sub(r"['\",\.\?:\-!]", "", text)
        text = text.strip()
        text = " ".join(text.split())
        text = text.lower()
        return text