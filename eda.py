import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from underthesea import sent_tokenize
from rank_bm25 import BM25Okapi
from nltk import ngrams
import re

class Visualization:
    def __init__(self,
                data_path='data/ise-dsc01-warmup.json'):
        self.raw_data = pd.read_json(data_path).transpose().sort_index().reset_index()

    @staticmethod
    def split_doc(graphs):
        graphs = re.sub(r'\n+', r'. ', graphs)
        graphs = re.sub(r'\.+', r'.', graphs)
        graphs = re.sub(r'\.', r'|.', graphs)
        outputs = sent_tokenize(graphs)
        return [output.rstrip('.').replace('|', '') for output in outputs]

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

    def bm25_result(self, top_k):
        correct = 0
        wrong = 0
        for i in range(len(self.raw_data['claim'])):
            raw_context = self.split_doc(self.raw_data['context'][i])
            bm25 = BM25Okapi([self.n_gram(txt) for txt in raw_context])
            doc_scores = np.array(bm25.get_scores(self.n_gram(self.raw_data['claim'][i].rstrip('.'))))
            sort_idx = np.flip(np.argsort(doc_scores))
            fact_list = [raw_context[idx] for idx in sort_idx[:top_k]]
            evident = self.raw_data['evidence'][i]
            if evident != None:
                evident = evident.rstrip('.')
                if evident in fact_list:
                    correct += 1
                else:
                    wrong += 1
        accuracy = correct / (correct + wrong)
        result = pd.DataFrame({"Labels": ["Correct", "Wrong"], "Values": [correct, wrong]})
        result.plot.bar(x="Labels", y="Values")
        print("Độ chính xác khi dùng bm25 là:", accuracy * 100, "%")

    @staticmethod
    def n_gram(sentence, n=3):
        result = [*sentence.split()]
        for gram in range(2, n+1):
            ngram = ngrams(sentence.split(), gram)
            result += map(lambda x: '_'.join(x), ngram)
        return result