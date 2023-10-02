import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from glob import glob
import json
from underthesea import sent_tokenize
import re
from rank_bm25 import BM25Okapi
import collections

class Visualization():
  def __init__(self,
               data_path):
    '''
    self.data_paths = glob(data_path + '/*/*.json')
    self.raw_data = self.read_files(self.data_paths)
    self.raw_data = pd.DataFrame(self.raw_data)
    '''
    self.raw_data = pd.read_json("/content/gdrive/MyDrive/Test/ise-dsc01-warmup.json").transpose().sort_index().reset_index()

  def read_files(self, paths):
        results = list(map(self.read_file, paths))
        return results

  @staticmethod
  def read_file(file):
    with open(file, 'r') as f:
      data = json.load(f)
    return data

  @staticmethod
  def split_doc(graphs):
    output = sent_tokenize(graphs)
    in_element = list(map(lambda x:x[:-1].strip(), output[:-1]))
    last_element = output[-1] if (output[-1][-1] != '.') else output[-1][-1].strip()
    return in_element + [last_element]

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
      if self.raw_data['verdict'][i] != 'NEI':
        raw_context = self.split_doc(self.raw_data['context'][i])
        bm25 = BM25Okapi([txt.split() for txt in raw_context])
        doc_scores = np.array(bm25.get_scores(self.raw_data['claim'][i].split()))
        sort_idx = np.flip(np.argsort(doc_scores))
        fact_list = [raw_context[idx] for idx in sort_idx[:top_k]]
        if self.raw_data['evidence'][i] in fact_list:
          correct += 1
        else:
          wrong += 1
    accuracy = correct / (correct + wrong)
    result = pd.DataFrame({"Labels": ["Correct", "Wrong"], "Values": [correct, wrong]})
    result.plot.bar(x="Labels", y="Values")
    print("Độ chính xác khi dùng bm25 là:", accuracy * 100, "%")

