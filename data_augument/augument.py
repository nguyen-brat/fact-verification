import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import multiprocessing
import glob
from glob import glob
import random
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from underthesea import sent_tokenize
from googletrans import Translator

class DataAugmentation(Dataset):
  def __init__(self,
               data_path,
               num_data):
    self.data_paths = glob(data_path + '/*/*.json')
    random.shuffle(self.data_paths)
    self.num_data = num_data
    self.raw_data = self.read_files(self.data_paths[:num_data])
    self.raw_data = pd.DataFrame(self.raw_data)
    self.raw_context = self.raw_data['context'].map(self.split_doc)
    self.eidx = self.index_of_evidient()

    CKPT = 'chieunq/vietnamese-sentence-paraphase'
    tokenizer_pr = MT5Tokenizer.from_pretrained(CKPT)
    model_pr = MT5ForConditionalGeneration.from_pretrained(CKPT).to('cuda')

    translator = Translator()

  def __call__(self,
               output_path,
               paraphrase: bool = False,
               back_translate: bool = False,
               remove_stopwords: bool = False):
    if(paraphrase):
      self.paraphrase = self.llm_paraphrase()
      json_data = self.paraphrase.to_json(output_path + 'Paraphrase_augmentation_data.json')
    if(back_translate):
      self.back_translate = self.back_translation()
      json_data = self.back_translate.to_json(output_path + 'Back_translate_augmentation_data.json')

  @staticmethod
  def split_doc(graphs):
    output = sent_tokenize(graphs)
    in_element = list(map(lambda x:x[:-1].strip(), output[:-1]))
    last_element = output[-1] if (output[-1][-1] != '.') else output[-1][-1].strip()
    return in_element + [last_element]
  
  def index_of_evidient(self):
    list_of_idx = []
    for i in range(self.num_data):
      if(self.raw_data['verdict'][i] != "NEI"):
        list_of_idx.append(self.raw_context[i].index(self.raw_data['evidient'][i]))
      else:
        list_of_idx.append(-1)
    return list_of_idx
  

  def pr(self, text):
    inputs = tokenizer_pr(text, padding='longest', max_length=64, return_tensors='pt')
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output = model_pr.generate(input_ids, attention_mask=attention_mask, max_length=64)
    return tokenizer_pr.decode(output[0], skip_special_tokens=True)

  def bt(self, text):
    en = translator.translate(text, src='vi', dest='en')
    vi = translator.translate(en.text, src='en', dest='vi')
    return vi.text

  def llm_paraphrase(self):
    new_data = self.raw_data.copy()
    with multiprocessing.Pool() as pool:
      new_data['context'] = pool.map(self.pr, new_data['context'])
      new_data['claim'] = pool.map(self.pr, new_data['claim'])
    raw_context = new_data['context'].map(self.split_doc)
    for i in range(self.num_data):
      if(self.raw_data['verdict'][i] != "NEI"):
        new_data['evidient'][i] = raw_context[i][self.eidx[i]]

    return new_data

  def back_translation(self):
    new_data = self.raw_data.copy()
    with multiprocessing.Pool() as pool:
      new_data['context'] = pool.map(self.bt, new_data['context'])
      new_data['claim'] = pool.map(self.bt, new_data['claim'])
    raw_context = new_data['context'].map(self.split_doc)
    for i in range(self.num_data):
      if(self.raw_data['verdict'][i] != "NEI"):
        new_data['evidient'][i] = raw_context[i][self.eidx[i]]
    return new_data

  def remove_stopwords(self):
    pass

  def read_files(self, paths):
    results = list(map(self.read_file, paths))
    return results

  @staticmethod
  def read_file(file):
    with open(file, 'r', encoding='utf8') as f:
      data = json.load(f)
    return data
