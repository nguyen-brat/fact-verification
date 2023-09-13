import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import json

class DataAugmentation(Dataset):
  def __init__(self,
               data_path):
    self.data_path = data_path
    with open(data_path, 'r') as f:
            self.data = json.load(f)

  def llm_paraphrase(self):
    pass

  def back_translation(self):
    pass

  def remove_stopwords(self):
    pass
