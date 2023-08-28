from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List

class feature_extract:
    def __init__(
            self,
            model = 'sentence-transformers/stsb-xlm-r-multilingual',
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def get_embedding(
            self,
            inputs:List,
    ):
        encode_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt', return_token_type_ids=True, return_attention_mask =True,)
        model_output = self.model(
            input_ids = encode_inputs['input_ids'].to(self.device),
            attention_mask = encode_inputs['attention_mask'].to(self.device),
            token_type_ids = encode_inputs['token_type_ids'].to(self.device),
        )
        sentences_embed = self.mean_pooling(model_output, encode_inputs['attention_mask'].to(self.device))
        sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
        return sentences_embed
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)