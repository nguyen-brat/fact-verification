import math
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from transformers import PreTrainedModel, PretrainedConfig

class feature_extract(nn.Module):
    def __init__(
            self,
            model = 'sentence-transformers/stsb-xlm-r-multilingual',
            max_length=256,
            device = None,
    ):
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device==None else device
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def forward(
            self,
            inputs,
    ):
        encode_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt',max_length=self.max_length).to(self.device)
        model_output = self.model(**encode_inputs)
        sentences_embed = self.mean_pooling(model_output, encode_inputs['attention_mask'].to(self.device))
        sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
        return sentences_embed
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SelfAttentionLayer(nn.Module):
    def __init__(
            self,
            nhid,
            nins,
            device=None,
    ):
        super(SelfAttentionLayer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device==None else device
        self.nhid = nhid
        self.nins = nins
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ReLU(True),
            Linear(64, 1)
        ).to(self.device)

    def forward(self, inputs, index, claims):
        tmp = None
        if index > -1:
            idx = torch.LongTensor([index]).to(self.device)
            own = torch.index_select(inputs, 1, idx)
            own = own.repeat(1, self.nins, 1)
            tmp = torch.cat((own, inputs), 2)
        else:
            claims = claims.unsqueeze(1)
            claims = claims.repeat(1, self.nins, 1)
            tmp = torch.cat((claims, inputs), 2)
        # before
        attention = self.project(tmp)
        weights = F.softmax(attention.squeeze(-1), dim=1)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs


class AttentionLayer(nn.Module):
    def __init__(self, nins, nhid, device=None):
        super(AttentionLayer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device==None else device
        self.nins = nins
        self.attentions = [SelfAttentionLayer(nhid=nhid * 2, nins=nins, device=self.device) for _ in range(nins)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, inputs):
        # outputs = torch.cat([att(inputs) for att in self.attentions], dim=1)
        outputs = torch.cat([self.attentions[i](inputs, i, None) for i in range(self.nins)], dim=1)
        outputs = outputs.view(inputs.shape)
        return outputs


class GEAR(nn.Module):
    def __init__(
            self,
            nfeat,
            nins,
            nclass,
            nlayer,
            pool,
            device=None,
    ):
        super(GEAR, self).__init__()
        self.nlayer = nlayer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device==None else device
        self.attentions = [AttentionLayer(nins, nfeat, device=self.device) for _ in range(nlayer)]
        self.batch_norms = [BatchNorm1d(nins) for _ in range(nlayer)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.pool = pool
        if pool == 'att':
            self.aggregate = SelfAttentionLayer(nfeat * 2, nins, device=self.device)
        self.index = torch.LongTensor([0]).to(self.device)

        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nclass)).to(self.device)
        self.bias = nn.Parameter(torch.FloatTensor(nclass)).to(self.device)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, claims):
        for i in range(self.nlayer):
            inputs = self.attentions[i](inputs)

        if self.pool == 'att':
            inputs = self.aggregate(inputs, -1, claims)
        if self.pool == 'max':
            inputs = torch.max(inputs, dim=1)[0]
        if self.pool == 'mean':
            inputs = torch.mean(inputs, dim=1)
        if self.pool == 'top':
            inputs = torch.index_select(inputs, 1, self.index).squeeze()
        if self.pool == 'sum':
            inputs = inputs.sum(dim=1)

        logits = F.relu(torch.mm(inputs, self.weight) + self.bias)
        return logits

class FactVerificationConfig(PretrainedConfig):
    model_type = 'factverification'
    def __init__(self,
                 nfeat=768, # feature dimension
                 nins=5, # number of evident per claim
                 nclass=3, # number of class output (support, refuted, not enough information)
                 nlayer=5, # number of layer
                 pool='att', # gather type
                 model='amberoad/bert-multilingual-passage-reranking-msmarco', # feature extractor model
                 max_length=256,
                 device=None,
                 **kwargs):
        self.nfeat = nfeat
        self.nins = nins
        self.nclass = nclass
        self.nlayer = nlayer
        self.pool = pool
        self.model = model
        self.max_length = max_length
        self.device = device
        super().__init__(**kwargs)

class FactVerification(PreTrainedModel):
    config_class = FactVerificationConfig
    def __init__(self,config,):
        super().__init__(config)
        self.config = config
        self.feature_extractor = feature_extract(model=config.model, device=config.device, max_length=config.max_length)
        self.gear = GEAR(nfeat=config.nfeat,
                         nins=config.nins,
                         nclass=config.nclass,
                         nlayer=config.nlayer,
                         pool=config.pool,
                         device=config.device)
    
    def forward(self, inputs):
        claim, fact = inputs.claim, inputs.facts
        claim_embed, fact_embed = self.feature_extractor(claim), self.feature_extractor(fact)
        output = self.gear(claim_embed, fact_embed)
        return output
    
if __name__ == "__main__":
    pass