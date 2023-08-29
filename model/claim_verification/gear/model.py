from .feature_extractor import feature_extract
import math
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU


class SelfAttentionLayer(nn.Module):
    def __init__(
            self,
            nhid,
            nins,
            device='cuda',
    ):
        super(SelfAttentionLayer, self).__init__()
        self.nhid = nhid
        self.nins = nins
        self.device = device
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ReLU(True),
            Linear(64, 1)
        ).to(device)

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
    def __init__(self, nins, nhid):
        super(AttentionLayer, self).__init__()
        self.nins = nins
        self.attentions = [SelfAttentionLayer(nhid=nhid * 2, nins=nins) for _ in range(nins)]

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
            device='cuda',
    ):
        super(GEAR, self).__init__()
        self.nlayer = nlayer
        self.device = device
        self.attentions = [AttentionLayer(nins, nfeat) for _ in range(nlayer)]
        self.batch_norms = [BatchNorm1d(nins) for _ in range(nlayer)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.pool = pool
        if pool == 'att':
            self.aggregate = SelfAttentionLayer(nfeat * 2, nins)
        self.index = torch.LongTensor([0]).to(device)

        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nclass)).to(device)
        self.bias = nn.Parameter(torch.FloatTensor(nclass)).to(device)

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

        inputs = F.relu(torch.mm(inputs, self.weight) + self.bias)
        return F.log_softmax(inputs, dim=1)

class fact_verification(nn.Module):
    def __init__(
        self,
        nfeat,
        nins,
        nclass,
        nlayer,
        pool='att',
        model='amberoad/bert-multilingual-passage-reranking-msmarco',
        #device='cuda',
    ):
        self.nfeat=nfeat,
        self.nins=nins,
        self.nclass=nclass,
        self.nlayer=nlayer,
        self.pool=pool,
        self.model=model,
        self.feature_extractor = feature_extract(model=model)
        self.gear = GEAR(nfeat, nins, nclass, nlayer, pool)
    
    def foward(self, inputs):
        embedding = self.feature_extractor(inputs)
        output = self.gear(embedding)
        return output
    
    @classmethod
    def from_pretrained(
        cls,
        path,
    ):
        checkpoint = torch.load(path)
        cls(
            nfeat=checkpoint['nfeat'],
            nins=checkpoint['nins'],
            nclass=checkpoint['nclass'],
            nlayer=checkpoint['nlayer'],
            pool=checkpoint['pool'],
            model=checkpoint['model'],
        )
        cls.feature_extractor.model = AutoModel.from_pretrained(path)
        cls.gear = cls.gear.load_state_dict(checkpoint['fact_verify_state_dict'])
        return cls

    def save_pretrained(
            self,
            path='model/claim_verification/saved_model', # ]folder store save model
    ):
        self.feature_extractor.model.save_pretrained(path, from_pt = True)
        torch.save({
            'nfeat':self.nfeat,
            'nins':self.nins,
            'nclass':self.nclass,
            'nlayer':self.nlayer,
            'pool':self.pool,
            'model':self.model,
            'fact_verify_state_dict': self.gear.state_dict(),
            }, path+'/gear_checkpoint.pt')
        

if __name__ == "__main__":
    pass