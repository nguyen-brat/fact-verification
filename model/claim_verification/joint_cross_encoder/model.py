import math
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from collections import OrderedDict

class AggregateTransformers(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 head_num,
                 bias=True,
                 activation=F.relu,
                 aggregation='mean'):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(AggregateTransformers, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.aggregation = aggregation
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, out_features, bias)

    def forward(self, q, k, v):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        
        y = self.AggregateScaledDotProductAttention(q, k, v, aggregate_type=self.aggregation)
        y = self._reshape_from_batches(y).squeeze()
        y = self.linear_o(y)

        return y
    
    def AggregateScaledDotProductAttention(self, query, key, value, aggregate_type='mean'):
        '''
        from size (batch_size, num_evident, hidden_dim) -> (batch_size, hidden_dim)
        '''
        batch_size, dk = query.size()[0], query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if aggregate_type == 'mean':
            scores = scores.mean(dim=1)
        if aggregate_type == 'max':
            scores = scores.max(dim=1).values
        attention = F.softmax(scores, dim=-1)
        attention = attention.unsqueeze(dim=1)
        output = attention.matmul(value)
        return output

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

class FeatureExtract(nn.Module):
    def __init__(
            self,
            model = 'sentence-transformers/stsb-xlm-r-multilingual',
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model, token='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')
        self.extractor_config = AutoConfig.from_pretrained(model, token='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')

    def forward(
            self,
            inputs,
    ):
        model_output = self.model(**inputs)
        sentences_embed = self.mean_pooling(
            model_output,
            inputs['attention_mask']
        )
        sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
        return sentences_embed

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class JointCrossEncoderConfig(PretrainedConfig):
    model_type = 'factverification'
    def __init__(self,
                 nins=5, # number of evident per claim
                 nclass=3, # number of class output (support, refuted, not enough information)
                 nlayer=5, # number of layer
                 pool='att', # gather type
                 model='amberoad/bert-multilingual-passage-reranking-msmarco', # feature extractor model
                 max_length=256,
                 **kwargs):
        self.max_length = max_length
        self.nins = nins
        self.nclass = nclass
        self.nlayer = nlayer
        self.pool = pool
        self.model = model
        super().__init__(**kwargs)
    
        

class JointCrossEncoder(PreTrainedModel):
    config_class = JointCrossEncoderConfig
    def __init__(self,config:JointCrossEncoderConfig,):
        super().__init__(config)
        self.config = config
        self.feature_extractor = FeatureExtract(model=config.model)
        self.single_evident_linear = torch.nn.Linear(in_features=self.feature_extractor.extractor_config.hidden_size, out_features=self.config.nclass)
        self.evident_aggrerators = nn.ModuleList([nn.MultiheadAttention(
                embed_dim=self.feature_extractor.extractor_config.hidden_size,
                num_heads=self.feature_extractor.extractor_config.num_attention_heads,
                batch_first=True,
            ) for _ in range(self.config.nlayer)])
        self.aggerator = AggregateTransformers(
            in_features=self.feature_extractor.extractor_config.hidden_size,
            out_features=config.nclass,
            head_num=self.feature_extractor.extractor_config.num_attention_heads,
        )
        self.positive_classify_linear = nn.Linear(in_features=self.feature_extractor.extractor_config.hidden_size, out_features=1)

    def forward(self, fact, is_positive):
        fact_embed = self.feature_extractor(fact)
        fact_embed = torch.reshape(fact_embed, shape=[-1, self.config.nins] + list(fact_embed.shape[1:])) # batch_size, num_evident, dim
        
        positive_logits = self.positive_classify_linear(fact_embed).squeeze() # batch_size, n_evidents

        multi_evident_output = fact_embed
        for evident_aggrerator in self.evident_aggrerators:
            multi_evident_output = evident_aggrerator(*[multi_evident_output]*3)[0]
        multi_evident_logits = self.aggerator(*[multi_evident_output]*3) # (batch_size, n_labels)


        single_evident_output = fact_embed[torch.arange(fact_embed.shape[0]).tolist(), is_positive.tolist(), :] # is positive is a 1-d tensor of id of positive sample (real sample) in every batch sample
        single_evident_logits = self.single_evident_linear(single_evident_output) # batch_size, n_labels

        return multi_evident_logits, single_evident_logits, positive_logits
    
    def predict(
            self,
            inputs, # dictionary of input id, mask attention, type_ids of claims and facts
    ):
        with torch.no_grad():
            inputs_embed = self.feature_extractor(inputs).unsqueeze(dim=0)
            for evident_aggrerator in self.evident_aggrerators:
                inputs_embed = evident_aggrerator(*[inputs_embed]*3)[0]
            multi_evident_logits = self.aggerator(*[inputs_embed]*3).squeeze() # (n_labels)
        return F.softmax(multi_evident_logits, dim=-1) 


    
if __name__ == "__main__":
    pass