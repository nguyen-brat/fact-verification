from transformers import AutoTokenizer, AutoModel
from .dataloader import BiEncoderSample
import faiss
import os
from tqdm import tqdm
import collections
from typing import Tuple, List, Dict
import torch
import torch.nn.functional as F
from torch import Tensor as T
from transformers import PreTrainedModel, PretrainedConfig

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "questions",
        "contexts",
        "is_positive",
    ],
)

def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)

class Encoder(torch.nn.Module):
    def __init__(
            self,
            model = 'sentence-transformers/stsb-xlm-r-multilingual',
            max_length = 256,
    ):
        super(Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(model, ignore_mismatched_sizes=True)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def forward(
            self,
            inputs:Dict,
    ):
        model_output = self.model(**inputs)
        sentences_embed = self.mean_pooling(model_output, inputs['attention_mask'])
        sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
        return sentences_embed
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
class BiencoderConfig(PretrainedConfig):
    def __init__(
            self,
            q_encoder='vinai/phobert-base-v2',
            ctx_encoder='vinai/phobert-base-v2',
            max_length=256,
            **kwargs
    ):
        self.q_encoder = q_encoder
        self.ctx_encoder = ctx_encoder
        self.max_length = max_length
        super().__init__(**kwargs)

class BiEncoder(PreTrainedModel):
    config_class = BiencoderConfig
    def __init__(self,config,):
        super().__init__(config)
        self.config = config
        self.q_encoder = Encoder(config.q_encoder, max_length=config.max_length)
        self.ctx_encoder = Encoder(config.q_encoder, max_length=config.max_length)
    
    def forward(
            self,
            questions:Dict, # tokenize of n question
            contexts:Dict, # tokenize of n*m context (m context per quesion)
    ):
        q_pooled_output = self.q_encoder(questions)
        ctx_pooled_oupput = self.ctx_encoder(contexts)
        return q_pooled_output, ctx_pooled_oupput
    
    def predict(
            self,
            question:str,
            contexts:List[str],
    ):
        self.q_encoder.eval()
        self.ctx_encoder.eval()
        with torch.no_grad():
            question_token = self.q_encoder.tokenizer([question], return_tensors='pt', max_length=self.config.max_length, padding='max_length', pad_to_max_length=True, truncation=True)
            ctx_token = self.ctx_encoder.tokenizer(contexts, return_tensors='pt', max_length=self.config.max_length, padding='max_length', pad_to_max_length=True, truncation=True)
            question_embed = self.q_encoder(question_token)
            contexts_embed = self.ctx_encoder(ctx_token)
            scores = dot_product_scores(question_embed, contexts_embed)[0]
        return scores

class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        positive_idx_per_question: list,
        loss_scale: float = None,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negative_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores