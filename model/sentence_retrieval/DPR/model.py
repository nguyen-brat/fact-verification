from transformers import AutoTokenizer, AutoModel
from .dataloader import BiEncoderSample
import faiss
import os
from tqdm import tqdm
import collections
from typing import Tuple, List
import torch
import torch.nn.functional as F
from torch import Tensor as T

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
            device=None,
    ):
        super(Encoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == None else device
        self.model = AutoModel.from_pretrained(model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def forward(
            self,
            inputs:List[str],
    ):
        encode_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(self.device)
        model_output = self.model(**encode_inputs)
        sentences_embed = self.mean_pooling(model_output, encode_inputs['attention_mask'].to(self.device))
        sentences_embed = F.normalize(sentences_embed, p=2, dim=1)
        return sentences_embed
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
class BiEncoder(torch.nn.Module):
    def __init__(
            self,
            q_model,
            ctx_model,
            device,
    ):
        super().__init__()
        self.device = device
        self.q_encoder = Encoder(q_model)
        self.ctx_encoder = Encoder(ctx_model)
        self.ctx_embed = None
    
    def forward(
            self,
            questions, # n question
            contexts, # n*m context (m context per quesion)
    ):
        q_pooled_output = self.q_encoder(questions, self.device)
        ctx_pooled_oupput = self.ctx_encoder(contexts, self.device)
        return q_pooled_output, ctx_pooled_oupput
    
    def predict(
            self,
            question:str,
            contexts:List[str],
    ):
        self.q_encoder.eval()
        self.ctx_encoder.eval()
        with torch.no_grad():
            question_embed = self.q_encoder(question)
            contexts_embed = self.ctx_encoder(contexts)
            scores = dot_product_scores(question_embed, contexts_embed)[0]
        return scores

    
    @classmethod
    def from_pretrained(
            cls,
            path='model/sentence_retrieval/saved_model',
    ):
        q_encoder_path = os.path.join(path, 'q_encoder')
        ctx_encoder_path = os.path.join(path, 'ctx_encoder')
        return cls(q_encoder_path, ctx_encoder_path)
    
    def save_pretrained(
            self,
            path='model/claim_verification/saved_model', # ]folder store save model
    ):
        q_encoder_path = os.path.join(path, 'q_encoder')
        ctx_encoder_path = os.path.join(path, 'ctx_encoder')
        self.q_encoder.save_pretrained(q_encoder_path)
        self.ctx_encoder.save_pretrained(ctx_encoder_path)

    def save_ctx_tensor(
            self,
            ctx_path:str='ctx_data_path',
            save_path='model/sentence_retrieval/saved_model/save_ctx_tensor.pt',
    ):
        ctx = self.read_raw_data(ctx_path)
        chuking = 100
        list_tensor = []
        print('running DPR embedding for the documents')
        for i in tqdm(range(0, len(ctx), chuking)):
            tensor = self.ctx_encoder.get_embedding(ctx[i:i+100])
            list_tensor.append(tensor)
        ctx_tensor = torch.cat(list_tensor, dim=-1)
        torch.save(ctx_tensor, save_path)

    def cal_documents_score(
            self,
            claim,
            save_tensor_path='model/sentence_retrieval/saved_model/save_ctx_tensor.pt',
    ):
        pass

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