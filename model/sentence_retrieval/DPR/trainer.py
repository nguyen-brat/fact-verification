from transformers import Trainer
from .model import BiEncoder, BiEncoderNllLoss
import torch
from torch import nn

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        q_pooled_output, ctx_pooled_oupput = model(question=inputs.questions, contexts=inputs.contexts)
        loss, correction_count = BiEncoderNllLoss.calc(q_pooled_output, ctx_pooled_oupput, inputs.is_positive)
        # compute custom loss (suppose one has 3 labels with different weights)
        return (loss, correction_count) if return_outputs else loss

if __name__ == "__main__":
    pass
