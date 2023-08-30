from .model import BiEncoder, BiEncoderNllLoss
from .dataloader import dataloader
import torch
import logging
import os
from typing import Dict, Type
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class DPRTrainer:
    def __init__(
            self,
            q_model,
            ctx_model,
            device=None,
    ):
        self.accelerator = Accelerator()
        if device is None:
            self.device = self.accelerator.device
            logger.info("Use pytorch device: {}".format(self.device))
        self.model = BiEncoder(q_model, ctx_model)
        
    def fit(self,
            train_dataloader: DataLoader,
            epochs: int = 1,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            show_progress_bar: bool = True
            ):

        self.model.to(self.device)
            
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        skip_scheduler = False
        self.model, optimizer, train_dataloader = self.accelerator.prepare(self.model, optimizer, train_dataloader)
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for questions, contexts in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                q_pooled_output, ctx_pooled_oupput = self.model(questions, contexts)
                loss_value = BiEncoderNllLoss(q_pooled_output, ctx_pooled_oupput)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
            self.evaluation(metrics=None, output_path=output_path, save_best_model=save_best_model)
        self.model.save_pretrained()
    
    def evaluation(
            self,
            metrics,
            output_path,
            save_best_model:bool=True,
    ):
        pass

        

if __name__ == "__main__":
    train_data = dataloader('data_path')
    train_dataloader = DataLoader(train_data)
    trainer = DPRTrainer('vinai/phobert-base-v2', 'vinai/phobert-base-v2')
    trainer.fit(
        epochs=10,
        output_path='model/sentence_retrieval/saved_model',
    )

