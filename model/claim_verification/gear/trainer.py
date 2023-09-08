from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.optim import Optimizer
from torcheval.metrics import MulticlassPrecision
from .model import FactVerification
from .dataloader import dataloader
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import argparse
from tqdm.autonotebook import tqdm, trange
import os
from typing import Dict, Type, Callable, List
import json

    
class FactVerifyTrainer:
    def __init__(self,
                 config,):
        self.model = FactVerification(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

    def __call__(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader=None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
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
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.best_score = -9999999
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

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        train_loss_list = []
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for batch in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                model_predictions = self.model(batch)
                logits = activation_fct(model_predictions)
                loss_value = loss_fct(logits, batch.label)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()

            if val_dataloader is not None:
                self.model.eval()
                acc = self.val_evaluation(val_dataloader, MulticlassPrecision(num_classes=3))
                print(f'model accuracy is {acc.item()}')
                self.model.zero_grad()
                self.model.train()

            print(f'loss value is {loss_value.item()}')
            train_loss_list.append(loss_value.item())
        self.model.save_pretrained(output_path)
        return train_loss_list


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for fact verification trainning")
    parser.add_argument("--train_data_path", default='data/train.json', type=str)
    parser.add_argument("--val_data_path", default=None, type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--config_path", default='model/claim_verification/gear/config.json', type=str)
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    train_data = dataloader(args.train_data_path)
    val_data = dataloader(args.val_data_path)
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(val_data)

    trainer = FactVerifyTrainer(config=config)
    trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
    )

if __name__ == '__main__':
    args = parse_args()
    main(args=args)