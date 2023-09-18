from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.optim import Optimizer
from torcheval.metrics import MulticlassF1Score
from .model import FactVerification, FactVerificationConfig
from .dataloader import dataloader, FactVerificationBatch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import argparse
from tqdm.autonotebook import tqdm, trange
import os
from typing import Dict, Type, Callable, List
import json
from accelerate import Accelerator, DeepSpeedPlugin

    
class FactVerifyTrainer:
    def __init__(self,
                 config,):
        self.config=config
        self.model = FactVerification(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)

        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1,
            offload_optimizer_device='cpu',
            offload_param_device='cpu',
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
            zero_stage=3,
        )
        self.accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
        self.device = self.accelerator.device

    def smart_batching_collate(self, batch:FactVerificationBatch):
        batch = batch[0]
        claim_tokenized = self.q_tokenizer(batch.claim, return_tensors='pt', max_length=self.config.max_length, padding='max_length', pad_to_max_length=True, truncation=True)
        facts_tokenized = self.ctx_tokenizer(batch.facts, return_tensors='pt', max_length=self.config.max_length, padding='max_length', pad_to_max_length=True, truncation=True)
        labels = batch.label
        return claim_tokenized, facts_tokenized, labels

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
            #max_grad_norm: float = 1,
            show_progress_bar: bool = True
    ):
        train_dataloader.collate_fn = self.smart_batching_collate
        if val_dataloader != None:
            val_dataloader.collate_fn = self.smart_batching_collate

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        self.best_score = -9999999
        self.best_loss = 999999
        num_train_steps = int(len(train_dataloader) * epochs)
        metrics = MulticlassF1Score(num_classes=3)
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
            loss_fct = nn.CrossEntropyLoss()

        if val_dataloader == None:
            self.model, optimizer, scheduler, train_dataloader = self.accelerator.prepare(self.model, optimizer, scheduler, train_dataloader)
        else:
            self.model, optimizer, scheduler, train_dataloader, val_dataloader = self.accelerator.prepare(self.model, optimizer, scheduler, train_dataloader, val_dataloader)


        skip_scheduler = False
        train_loss_list = []
        acc_list = []
        for _ in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            self.model.zero_grad()
            self.model.train()

            for batch in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                model_predictions = self.model(batch)
                logits = activation_fct(model_predictions)
                loss_value = loss_fct(logits, batch.label)
                self.accelerator.backward(loss_value)
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()

            if val_dataloader != None:
                self.model.eval()
                acc = self.val_evaluation(val_dataloader, metrics=metrics)
                if save_best_model and (self.best_score < acc):
                    self.accelerator.wait_for_everyone()
                    self.save_during_training(output_path=output_path)
                self.accelerator.print(f'model accuracy is {acc.item()}')
                self.model.zero_grad()
                self.model.train()
            else:
                if save_best_model and (self.best_loss > loss_value.item()):
                    self.accelerator.wait_for_everyone()
                    self.save_during_training(output_path=output_path)

            self.accelerator.print(f'loss value is {loss_value.item()}')
            train_loss_list.append(loss_value.item())
            self.accelerator.wait_for_everyone()

        self.accelerator.wait_for_everyone()
        self.save_during_training(output_path)
        
        return train_loss_list
    
    def val_evaluation(self,
                       val_dataloader,
                       metrics,
                       ):
        with torch.no_grad():
            for feature, label in val_dataloader:
                logits = self.model(**feature, return_dict=True).logits
                if self.config.num_labels == 1:
                    logits = logits.view(-1)
                metrics.update(logits, label)
        return metrics.compute()
    
    def save_during_training(self, output_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            output_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )
        self.tokenizer.save_pretrained(
            output_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for fact verification trainning")
    parser.add_argument("--train_data_path", default='data/train.json', type=str)
    parser.add_argument("--val_data_path", default=None, type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--nfeat", default=768, type=int)
    parser.add_argument("--nins", default=5, type=int)
    parser.add_argument("--nclass", default=3, type=int)
    parser.add_argument("--nlayer", default=5, type=int)
    parser.add_argument("--pool", default='att', type=str)
    parser.add_argument("--model", default='amberoad/bert-multilingual-passage-reranking-msmarco', type=str)
    parser.add_argument("--max_length", default=256, type=int)
    args = parser.parse_args()
    return args

def main(args):
    config = FactVerificationConfig(
        nfeat=args.nfeat, # feature dimension
        nins=args.nfeat, # number of evident per claim
        nclass=args.nclass, # number of class output (support, refuted, not enough information)
        nlayer=args.nlayer, # number of layer
        pool=args.pool, # gather type
        model=args.model, # feature extractor model
        max_length=args.max_length,
    )

    train_data = dataloader(args.train_data_path)
    val_data = dataloader(args.val_data_path)
    train_dataloader = DataLoader(train_data)
    val_dataloader = None
    if args.val_data_path:
        val_dataloader = DataLoader(val_data)

    trainer = FactVerifyTrainer(config=config)
    trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
    )

def fact_verify_run():
    args = parse_args()
    main(args=args)

if __name__ == '__main__':
    args = parse_args()
    main(args=args)