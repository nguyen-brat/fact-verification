from .model import BiEncoder, BiEncoderNllLoss, BiencoderConfig
from .dataloader import dataloader
import torch
import logging
import os
from typing import Dict, Type
import argparse
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import SentenceTransformer
from torcheval.metrics import RetrievalPrecision
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DPRTrainer:
    def __init__(
            self,
            q_model_name, # str to model path or model name
            ctx_model_name, # str to model path or model name
            max_length = 256,
    ):
        self.config = BiencoderConfig(q_encoder=q_model_name, ctx_encoder=ctx_model_name, max_length=max_length)
        self.model = BiEncoder(config=self.config)
        self.q_tokenizer = AutoTokenizer.from_pretrained(q_model_name)
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model_name)
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=2,
            gradient_clipping=1,
            offload_optimizer_device=True,
            offload_param_device=True,
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
            zero_stage=3,
        )
        DeepSpeedPlugin()
        self.accelerator = Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
        self.device = self.accelerator.device
        
    def smart_batching_collate(self, batch):
        question_tokenized = self.q_tokenizer(batch.questions, return_tensors='pt', max_length=self.config.max_length, padding='max_length', pad_to_max_length=True, truncation=True)
        ctx_tokenized = self.ctx_tokenizer(batch.contexts, return_tensors='pt', max_length=self.config.max_length, padding='max_length', pad_to_max_length=True, truncation=True)
        return question_tokenized, ctx_tokenized, batch.is_positive

    def __call__(self,
            train_dataloader: DataLoader,
            val_dataloader:DataLoader,
            epochs: int = 1,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            output_path: str = None,
            save_best_model: bool = True,
            show_progress_bar: bool = True,
            #max_grad_norm: float = 1,
            ):
        train_dataloader.collate_fn = self.smart_batching_collate
        if val_dataloader != None:
            val_dataloader.collate_fn = self.smart_batching_collate

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        
        self.best_score = -99999
        self.best_loss = 99999
        num_train_steps = int(len(train_dataloader) * epochs)
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        train_loss_list = []
        if val_dataloader == None:
            self.model, optimizer, scheduler, train_dataloader = self.accelerator.prepare(self.model, optimizer, scheduler, train_dataloader)
        else:
            self.model, optimizer, scheduler, train_dataloader, val_dataloader = self.accelerator.prepare(self.model, optimizer, scheduler, train_dataloader, val_dataloader)
        
        self.model.train()
        metrics = RetrievalPrecision(k=2)
        for _ in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            self.model.zero_grad()
            self.model.train()

            for questions, contexts, is_positive in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                q_pooled_output, ctx_pooled_oupput = self.model(questions, contexts)
                loss_value = BiEncoderNllLoss.calc(q_pooled_output, ctx_pooled_oupput, is_positive)
                self.accelerator.backward(loss_value)
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
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
        self.save_during_training(output_path=output_path)

        return train_loss_list
    
    def val_evaluation(self,
                       val_dataloader,
                       metrics,
                       ):
        self.model.eval()
        with torch.no_grad():
          for sample in val_dataloader:
              q_pooled_output, ctx_pooled_oupput = self.model(sample.questions, sample.contexts)
              scores = BiEncoderNllLoss.get_similarity_function(q_pooled_output, ctx_pooled_oupput)
              metrics.update(scores, sample.is_positive)
        return metrics.compute()
    
    def save_during_training(self, output_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
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
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--q_model", default='vinai/phobert-base-v2', type=str, help="model name or path use for question encoder")
    parser.add_argument("--ctx_encoder", default='vinai/phobert-base-v2', type=str, help="model name or path use for context encoder")
    parser.add_argument("--output_path", default='model/sentence_retrieval/DPR/saved_model', type=str)
    args = parser.parse_args()
    return args

def main(args):
    train_data = dataloader(args.train_data_path)
    val_data =dataloader(args.val_data_path)
    train_dataloader = DataLoader(train_data)
    val_dataloader = DataLoader(val_data)
    trainer = DPRTrainer('vinai/phobert-base-v2', 'vinai/phobert-base-v2')
    trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)

