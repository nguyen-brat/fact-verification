from .model import JointCrossEncoder, JointCrossEncoderConfig
from .dataloader import FactVerifyDataloader, RerankDataloaderConfig
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
import argparse
import math
from typing import Type, Dict
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
    
class JointCrossEncodeerTrainer:
    def __init__(
            self,
            config,
    ):
        self.config = AutoConfig.from_pretrained(config.model)

        self.model = AutoModel.from_pretrained(config.model,
                                               config=self.config,
                                               ignore_mismatched_sizes=True,)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)

        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=1,
            gradient_clipping=1,
            offload_optimizer_device='cpu',
            offload_param_device='cpu',
            zero3_init_flag=True,
            zero3_save_16bit_model=True,
            zero_stage=3,
        )
        self.accelerator = Accelerator(mixed_precision='fp16',
                                       deepspeed_plugin=deepspeed_plugin
                                       )
        self.device = self.accelerator.device

    
    def smart_batching_collate(self, batch):
        batch = batch[0]
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long)

        for name in tokenized:
            tokenized[name] = tokenized[name]

        return tokenized, labels


    def __call(
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
            show_progress_bar: bool = True
    ):
        train_dataloader.collate_fn = self.smart_batching_collate
        if val_dataloader != None:
            val_dataloader.collate_fn = self.smart_batching_collate

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        self.best_losses = 9999999
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

        if val_dataloader == None:
            self.model, optimizer, scheduler, train_dataloader = self.accelerator.prepare(self.model, optimizer, scheduler, train_dataloader)
        else:
            self.model, optimizer, scheduler, train_dataloader, val_dataloader = self.accelerator.prepare(self.model, optimizer, scheduler, train_dataloader, val_dataloader)

        skip_scheduler = False
        train_loss_list = []
        acc_list = []

        if self.config.num_labels == 1:
            metrics = BinaryF1Score()
        else:
            metrics = MulticlassF1Score(num_classes=self.config.num_labels)

        for epoch in range(epochs):
            print(f'epoch {epoch}/{epochs} ')
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)
                if self.config.num_labels == 1:
                    logits = logits.view(-1)
                loss_value = loss_fct(logits, labels)
                self.accelerator.backward(loss_value)
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if not skip_scheduler:
                    scheduler.step()

            if val_dataloader is not None:
                self.model.eval()
                acc = self.val_evaluation(val_dataloader, metrics=metrics)
                acc_list.append(acc)
                if (acc > self.best_score) and save_best_model:
                    self.best_score = acc
                    self.accelerator.wait_for_everyone()
                    self.save_during_training(output_path)
                self.accelerator.print(f'model accuracy is {acc.item()}')
                self.model.zero_grad()
                self.model.train()
            else:
                if (loss_value.item() < self.best_losses) and save_best_model:
                    self.best_losses = loss_value.item()
                    self.accelerator.wait_for_everyone()
                    self.save_during_training(output_path)

            self.accelerator.print(f'loss value is {loss_value.item()}')
            train_loss_list.append(loss_value.item())
            self.accelerator.wait_for_everyone()

        if not save_best_model:
            self.accelerator.wait_for_everyone()
            self.save_during_training(output_path)
        return train_loss_list, acc_list


def main(args):
    dataloader_config = RerankDataloaderConfig(
        num_hard_negatives = args.num_hard_negatives,
        num_other_negatives = args.num_other_negatives,
        shuffle = args.shuffle,
        shuffle_positives = args.shuffle_positives,
        batch_size = args.batch_size,
        remove_duplicate_context = args.remove_duplicate_context,
    )

    train_data = FactVerifyDataloader(
        config=dataloader_config,
        data_path=args.train_data_path,
    )
    val_dataloader = None
    if args.val_data_path != None:
        val_data = FactVerifyDataloader(
            config=dataloader_config,
            data_path=args.val_data_path,
        )
        val_dataloader = DataLoader(val_data) # batch size is always  because it has bactched when creat data
    train_dataloader = DataLoader(train_data)

    loss_fct = None
    if args.use_focal_loss:
        if args.num_label==1:
            loss_fct = FocalLoss()
        else:
            loss_fct = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='focal_loss',
                alpha=[.75, .25],
                gamma=2,
                reduction='mean',
                device=args.device,
                dtype=torch.float32,
                force_reload=False
            )
    trainer = JointCrossEncodeerTrainer(config=JointCrossEncoderConfig())
    warnmup_step = math.ceil(len(train_dataloader) * 10 * 0.1)
    trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs ,
        loss_fct = loss_fct,
        warmup_steps = warnmup_step,
        output_path = args.save_model_path,
    )

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for rerank Trainning")
    parser.add_argument("--model", default='amberoad/bert-multilingual-passage-reranking-msmarco', type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_label", default=2, type=int)
    parser.add_argument("--train_data_path", default='dump_data/train', type=str)
    parser.add_argument("--val_data_path", default=None, type=str)
    parser.add_argument("--num_hard_negatives", default=1, type=int)
    parser.add_argument("--num_other_negatives", default=1, type=int)
    parser.add_argument("--shuffle", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shuffle_positives", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--remove_duplicate_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--use_focal_loss", default=False, action=argparse.BooleanOptionalAction, help='whether to use focal loss or not')
    parser.add_argument("--save_model_path", default="model/reranking/saved_model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    args = parser.parse_args()
    return args

def rerank_run():
    args = parse_args()
    main(args=args)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)