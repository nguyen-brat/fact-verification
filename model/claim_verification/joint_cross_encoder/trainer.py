from .model import JointCrossEncoder, JointCrossEncoderConfig
from .dataloader import FactVerifyDataloader, RerankDataloaderConfig, FactVerificationBatch
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torcheval.metrics import MulticlassF1Score, BinaryF1Score
import argparse
import math
from typing import Type, Dict, List
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
    
class JointCrossEncoderTrainer:
    def __init__(
            self,
            config:JointCrossEncoderConfig,
    ):
        self.config = config

        self.model = JointCrossEncoder(config=config)
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
        fact_claims_ids = self.tokenizer(*batch.claims_facts, padding='max_length', truncation='longest_first', return_tensors="pt", max_length=self.config.max_length)

        return fact_claims_ids, batch.label, batch.is_positive, batch.is_positive_ohot


    def __call__(
            self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader=None,
            epochs: int = 10,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            loss_fct = None,
            output_path: str = None,
            save_best_model: bool = True,
            show_progress_bar: bool = True,
            patient: int = 5,
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

        if loss_fct:
            multi_loss_fct, binary_loss_fct = loss_fct[1], loss_fct[0]
        else:
            multi_loss_fct, binary_loss_fct = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

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

            for fact_claims_ids, labels, is_positive, is_positive_ohot in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                multi_evident_logits, single_evident_logits, positive_logits = self.model(fact_claims_ids, is_positive)
                if self.config.num_labels == 1:
                    logits = logits.view(-1)
                multi_evident_loss_value = multi_loss_fct(multi_evident_logits, labels)
                single_evident_loss_value = multi_loss_fct(single_evident_logits, labels)
                is_positive_loss_value = binary_loss_fct(positive_logits, is_positive_ohot)
                loss_value = (multi_evident_loss_value + single_evident_loss_value + is_positive_loss_value)/3
                self.accelerator.backward(loss_value)
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
            self.accelerator.print(f'multiple evident loss value is {multi_evident_loss_value.item()}')
            self.accelerator.print(f'single evident loss value is {single_evident_loss_value.item()}')
            self.accelerator.print(f'positive loss value is {is_positive_loss_value.item()}')
            train_loss_list.append(loss_value.item())
            self.accelerator.wait_for_everyone()

        if not save_best_model:
            self.accelerator.wait_for_everyone()
            self.save_during_training(output_path)
        return train_loss_list, acc_list
    
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
        self.tokenizer.save_pretrained(output_path)
    

def main(args):
    dataloader_config = RerankDataloaderConfig(
        num_hard_negatives = args.num_hard_negatives,
        num_other_negatives = 0,
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
        binary_loss_fct = BinaryFocalLoss()
        multi_loss_fct = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='focal_loss',
            alpha=[.75, .25],
            gamma=2,
            reduction='mean',
            device=args.device,
            dtype=torch.float32,
            force_reload=False
        )
        loss_fct = [binary_loss_fct, multi_loss_fct]
    '''
    nins: int = 5,
    nclass: int = 3,
    nlayer: int = 5,
    pool: str = 'att',
    model: str = 'amberoad/bert-multilingual-passa',
    max_length: int = 256,
    **kwargs: Any
    '''
    trainer = JointCrossEncoderTrainer(config=JointCrossEncoderConfig(
        model=args.model,
        nins=args.num_hard_negatives+1,
        max_length=args.max_length,
    ))
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
    parser.add_argument("--model", default='bert-base-cased', type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--num_label", default=2, type=int)
    parser.add_argument("--train_data_path", default='dump_data/train', type=str)
    parser.add_argument("--val_data_path", default=None, type=str)
    parser.add_argument("--num_hard_negatives", default=4, type=int)
    parser.add_argument("--shuffle", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shuffle_positives", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--remove_duplicate_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--use_focal_loss", default=True, action=argparse.BooleanOptionalAction, help='whether to use focal loss or not')
    parser.add_argument("--save_model_path", default="model/claim_verification/joint_cross_encoder/saved_model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    args = parser.parse_args()
    return args

def join_fact_verify_run():
    args = parse_args()
    main(args=args)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)