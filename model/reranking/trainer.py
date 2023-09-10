from .model import CrossEncoder
from .dataloader import dataloader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math

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

class RerankTrainer:
    def __init__(
            self,
            model:str='amberoad/bert-multilingual-passage-reranking-msmarco',
            max_length = 256,
            num_labels=2
    ):
        self.model = CrossEncoder(model, num_labels=num_labels, max_length=max_length)
        self.warnmup_step = math.ceil(len(dataloader) * 10 * 0.1)
    
    def __call__(self,
                 train_dataloader,
                 val_dataloader=None,
                 epochs:int=10,
                 loss = None,
                 save_path="model/reranking/saved_model",):
        self.model.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            evaluation_steps=10000,
            loss_fct=loss,
            warmup_steps=self.warnmup_step,
            output_path=save_path,
        )

def main(args):
    train_data = dataloader(
        data_path=args.train_data_path,
        num_hard_negatives=args.num_hard_negatives,
        num_other_negatives=args.num_other_negatives,
        shuffle=args.shuffle,
        shuffle_positives=args.shuffle_positives,
    )
    val_dataloader = None
    if args.val_data_path != None:
        val_data = dataloader(
            data_path=args.val_data_path,
            num_hard_negatives=args.num_hard_negatives,
            num_other_negatives=args.num_other_negatives,
            shuffle=args.shuffle,
            shuffle_positives=args.shuffle_positives,
        )
        val_dataloader = DataLoader(val_data, batch_size=1) # batch size is always  because it has bactched when creat data
    train_dataloader = DataLoader(train_data, batch_size=1)
    trainer = RerankTrainer(
        model=args.model,
        max_length=args.max_length,
        num_labels=args.num_label,
    )
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
    trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        loss_fct=loss_fct,
        save_path=args.save_model_path,
    )

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for rerank Trainning")
    parser.add_argument("--model", default='amberoad/bert-multilingual-passage-reranking-msmarco', type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_label", default=2, type=int)
    parser.add_argument("--train_data_path", default='data/train.json', type=str)
    parser.add_argument("--val_data_path", default='data/val.json', type=str)
    parser.add_argument("--num_hard_negatives", default=1, type=int)
    parser.add_argument("--num_other_negatives", default=1, type=int)
    parser.add_argument("--shuffle", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shuffle_positives", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--use_focal_loss", default=False, action=argparse.BooleanOptionalAction, help='whether to use focal loss or not')
    parser.add_argument("--save_model_path", default="model/reranking/saved_model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args=args)