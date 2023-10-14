from model.reranking.cross_encoder.model import CrossEncoder
from model.reranking.cross_encoder.dataloader import RerankDataloader, RerankDataloaderConfig
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

def main(args):
    dataloader_config = RerankDataloaderConfig()
    dataloader_config.num_hard_negatives = args.num_hard_negatives
    dataloader_config.batch_size = args.batch_size
    dataloader_config.remove_duplicate_context = args.remove_duplicate_context


    train_data = RerankDataloader(
        config=dataloader_config,
        data_path=args.train_data_path,
    )
    val_dataloader = None
    if args.val_data_path != None:
        val_data = RerankDataloader(
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
                alpha=[.2, .8],
                gamma=2,
                reduction='mean',
                device=args.device,
                dtype=torch.float32,
                force_reload=False
            )
    model = CrossEncoder(args.model, num_labels=args.num_label, max_length=args.max_length)
    warnmup_step = math.ceil(len(train_dataloader) * 10 * 0.1)
    model.fit(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs ,
        loss_fct = loss_fct,
        warmup_steps = warnmup_step,
        output_path = args.save_model_path,
        patient=args.patient,
        model_name=args.model_name,
    )

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for rerank Trainning")
    parser.add_argument("--model", default='amberoad/bert-multilingual-passage-reranking-msmarco', type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--num_label", default=2, type=int)
    parser.add_argument("--train_data_path", default='data/raw_data/ise-dsc01-warmup.json', type=str)
    parser.add_argument("--val_data_path", default=None, type=str)
    parser.add_argument("--num_hard_negatives", default=1, type=int)
    parser.add_argument("--num_other_negatives", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--remove_duplicate_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--patient", default=4, type=int)
    parser.add_argument("--model_name", default="rerank_crossencoder", type=str)
    parser.add_argument("--use_focal_loss", default=False, action=argparse.BooleanOptionalAction, help='whether to use focal loss or not')
    parser.add_argument("--save_model_path", default="model/reranking/cross_encoder/saved_model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    args = parser.parse_args()
    return args

def rerank_run():
    args = parse_args()
    main(args=args)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)