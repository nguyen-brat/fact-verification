from .model import CrossEncoder
from .dataloader import dataloader
from torch.utils.data import DataLoader
import argparse
import math

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
                 save_path="model/reranking/saved_model",):
        self.model.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            epochs=epochs,
            evaluation_steps=10000,
            warmup_steps=self.warnmup_step,
        )
        self.model.save_pretrained(save_path) 

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
    trainer(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        save_path=args.save_model_path,
    )

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for rerank Trainning")
    parser.add_argument("--model", default='amberoad/bert-multilingual-passage-reranking-msmarco', type=str)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--num_label", default=2, type=int)
    parser.add_argument("--train_data_path", default='data/train.json', type=str)
    parser.add_argument("--val_data_path", default='data/val.json', type=str)
    parser.add_argument("--num_hard_negatives", default=1, type=int)
    parser.add_argument("--num_other_negatives", default=1, type=int)
    parser.add_argument("--shuffle", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shuffle_positives", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_model_path", default="model/reranking/saved_model", type=str)
    parser.add_argument("--device", type=str, default="cuda:0", help="Specify which gpu device to use.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args=args)