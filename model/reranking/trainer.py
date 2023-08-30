from .model import CrossEncoder
from .dataloader import dataloader
from torch.utils.data import DataLoader
import math

class Trainer:
    def __init__(
            self,
            dataloader,
            model:str='amberoad/bert-multilingual-passage-reranking-msmarco',
    ):
        self.dataloader = dataloader
        self.model = CrossEncoder(model, num_labels=2)
        self.warnmup_step = math.ceil(len(dataloader) * 10 * 0.1)
    
    def __call__(self, epochs:int=10):
        self.model.fit(
            train_dataloader=self.dataloader,
            epochs=epochs,
            evaluation_steps=10000,
            warmup_steps=self.warnmup_step,
        )
        self.model.save_pretrained("model/reranking/saved_model") 

if __name__ == "__main__":
    train_data = dataloader('data_path')
    train_dataloader = DataLoader(train_data)
    trainer = Trainer(train_data)
    trainer(epochs=10)