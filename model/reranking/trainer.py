from sentence_transformers.cross_encoder import CrossEncoder
import math

class trainer:
    def __init__(
            self,
            dataloader,
            MODEL:str='xlm-roberta-base',
    ):
        self.dataloader = dataloader
        self.model = CrossEncoder(MODEL, num_labels=2)
        self.warnmup_step = math.ceil(len(dataloader) * 10 * 0.1)
    
    def __call__(self, epochs:int=10):
        self.model.fit(
            train_dataloader=self.dataloader,
            epochs=epochs,
            evaluation_steps=10000,
            warmup_steps=self.warnmup_step,
        )
        self.model.save_pretrained("reranking_model") 

if __name__ == "__main__":
    pass