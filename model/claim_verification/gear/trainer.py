from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from .model import fact_verification
from .dataloader import dataloader
from torch.utils.data import DataLoader

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    model = fact_verification()
    train_data = dataloader('train_data_path')
    dev_data = dataloader('dev_data_path')
    train_dataloader = DataLoader(train_data)
    dev_dataloader = DataLoader(dev_data)

    training_args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=dev_dataloader,
        #data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained("model/claim_verification/saved_model")