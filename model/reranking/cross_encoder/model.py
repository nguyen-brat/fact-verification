from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import torch
from torch import nn
from torcheval.metrics import MulticlassF1Score, BinaryF1Score
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator, DeepSpeedPlugin

logger = logging.getLogger(__name__)

class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {},):

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        config=self.config,
                                                                        ignore_mismatched_sizes=True,
                                                                        **automodel_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

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

    def smart_batching_collate_text_only(self, batch):
        batch = batch[0]
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length).to(self.device)

        for name in tokenized:
            tokenized[name] = tokenized[name]

        return tokenized

    def fit(self,
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
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(self.device))

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
            self.tokenizer.save_pretrained(output_path)
        self.save_to_hub()
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

    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               num_workers: int = 0,
               activation_fct = nn.Identity(),
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device:str=None,
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        iterator = inp_dataloader
        target_device = self.device if device == None else device
        pred_scores = []
        self.model.eval()
        self.model.to(target_device)
        with torch.no_grad():
            for features in iterator:
                features.to(target_device)
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)

    def save_during_training(self, output_path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            output_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )

    def save_to_hub(
            self,
            model_name='rerank_crossencoder',
    ):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.push_to_hub(model_name, token='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')
        self.tokenizer.push_to_hub(model_name, token='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')