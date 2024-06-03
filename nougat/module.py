import gc
import os
from typing import Optional, Any, Union, List, Dict, Tuple

import pytorch_lightning as pl
import torch
import wandb
from PIL.Image import Image
from torch import Tensor

from nougat import NougatModel, NougatConfig


class NougatRunner(pl.LightningModule):
    def __init__(self, config: NougatConfig):
        super().__init__()
        if config.checkpoint:
            self.model = NougatModel.from_pretrained(config.checkpoint)
        else:
            self.model = NougatModel(config)

        self.save_hyperparameters(config)

    def forward(self, image_tensors: Tensor, decoder_input_ids: Optional[Tensor], attention_mask:Optional[Tensor]=None):
        return self.model.forward(image_tensors, decoder_input_ids, attention_mask)

    def predict_step(self, batch: Tuple[Tensor, bool], batch_idx) -> Tuple[List[str], Any]:
        sample, is_last_page = batch
        self.model.empty_cache()

        model_output = self.model.inference(image_tensors=sample, early_stopping=True)

        predictions = []
        for j, output in enumerate(model_output["predictions"]):
            if output.strip() == "[MISSING_PAGE_POST]":
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{batch_idx}]\n\n")
            elif model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{batch_idx}]\n\n")
                else:
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{batch_idx}]\n\n")
            else:
                predictions.append(output)

        return predictions, is_last_page

    def training_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, attention_mask = batch
        output = self(image_tensors, decoder_input_ids, attention_mask)
        loss = output.loss  # assuming the output is a dictionary with a 'loss' key
        self.log('train_loss', loss)
        self.wandb_run.log({'train_loss': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        image_tensors, decoder_input_ids, attention_mask = batch
        output = self(image_tensors, decoder_input_ids, attention_mask)
        loss = output.loss  # assuming the output is a dictionary with a 'loss' key
        self.log('val_loss', loss)
        self.wandb_run.log({'val_loss': loss.item()})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer