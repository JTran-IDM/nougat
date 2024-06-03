import time
from typing import Optional, Any, List, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from nougat import NougatModel, NougatConfig


class NougatRunner(pl.LightningModule):
    def __init__(self, config: NougatConfig):
        super().__init__()
        if config.checkpoint:
            self.model = NougatModel.from_pretrained(config.checkpoint)
        else:
            self.model = NougatModel(config)

        self.pdf_start_time = time.time()
        self.save_hyperparameters(config)

    def forward(self, image_tensors: Tensor, decoder_input_ids: Optional[Tensor], attention_mask:Optional[Tensor]=None):
        return self.model.forward(image_tensors, decoder_input_ids, attention_mask)

    @torch.no_grad()
    def predict_step(self, batch: Tuple[Tensor, List[str]], batch_idx: int) -> Tuple[Dict[str, Any], Any]:
        sample, is_last_page = batch
        self.model.empty_cache()
        page_start_time = time.time()

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

        # Stop the timer and log the processing time when the last page of the current PDF is processed
        if is_last_page:
            pdf_end_time = time.time()
            elapsed_time = pdf_end_time - self.pdf_start_time
            self.logger.log_metrics({"pdf_runtime": elapsed_time},
                                    step=self.global_step)

            self.pdf_start_time = time.time()

        page_end_time = time.time()
        page_runtime = page_end_time - page_start_time
        num_pages = len(model_output["predictions"])
        self.logger.log_metrics({"page_runtime": page_runtime / num_pages},
                                step=self.global_step)

        return model_output, is_last_page

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer