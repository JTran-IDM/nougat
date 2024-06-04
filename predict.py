"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import argparse
import logging
import os.path
import re
import sys
from functools import partial
from pathlib import Path

import pypdf
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset

from nougat.postprocessing import markdown_compatible
from nougat.runner import NougatRunner
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import default_batch_size

logging.basicConfig(level=logging.INFO)

def get_args()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=default_batch_size(),
        help="Batch size to use.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="0.1.0-small",
        help=f"Model tag to use.",
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed PDF, discarding previous predictions.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default='16',
        help='One of ["transformer-engine", "transformer-engine-float16", "16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true", "64", "32", "16", "bf16"]',
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="Do not add postprocessing step for markdown compatibility.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Add postprocessing step for markdown compatibility (default).",
    )
    parser.add_argument(
        "--no-skipping",
        dest="skipping",
        action="store_false",
        help="Don't apply failure detection heuristic.",
    )
    parser.add_argument(
        "--pages",
        "-p",
        type=str,
        help="Provide page numbers like '1-4,7' for pages 1 through 4 and page 7. Only works for single PDF input.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run tensorboard locally for pytorch profiler.",
    )
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    args = parser.parse_args()

    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        # input is a list of pdfs
        try:
            pdfs_path = args.pdf[0]
            if pdfs_path.is_dir():
                args.pdf = list(pdfs_path.rglob("*.pdf"))
            else:
                args.pdf = [
                    Path(l) for l in open(pdfs_path).read().split("\n") if len(l) > 0
                ]
            logging.info(f"Found {len(args.pdf)} files.")
        except:
            pass
    if args.pages and len(args.pdf) == 1:
        pages = []
        for p in args.pages.split(","):
            if "-" in p:
                start, end = p.split("-")
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(p) - 1)
        args.pages = pages
    else:
        args.pages = None
    return args


def main():
    args = get_args()
    model = NougatRunner(args)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        model = torch.compile(model, mode="reduce-overhead")  # model fusion & reduce overhead

    if args.batchsize <= 0:
        # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
        args.batchsize = 1

    datasets = []
    for pdf in args.pdf:
        if not pdf.exists():
            continue
        if args.out:
            out_path = args.out / pdf.with_suffix(".mmd").name
            if out_path.exists() and not args.recompute:
                logging.info(
                    f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                )
                continue
        try:
            dataset = LazyDataset(
                pdf,
                partial(model.model.encoder.prepare_input, random_padding=False),
                args.pages,
            )
        except pypdf.errors.PdfStreamError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)
    if len(datasets) == 0:
        return
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
        num_workers=0,
    )

    if args.debug:
        logger = TensorBoardLogger(
            save_dir=os.path.join(args.out, 'logs'),
            default_hp_metric=False,
            name=args.model,
        )
        profiler = PyTorchProfiler()
    else:
        logger = WandbLogger(entity='extralit', project="nougat-usecase-optim")
        profiler = None

    trainer = Trainer(
        devices="auto",
        logger=logger, profiler=profiler,
        precision=args.precision,
    )
    batch_outputs = trainer.predict(model, dataloader, return_predictions=True)

    predictions = []
    file_index = 0
    page_num = 0

    for i, (model_output, is_last_page) in enumerate(batch_outputs):
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)

            if is_last_page[j]:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                if args.out:
                    out_path = args.out / Path(is_last_page[j]).with_suffix(".mmd").name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")
                else:
                    print(out, "\n\n")
                predictions = []
                page_num = 0
                file_index += 1


if __name__ == "__main__":
    main()
