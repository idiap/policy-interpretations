# SPDX-FileCopyrightText: 2024 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Loads a model and saves its outputs, grouped by refdoc. """

import glob
import os
from collections import defaultdict

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.core.saving import load_hparams_from_yaml

from dataloader import SummarizationDataModule
from main import MODELS


def evaluation_step(model, hparams, batch):
    beam_output = model.model.generate(
        input_ids=batch.src,
        attention_mask=batch.mask_src,
        max_length=hparams.max_length,
        min_length=hparams.min_length,
        no_repeat_ngram_size=hparams.ngram_blocking,
        length_penalty=hparams.length_penalty,
        num_beams=5,
        use_cache=True,
        early_stopping=True,
    )

    # decode and save outputs
    source = model.decode(batch.src[0].tolist())
    reference = model.decode(batch.tgt[0].tolist())
    candidate = model.decode(beam_output[0].tolist())
    return source, reference, candidate


def main(args):
    seed_everything(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    checkpoints = glob.glob(os.path.join(args.model_dir, '*.ckpt'))
    assert len(checkpoints) == 1
    checkpoint_path = checkpoints[0]

    # restore args from hparams file
    hparams = load_hparams_from_yaml(os.path.join(args.model_dir, 'version_0', 'hparams.yaml'))
    for key, value in hparams.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # init model
    model = MODELS[args.model].load_from_checkpoint(args=args, checkpoint_path=checkpoint_path)
    model.eval()
    if args.gpus > 0 and torch.cuda.is_available():
        model.to("cuda")

    data_module = SummarizationDataModule(args)
    data_loader = data_module.val_dataloader() if args.data_split == 'valid' else data_module.test_dataloader()
    outputs = defaultdict(list)
    refdocs_processed = set()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= args.num_batches > 0:
                break
            refdoc = batch.refdoc[0]
            if refdoc in refdocs_processed:
                index = outputs['refdocs'].index(refdoc)
                reference = model.decode(batch.tgt[0].tolist())
                outputs['references'][index].append(reference)
                continue
            if args.gpus > 0 and torch.cuda.is_available():
                batch.to("cuda")
            source, reference, candidate = evaluation_step(model, args, batch)
            outputs['sources'].append(source)
            outputs['references'].append([reference])
            outputs['candidates'].append(candidate)
            outputs['refdocs'].append(refdoc)
            refdocs_processed.add(refdoc)

    # write results to files
    for name in ('sources', 'candidates', 'refdocs'):
        with open(os.path.join(args.output_dir, f'{name}.txt'), 'w') as f:
            for output in outputs[name]:
                f.write(output + '\n')
    with open(os.path.join(args.output_dir, 'references.txt'), 'w') as f:
        for refdoc_references in outputs['references']:
            f.write(args.reference_join.join(refdoc_references) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Save model outputs on validation data.')
    parser.add_argument('--model_dir', default='models', help='Path to model directory')
    parser.add_argument('--seed', default=1, help='Random seed')

    # override training args
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for data loading')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use.')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='Fraction of validation batches to sample')
    parser.add_argument('--limit_test_batches', type=float, default=1.0, help='Fraction of test batches to sample')

    # override generation args
    parser.add_argument('--max_length', type=int, default=512, help='Max generation length')
    parser.add_argument('--min_length', type=int, default=100, help='Min generation length')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Alpha for length penalty')
    parser.add_argument('--ngram_blocking', type=int, default=0, help='Block repetition of n-grams (0: off)')

    # task args
    parser.add_argument('--data_split', default='valid', choices=['valid', 'test'], help='Data split to use')
    parser.add_argument('--output_dir', default='results', help='Path to output directory')
    parser.add_argument('--num_batches', type=int, default=0, help='Number of batches (default: all)')
    parser.add_argument('--reference_join', default='<ref>', help='Marker to join multiple references.')
    main(parser.parse_args())
