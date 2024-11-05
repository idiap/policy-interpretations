# SPDX-FileCopyrightText: 2024 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Main script to finetune seq2seq models. """

from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, GPUStatsMonitor

from bart import BartSummarizer
from dataloader import SummarizationDataModule
from t5 import T5Summarizer

MODELS = {
    'bart': BartSummarizer,
    't5': T5Summarizer,
}


def main(args):
    seed_everything(args.seed)
    data_module = SummarizationDataModule(args)
    model_class = MODELS[args.model]
    model = model_class(args)
    monitor_mode = 'max' if args.monitor == 'val_rouge' else 'min'
    model_checkpoint = ModelCheckpoint(
        dirpath=args.model_dir,
        filename=args.model + '-{epoch}-{' + args.monitor + ':.2f}',
        monitor=args.monitor,
        save_top_k=1,
        mode=monitor_mode,
    )
    early_stopping = EarlyStopping(args.monitor, mode=monitor_mode, patience=5)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [model_checkpoint, early_stopping, lr_monitor]
    if isinstance(args.gpus, int):
        gpu_monitor = GPUStatsMonitor(intra_step_time=True, inter_step_time=True)
        callbacks.append(gpu_monitor)
    logger = loggers.TensorBoardLogger(
        save_dir=args.model_dir,
        name='',
    )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generates interpretations from filtered announcements.')

    # select model and add model args
    parser.add_argument('--model', default='bart', choices=MODELS.keys(), help='Model name')
    temp_args, _ = parser.parse_known_args()
    model_class = MODELS[temp_args.model]
    parser = model_class.add_model_specific_args(parser)

    # data args
    parser.add_argument('--data_dir', default='data', help='Path to data directory')
    parser.add_argument('--filter_model', default='oracle', choices=['oracle', 'lead'],
                        help='Model used for filtering input data')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for data loading')
    parser.add_argument('--batch_size', type=int, default=5, help='Train batch size')

    # trainer args
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model_dir', default='models', help='Path to model directory')
    parser.add_argument('--monitor', default='val_rouge', choices=['val_rouge', 'val_loss'],
                        help='Monitor variable to select the best model')
    parser.add_argument('--seed', default=1, help='Random seed')
    main(parser.parse_args())
