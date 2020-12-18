import os
import sys
sys.path.append(os.getcwd())

from argparse import ArgumentParser
from lightning import LightningModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from constants import *


seed_everything(6)

def main(args):

    # create experiment dir
    experiment_dir = CKPT_DIR / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # logger
    logger = pl_loggers.WandbLogger(
        name=args.experiment_name,
        save_dir=LOG_DIR,
    )

    # early stop call back
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        strict=False,
        verbose=False,
        mode='min'
    )

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        filepath=experiment_dir,
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    model = LightningModel(args)
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        early_stop_callback=early_stop, 
        checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()

    # add model specific args
    parser = LightningModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # create experiment name
    if args.experiment_name is None: 
        args.experiment_name = f"{args.data_type}_{args.optimizer}_{args.activation}" +\
            f"_{args.init_method}_{args.num_neurons}_{args.num_hidden}" + \
            f"_{args.dropout_prob}"

    main(args)