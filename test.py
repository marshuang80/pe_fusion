from pytorch_lightning import Trainer, seed_everything
from argparse          import ArgumentParser
from lightning         import LightningModel

seed_everything(6)

def main(args):

    model = LightningModel.load_from_checkpoint(args.checkpoint_path)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)

    # add model specific args
    parser = LightningModel.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)