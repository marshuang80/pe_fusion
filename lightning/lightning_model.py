import sys
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())

from models          import FCNN, JointModel
from argparse        import ArgumentParser
from dataset         import get_emr_dataloader, get_fusion_dataloader
from sklearn.metrics import average_precision_score, roc_auc_score
from constants       import *

class LightningModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.experiment_name = self.hparams.experiment_name
        
        # check if using single or multiple modalities
        if self.hparams.data_type == 'JointAll': 
            self.hparams.data_type = list(PARSED_EMR_DICT.keys()) + ['Vision']
        if self.hparams.data_type == 'JointSeparate': 
            self.hparams.data_type = ['All', 'Vision']

        # get feature size
        tmp_dataloader = self.train_dataloader()
        feature_size = tmp_dataloader.dataset.feature_size

        # use fcnn if only one modality provided, else use joint model
        nn = FCNN if type(self.hparams.data_type) == str else JointModel
        self.model = nn(
            feature_size = feature_size, 
            num_neurons = self.hparams.num_neurons,
            num_hidden = self.hparams.num_hidden,
            init_method = self.hparams.init_method,
            activation = self.hparams.activation,
            dropout_prob = self.hparams.dropout_prob, 
        )

        # for computing metrics
        self.train_probs = []
        self.val_probs = []
        self.test_probs = []
        self.train_true = []
        self.val_true = []
        self.test_true = []
        self.test_acc = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # compute loss
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # logging
        result = pl.TrainResult(loss)
        result.log(
            'train_loss', loss, on_epoch=True, on_step=True, 
            sync_dist=True, logger=True, prog_bar=True
        )
        self.train_probs.append(probs.cpu().detach().numpy())
        self.train_true.append(y.cpu().detach().numpy())

        return result

    def training_epoch_end(self, training_result):
        # log metric
        auroc, auprc = self.evaluate(self.train_probs, self.train_true)
        training_result.log('train_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        training_result.log('train_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        training_result.epoch_train_loss = torch.mean(training_result.train_loss)

        # reset 
        self.train_probs = []
        self.train_true = []
        return training_result
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # compute loss
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # log loss
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.val_probs.append(probs.cpu().detach().numpy())
        self.val_true.append(y.cpu().detach().numpy())

        return result

    def validation_epoch_end(self, validation_result):
        # log metrics
        auroc, auprc = self.evaluate(self.val_probs, self.val_true)
        validation_result.log('val_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        validation_result.log('val_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        validation_result.val_loss = torch.mean(validation_result.val_loss)

        # reset 
        self.val_probs = []
        self.val_true = []
        return validation_result

    def test_step(self, batch, batch_idx):
        x, y, acc = batch
        y_hat = self.model(x)

        # compute loss
        loss = self.loss(y_hat, y)
        probs = torch.sigmoid(y_hat)

        # log loss
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.test_probs.append(probs.cpu().detach().numpy())
        self.test_true.append(y.cpu().detach().numpy())
        self.test_acc.append(acc)

        return result

    def test_epoch_end(self, test_result):
        # log metrics
        auroc, auprc = self.evaluate(self.test_probs, self.test_true)
        print(f"Test AUROC: {auroc}")
        print(f"Test AUPRC: {auprc}")

        test_result.log('test_auroc', auroc, on_epoch=True, sync_dist=True, logger=True)
        test_result.log('test_auprc', auprc, on_epoch=True, sync_dist=True, logger=True)
        test_result.test_loss = torch.mean(test_result.test_loss)

        # generate results
        results = {
            PROBS_COL: [p[0] for p in self.test_probs[0]],  # flattern outputs 
            LABELS_COL: [p[0] for p in self.test_true[0]],  # flattern outputs 
            ACCESSION_COL: self.test_acc[0].tolist()
        }
        results_df = pd.DataFrame.from_dict(results)

        # save results
        exp_result_dir = RESULTS_DIR / self.experiment_name 
        exp_result_dir.mkdir(parents=True, exist_ok=True)
        results_path = exp_result_dir / 'results.csv'
        results_df.to_csv(results_path)
        print(f'Results saved at {results_path}')
        return test_result 


    def evaluate(self, probs, true):

        # concat results from all iterations
        probs = np.concatenate(probs)
        true = np.concatenate(true)

        # can't calculate metric if no positive example 
        if (1 not in true) or (0 not in true):
            auprc = 0
            auroc = 0
        else:
            auprc = average_precision_score(true, probs)
            auroc = roc_auc_score(true, probs)
        
        return auroc, auprc

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer.lower() == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer.lower() == "adadelta":
            return torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        else: 
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)

    def __dataloader(self, split):

        shuffle = split == "train"
        get_dataloader = get_fusion_dataloader \
            if type(self.hparams.data_type) == list \
            else get_emr_dataloader

        dataset_args = {
            'data_type': self.hparams.data_type,
            'label_path': self.hparams.label_path,
            'split': split
        }
        dataloader_args = {
            'batch_size': self.hparams.batch_size,
            'num_workers': self.hparams.num_workers,
            'pin_memory': True, 
            'shuffle': shuffle, 
        }
        dataloader = get_dataloader(
            dataset_args=dataset_args,
            dataloader_args=dataloader_args,
        )
        return dataloader
    
    def train_dataloader(self):
        dataloader = self.__dataloader("train")
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader("val")
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader("test")
        return dataloader 

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_neurons', type=int, default=None)
        parser.add_argument('--num_hidden', type=int, default=None)
        parser.add_argument('--init_method', type=str, default=None)
        parser.add_argument('--activation', type=str, default=None)
        parser.add_argument('--optimizer', type=str, default=None)
        parser.add_argument('--dropout_prob', type=float, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--num_workers', type=int, default=None)
        parser.add_argument('--data_type', type=str, default=None)
        parser.add_argument('--label_path', type=str, default=ACC_2_LABEL)
        parser.add_argument('--experiment_name', type=str, default=None)
        return parser

