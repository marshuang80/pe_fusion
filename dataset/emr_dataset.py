import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from constants import *


class EMRDataset(Dataset):
    """Dataset class for EMR data"""

    def __init__(self, data_type:str, label_path:str, split:str):

        self.data_path = PARSED_EMR_DICT[data_type] / f"{data_type}_{split}.pkl"
        self.data = pickle.load(open(self.data_path, "rb"))
        self.labels = pickle.load(open(label_path, "rb"))
        self.keys = [k for k in self.labels.keys() if k in self.data]
        self.feature_size = len(self.data[self.keys[0]])
        self.split = split

    def __len__(self):

        return len(self.keys)

    def __getitem__(self, idx):

        accession = self.keys[idx]
        x = self.data[accession]
        y = self.labels[accession]

        x = np.array(x, dtype="float32")
        y = np.array([y], dtype="float32")

        if self.split == "test":
            return x, y, accession

        return x, y


def get_emr_dataloader(
        dataset_args:dict, 
        dataloader_args:dict
    ):
    dataset = EMRDataset(**dataset_args)
    dataloader = DataLoader(dataset, **dataloader_args)

    return dataloader
