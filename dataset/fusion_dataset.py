import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from constants import *


class FusionDataset(Dataset):
    """Dataset class for Fuion data"""

    def __init__(self, data_type:list, label_path:str, split:str):

        # get all data paths 
        all_features = []
        for data_type in data_type:
            data_path = PARSED_EMR_DICT[data_type] / f"{data_type}_{split}.pkl"
            features = pickle.load(open(data_path, "rb"))
            all_features.append(features)

        # organize input features from different modality
        self.data = defaultdict(list)
        for acc in all_features[0].keys():    # loop over accesssions
            for features in all_features:
                self.data[acc].append(np.array(features[acc], dtype="float32"))

        self.labels = pickle.load(open(label_path, "rb"))
        self.keys = [k for k in self.labels.keys() if k in self.data]
        self.feature_size = [len(feat) for feat in self.data[self.keys[0]]]
        self.split = split

    def __len__(self):

        return len(self.keys)

    def __getitem__(self, idx):

        accession = self.keys[idx]
        x = self.data[accession]
        y = self.labels[accession]

        #x = np.array(x, dtype="float32")
        y = np.array([y], dtype="float32")

        if self.split == "test":
            return x, y, accession

        return x, y


def get_fusion_dataloader(
        dataset_args:dict, 
        dataloader_args:dict
    ):
    dataset = FusionDataset(**dataset_args)
    dataloader = DataLoader(dataset, **dataloader_args)

    return dataloader
