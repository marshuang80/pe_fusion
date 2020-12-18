import pickle
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())

from constants import *


def create_dataset(input_file:str, out_dir:str):
    """Save raw features into train,val,test dataset pickle files 

    Args: 
        input_file (str): path to the raw features file
        out_dir (str): path to the parsed dataset 
    """

    # mapping
    idx_2_split = pickle.load(open(IDX_2_SPLIT, "rb"))
    idx_2_acc = pickle.load(open(IDX_2_ACC, "rb"))

    # read in features
    df = pd.read_csv(input_file)

    # remove zero variance
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    df = df.set_index(ACCESSION_COL)

    # normalization
    df = df.apply(lambda x: (x - x.mean())/(x.std()))

    # convert to dictionary
    feature_dict = df.T.to_dict("list")

    # split data
    feature_dict = {k:v for k,v in feature_dict.items() if str(k) in idx_2_split}
    train = {idx_2_acc[str(k)]:v for k,v in feature_dict.items() \
             if idx_2_split[str(k)] == "train"}
    val = {idx_2_acc[str(k)]:v for k,v in feature_dict.items() \
             if idx_2_split[str(k)] == "val"}
    test = {idx_2_acc[str(k)]:v for k,v in feature_dict.items() \
             if idx_2_split[str(k)] == "test"}

    # create save directory
    feature_type = str(input_file).split(".")[0].split("/")[-1]
    out_dir = os.path.join(out_dir, feature_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save to pickle
    pickle.dump(train, open(f"{out_dir}/{feature_type}_train.pkl","wb"))
    pickle.dump(val, open(f"{out_dir}/{feature_type}_val.pkl","wb"))
    pickle.dump(test, open(f"{out_dir}/{feature_type}_test.pkl","wb"))


if __name__ == "__main__":

    for raw_data in RAW_EMR_DATA:
        create_dataset(raw_data, PARSED_DATA_DIR)