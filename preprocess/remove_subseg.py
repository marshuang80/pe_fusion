import pickle
import os
import sys

sys.path.append(os.getcwd())
from constants import *

def main(in_dir:str):
    """Remove subsegmental cases from parsed data

    Args: 
        in_dir (str): input directory of the parsed data
    """

    # mapping
    mapping = pickle.load(open(ACC_2_TYPE, "rb"))

    # read in parsed files
    study = str(in_dir).split("/")[-1]
    train = pickle.load(open(os.path.join(in_dir, study + "_train.pkl"), "rb"))
    val = pickle.load(open(os.path.join(in_dir, study + "_val.pkl"), "rb"))
    test = pickle.load(open(os.path.join(in_dir, study + "_test.pkl"), "rb"))

    # filter out subsegmental cases
    train = {k:v for k,v in train.items() if mapping[k] != "subsegmental"}
    val = {k:v for k,v in val.items() if mapping[k] != "subsegmental"}
    test = {k:v for k,v in test.items() if mapping[k] != "subsegmental"}

    # create save directory
    out_dir = os.path.join(in_dir, study+ "_no_subseg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save to pickle
    pickle.dump(train, open(f"{out_dir}/{study}_train.pkl","wb"))
    pickle.dump(val, open(f"{out_dir}/{study}_val.pkl","wb"))
    pickle.dump(test, open(f"{out_dir}/{study}_test.pkl","wb"))


if __name__ == "__main__":

    for parsed_data in PARSED_EMR_DATA:
        main(parsed_data)
