import os
import json
import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", 
                        type=str, 
                        help="Enter path for config file")

    return parser.parse_args()

def predict_with_wdnn(input, features, labels, forest_preprocessing=True):
    """
        Traing with cross validation
    """
    df = pd.read_csv(input)
    X = df[features]

    for label in labels:
        Y = df[label]

        # Preprocess feature with tree-based model
        if forest_preprocessing:
            pass

    


def main():
    args = arg_parse()
    config_path = args['config_path']
    
    # load config file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    input = config["input_path"]
    features = config["feature_col_index"]
    labels = config["label_cols"]
    forest_preprocessing = config["forest_preprocessing"] == 1
    
    predict_with_wdnn(input, features, labels, forest_preprocessing)

if __name__ == "__main__":
    main()
