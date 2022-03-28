import os
import json
import argparse
from random import Random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from lightgbm import LGBMClassifier

import shap


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", 
                        type=str, 
                        help="Enter path for config file")
    parser.add_argument("label_index", 
                        type=str, 
                        help="Slurm array ID as label index")                    

    return parser.parse_args()

def main():
    args = arg_parse()
    config_path = args.config_path
    split_ratio = 0.8
    seed = 22
    
    # load config file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    # forest_preprocessing = config["forest_preprocessing"] == 1

    # Get drug AST label
    with open(config['label_path'], 'rb') as label_file:
        labels = pickle.load(label_file)

    # labels = ['EUCASTv11_P/TZ', 'EUCASTv11_TOL/TZ']

    label_index  = int(args.label_index) - 1
    label = labels[label_index]

    # Get feature list
    with open(config['feature_path'], 'rb') as feature_file:
        features = pickle.load(feature_file)


    df = pd.read_csv(config["input_path"])
    X = df[features]
    y = df[label]

    mask = y.notnull()

    # filter out isolates without AST label
    y = y[mask].reset_index(drop=True)
    X = X[mask].reset_index(drop=True)

    # # encode label
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    # split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        stratify=y, 
                                                        train_size=split_ratio,
                                                        random_state=seed)
    
    auc_results = {"y_true": y_test}

    print(f"X train: {X_train.shape}\nX test: {X_test.shape}\nY train: {y_train.shape}\nY test: {y_test.shape}")

    # Predict with lightGBM
    print('Running lightGBM')
    gbm_clf = LGBMClassifier()
    gbm_clf.fit(X_train, y_train)

    explainer_gbm = shap.TreeExplainer(gbm_clf, X_train)
    shap_values = explainer_gbm.shap_values(X_test)
    print(shap_values[:3,:])
    np.savetxt(shap_values, 'Testing.csv', delimiter=',')

    return

    # print(f"shap_values shape: {shap_values.shape}")

    feature_names = X_test.columns
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    vals = np.abs(shap_df.values).mean(0)

    print(vals)

    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), 
                                   columns=['col_name', 'feature_importance_vals'])

    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True).reset_index(drop=True, inplace=True)
    shap_importance['label'] = label

    if label in ['EUCASTv11_P/TZ', 'EUCASTv11_TOL/TZ']:
        label = "_".join(label.split('/'))
    shap_importance.iloc[:30,:].to_csv(f"shap_30_features_{label}.csv", index=False)
    

if __name__ == "__main__":
    main()