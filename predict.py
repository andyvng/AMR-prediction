from cgi import test
import os
import json
import argparse
from random import Random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression



from lightgbm import LGBMClassifier


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", 
                        type=str, 
                        help="Enter path for config file")
    parser.add_argument("label_index", 
                        type=str, 
                        help="Slurm array ID as label index")                    

    return parser.parse_args()

def predict_with_wdnn(X,
                      y,
                      training_indices,
                      testing_indices,
                      forest_preprocessing=False):
    """
        Develop wide and deep neural network models
        
    """
    X_train = X.iloc[training_indices, :]
    X_test = X.iloc[testing_indices, :]
    y_train = y.iloc[training_indices, :]
    y_test = y.iloc[testing_indices, :]

    # preprocess feature with tree-based model
    if forest_preprocessing:
        n_trees = 500
        forest_clf = RandomForestClassifier(n_estimators=n_tress, n_jobs=1)
        forest_clf.fit(X_train, y_train)
        X_preprocessed = [tree.predict(X) for tree in forest_clf.estimators_]
        X_preprocessed = np.transpose(X_preprocessed)

        X_train = X_preprocessed.iloc[training_indices, :]
        X_test = X_preprocessed.iloc[testing_indices, :]

    # develop wide and deep neural networks model
    learning_rate= 0.001
    batch_size = 256
    num_epochs = 10

    hidden_units = [128, 128, 64]
    inputs= keras.Input(shape=(X_train.shape[1],))
    wide = inputs
    wide = layers.BatchNormalization()(wide)
    deep = inputs
    
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.ReLU()(deep)
        deep = layers.BatchNormalization()(deep)

    merged = layers.concatenate([wide, deep])
    outputs = layers.Dense(units=1, activation="sigmoid")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="binary_crossentropy",
                    metrics=["accuracy", "AUC"])

    # train model and predict

    print(f"Begin training WDNN model")
    model.fit(X_train, 
                y_train, 
                batch_size=batch_size, 
                epochs=num_epochs, 
                validation_split=0.2)

    # y_pred = model.predict(X_test)

    # loss, accuracy, auc = model.evaluate(X_test, y_test)

    # print(f"Loss: {loss}")
    # print(f"Accuracy: {accuracy}")
    # print(f"AUC: {auc}")

    return model.predict(X_test)    

def main():
    args = arg_parse()
    config_path = args['config_path']
    split_ratio = 0.8
    seed = 22
    
    # load config file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    # forest_preprocessing = config["forest_preprocessing"] == 1

    # Get drug AST label
    with open(config['label_path'], 'rb') as label_file:
        labels = pickle.load(label_file)

    label_index  = int(args['label_index']) - 1
    label = labels[label_index]

    # Get feature list
    with open(config['feature_path'], 'rb') as feature_file:
        features = pickle.load(feature_file)


    df = pd.read_csv(config["input_path"])
    X = df.iloc[features]
    y = df[label]

    mask = y.notnull()

    # filter out isolates without AST label
    y = y[mask].reset_index(drop=True)
    X = X[mask].reset_index(drop=True)

    # # encode label
    # le = LabelEncoder()
    # y = le.fit_transform(y)

    auc_results = {"y_true": y}

    # split data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        stratify=y, 
                                                        train_size=split_ratio,
                                                        random_state=seed)
    
    training_indices = X_train.index
    testing_indices = X_test.index

    print(f"X shape {X_train.shape}")


    # Predict with Wide and Deep neural network (WDNN)
    auc_results['WDNN'] = predict_with_wdnn(X,
                                            y,
                                            training_indices,
                                            testing_indices,
                                            forest_preprocessing=False)

    # Predict with ForestWDNN
    print('Running fWDNN')
    auc_results['fWDNN'] = predict_with_wdnn(X,
                                             y,
                                             training_indices,
                                             testing_indices,
                                             forest_preprocessing=True)

    # Predict with Random Forest
    print('Running Random Forest')
    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train, y_train)
    auc_results['RF'] = rf_clf.predict_proba(X_test)

    # Predict with Logistic Regression
    print('Running Logistic regression')
    lr_clf = LogisticRegression(max_iter=2000)
    lr_clf.fit(X_train, y_train)
    auc_results['LR'] = lr_clf.predict_proba(X_test)

    # Predict with Support Vector Machine
    print('Running Support vector machine')
    svm_clf = svm.SVC(kernel='linear', probability=True)
    svm_clf.fit(X_train, y_train)
    auc_results['SVM'] = svm_clf.predict_proba(X_test)

    # Predict with lightGBM
    print('Running lightGBM')
    gbm_clf = LGBMClassifier()
    gbm_clf.fit(X_train, y_train)
    auc_results['lightGBM'] = gbm_clf.predict_proba(X_test)
    
    auc_df = pd.DataFrame(auc_results)
    auc_df.to_csv(f"auc_result_{label}.csv", index=False)


if __name__ == "__main__":
    main()