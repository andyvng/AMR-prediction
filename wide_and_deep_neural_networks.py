import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", 
                        type=str, 
                        help="Enter path for config file")

    return parser.parse_args()

def predict_with_wdnn(input, 
                      last_feature_col_index, 
                      labels, 
                      forest_preprocessing=True,
                      split_ratio=0.8,
                      seed=22):
    """
        Develop wide and deep neural network models
        
    """
    df = pd.read_csv(input)
    X = df.iloc[:, :(last_feature_col_index+1)]

    for label in labels:
        y = df[label]
        mask = y.notnull()

        # filter out isolates without AST label
        y = y[mask].reset_index(drop=True)
        X = X[mask].reset_index(drop=True)

        # encode label
        le = LabelEncoder()
        y = le.fit_transform(y)

        # split data into training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y,
                                                            stratify=y, 
                                                            train_size=split_ratio,
                                                            random_state=seed)

        print(f"X shape {X_train.shape}")
        # preprocess feature with tree-based model
        if forest_preprocessing:
            pass

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

        print("Begin training model")
        model.fit(X_train, 
                  y_train, 
                  batch_size=batch_size, 
                  epochs=num_epochs, 
                  validation_split=0.2)

        # y_pred = model.predict(X_test)

        loss, accuracy, auc = model.evaluate(X_test, y_test)

        # print(f"Loss: {loss}")
        # print(f"Accuracy: {accuracy}")
        # print(f"AUC: {auc}")

        return accuracy, auc        

def main():
    args = arg_parse()
    config_path = args['config_path']
    
    # load config file
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    
    forest_preprocessing = config["forest_preprocessing"] == 1
    
    accuracy, auc = predict_with_wdnn(config["input_path"],
                                      config["last_feature_col_index"], 
                                      config["label_cols"], 
                                      forest_preprocessing,
                                      split_ratio=config["split_ratio"],
                                      seed=config["seed"])

    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")


if __name__ == "__main__":
    main()
