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

def main():
    args = arg_parse()
    
    # load config file
    pass


if __name__ == "__main__":
    main()
