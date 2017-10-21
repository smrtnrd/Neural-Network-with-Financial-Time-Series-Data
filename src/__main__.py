#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd
from pandas import datetime

from .model import data


data = 'data/raw.csv'
seq_len = 22
d = 0.2 #decay
shape = [3, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 300


# print("Hello")

# 1. Download data and normalize it

df = get_ili_data(data, normalize=True)

# 2. Plot out the Normalized Adjusted close price

# plot_ili(df)

# summarize first 5 rows
# print(df.head(5))