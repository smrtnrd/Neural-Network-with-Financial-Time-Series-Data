#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys

import numpy as np


import model
from model import data
from model import helpers
from model import rnn

# 1. Input of the model
data = data.Data()
seq_len = 50
decay = 0.2 
dropout = 0.3
shape = [seq_len, 3, 1] # window,feature, output
neurons = [128, 128, 32, 1]
epochs = 1

# 2. Pull the data and normalize it
df = data.get_ili_data()

# 2. Plot the data
helpers.plot_ili(df, name='activity_level', label='ILI activity')
helpers.plot_corr(df)
# 3. Split out training set and testing set data

#create the model
m = rnn.Model(df, seq_len, shape, neurons, dropout, decay, epochs)
print("Model initialized")

m.train_model()
print("\nModel is trained")
