#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys

import numpy as np

import model
from model import data
from model import helpers
from model import rnn

# load dataset
series = data.Data()
df = series.get_ili_data()
# configure

n_lag = 12
n_seq = 4
n_test = 10
n_epochs = 5
n_batch = 1
n_neurons = 1

n_decay = 0.1 
n_dropout = 0.6


#neurons = [20, 50, 5, 1]
#shape = [seq_len, 3, 1] # window,feature, output



# 2. Plot the data
helpers.plot_ili(df, name='activity_level', label='ILI activity')
helpers.plot_corr(df)

# 3. Split out training set and testing set data
#create the model

m = rnn.Model(df,n_lag,n_seq, n_test, n_epochs, n_batch, n_neurons)
print("Model initialized")

m.train_model()
print("\nModel is trained")



#- develop # (Lab-notebook style)
#   + [ISO 8601 date]-[DS-initials]-[2-4 word description].ipynb
#   + 2015-06-28-jw-initial-data-clean.html
#   + 2015-06-28-jw-initial-data-clean.ipynb
#   + 2015-06-28-jw-initial-data-clean.py
#   + 2015-07-02-jw-coal-productivity-factors.html
#   + 2015-07-02-jw-coal-productivity-factors.ipynb
#   + 2015-07-02-jw-coal-productivity-factors.py
# - deliver # (final analysis, code, presentations, etc)
#   + Coal-mine-productivity.ipynb
#   + Coal-mine-productivity.html
#   + Coal-mine-productivity.py
# - figures
#   + 2015-07-16-jw-production-vs-hours-worked.png
# - src # (modules and scripts)
#   + init.py
#   + load_coal_data.py
#   + figures # (figures and plots)
#   + production-vs-number-employees.png
#   + production-vs-hours-worked.png
# - data (backup-separate from version control)
#   + coal_prod_cleaned.csv