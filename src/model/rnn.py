import math, time
from math import sqrt

import numpy as np

import itertools
import sklearn

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

import h5py
from keras import backend as K

from model import helpers
from model import tuning
class Model(object):

    def __init__(self,data, seq_len, shape, neurons, dropout, decay, epochs):
        self.data = data
        self.seq_len = seq_len
        self.decay = decay
        self.shape =shape
        self.dropout = dropout
        self.neurons = neurons
        self.epochs = epochs
    

    
    def load_data(self):
        amount_of_features = len(self.data.columns)
        print ("Amount of features = {}".format(amount_of_features))
        data = self.data.as_matrix()
        sequence_length = self.seq_len + 1 # index starting from 0
        result = []

        for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
            result.append(data[index: index + sequence_length]) # index : index + 22days

        result = np.array(result)
        row = round(0.8 * result.shape[0]) # 80% split
        print ("Amount of training data = {}".format(0.9 * result.shape[0]))
        print ("Amount of testing data = {}".format(0.1 * result.shape[0]))

        train = result[:int(row), :] # 90% date
        X_train = train[:, :-1] # all data until day m
        y_train = train[:, -1][:,-1] # day m + 1 adjusted close price

        X_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:,-1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

        return [X_train, y_train, X_test, y_test]


    def build_model(self):
        model = Sequential()

        model.add(LSTM(self.neurons[0], input_shape=(self.shape[0], self.shape[1]), return_sequences=True))
        model.add(Dropout(self.dropout))

        model.add(LSTM(self.neurons[1], input_shape=(self.shape[0], self.shape[1]), return_sequences=False))
        model.add(Dropout(self.dropout))

        model.add(Dense(self.neurons[2],kernel_initializer="uniform",activation='relu'))
        model.add(Dense(self.neurons[3],kernel_initializer="uniform",activation='linear'))
        # model = load_model('my_LSTM_stock_model1000.h5')
        adam = keras.optimizers.Adam(decay=self.decay)
        model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
    
    @classmethod
    def model_score(model, X_train, y_train, X_test, y_test):
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
        return trainScore[0], testScore[0]
    
    @classmethod
    def percentage_difference(model, X_test, y_test):
        percentage_diff=[]

        p = model.predict(X_test)
        for u in range(len(y_test)): # for each data index in test data
            pr = p[u][0] # pr = prediction on day u

        percentage_diff.append((pr-y_test[u]/pr)*100)
        return p

    def train_model(self):
        X_train, y_train, X_test, y_test = self.load_data()
        # create the model
        print("\nData prepared for training")
        
        model = self.build_model()
        print("\nModel is built")
        
        model.fit(
            X_train,
            y_train,
            batch_size = 512,
            epochs = self.epochs,
            validation_split = 0.2,
            verbose = 1)

        Model.model_score(model, X_train, y_train, X_test, y_test)
        p = Model.percentage_difference(model, X_test, y_test)
        helpers.plot_result(df,p,y_test)

    
