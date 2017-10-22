import math
import time
from math import sqrt

import numpy as np
from numpy import array

import itertools

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import datetime

import h5py

from model import helpers


class Model(object):

    def __init__(self, data, n_lag,n_seq, n_test, n_epochs, n_batch, n_neurons):
        self.data = data
        self.n_lag = n_lag
        self.n_seq = n_seq
        self.n_test = n_test
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_neurons = n_neurons
    
    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    @staticmethod
    # create a differenced series
    def difference(dataset, interval=1):
        """
        This will transform the series of values into a series of differences,
        a simpler representation to work with
        """
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)
    
    @staticmethod
    # transform series into train and test sets for supervised learning
    def prepare_data(series, n_test, n_lag, n_seq):
        """
        This will transform the series of values into a series of differences,
        a simpler representation to work with
        """
        amount_of_features = len(series.columns)
        print("Amount of features = {}".format(amount_of_features))

        # extract raw values
        raw_values = series.values
        # transform data to be stationary
        diff_series = Model.difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        # rescale values to -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = Model.series_to_supervised(scaled_values, n_lag, n_seq)
        supervised_values = supervised.values
        # split into train and test sets
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        print("\nAmount of training data = {}".format(train.shape[0]))
        print("Amount of testing data = {}".format(test.shape[0]))
        
        return scaler, train, test

    # def build_model(self):
    #     model = Sequential()

    #     model.add(LSTM(self.neurons[0], input_shape=(self.shape[0], self.shape[1]), return_sequences=True))
    #     model.add(Dropout(self.dropout))

    #     model.add(LSTM(self.neurons[1], input_shape=(self.shape[0], self.shape[1]), return_sequences=False))
    #     model.add(Dropout(self.dropout))

    #     model.add(Dense(self.neurons[2],kernel_initializer="uniform",activation='relu'))
    #     model.add(Dense(self.neurons[3],kernel_initializer="uniform",activation='linear'))
    #     # model = load_model('my_LSTM_stock_model1000.h5')

    #     return model

    # fit an LSTM network to training data
    def fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:n_lag], train[:, n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])

        # design network
        model = Sequential()

        model.add(LSTM(n_neurons, batch_input_shape=(
            n_batch, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dropout(n_dropout))

        model.add(Dense(y.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        adam = keras.optimizers.Adam(decay=n_decay)

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.save('LSTM-test-201710.h5')
        # fit network
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=n_batch,
                      verbose=0, shuffle=False)
            model.reset_states()
        return model

    # make one forecast with an LSTM,
    def forecast_lstm(model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    # evaluate the persistence model
    def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    # invert differenced forecast
    def inverse_difference(last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i-1])
        return inverted
    
    # inverse data transform on forecasts
    def inverse_transform(series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted

    # evaluate the RMSE for each forecast time step
    def evaluate_forecasts(test, forecasts, n_lag, n_seq):
        for i in range(n_seq):
            actual = [row[i] for row in test]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))

        # @classmethod
    # def model_score(model, X_train, y_train, X_test, y_test):
    #     trainScore = model.evaluate(X_train, y_train, verbose=0)
    #     print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    #     testScore = model.evaluate(X_test, y_test, verbose=0)
    #     print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    #     return trainScore[0], testScore[0]

    # @classmethod
    # def percentage_difference(model, X_test, y_test):
    #     percentage_diff=[]

    #     p = model.predict(X_test)
    #     for u in range(len(y_test)): # for each data index in test data
    #         pr = p[u][0] # pr = prediction on day u

    #     percentage_diff.append((pr-y_test[u]/pr)*100)
    #     return p

    def train_model(self):
        series = self.data
        n_lag = self.n_lag
        n_seq = self.n_seq
        n_test = self.n_test
        n_epochs = self.n_epochs
        n_batch = self.n_batch
        n_neurons = self.n_neurons


        # prepare data
        scaler, train, test = Model.prepare_data(series, n_test, n_lag, n_seq)
        # fit model
        model = self.fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
        # make forecasts
        forecasts = self.make_forecasts(model, n_batch, train, test, n_lag, n_seq)
        # inverse transform forecasts and test
        forecasts = self.inverse_transform(series, forecasts, scaler, n_test+2)
        actual = [row[n_lag:] for row in test]
        actual = self.inverse_transform(series, actual, scaler, n_test+2)
        # evaluate forecasts
        self.evaluate_forecasts(actual, forecasts, n_lag, n_seq)
        # plot forecasts
        helpers.plot_forecasts(series, forecasts, n_test+2)
