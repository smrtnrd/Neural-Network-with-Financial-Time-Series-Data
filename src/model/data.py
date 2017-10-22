#TO DO : create a function to preprocess the data

import pandas as pd
from pandas import read_csv
from datetime import datetime

import sklearn
from sklearn import preprocessing

class Data(object) :
    """
    Class Data

    Attributes :
        data : csv file
        df : dataframe object that will be returned
    """
    def __init__(self,data ='data/raw.csv') :
        self.data = data
        self.df = pd.DataFrame()

    def get_ili_data(self, normalize=True, ma=[]):
        """
        Return a dataframe of the ILI activity and normalize all the values. 
        ie. putting 50days, 100days, 200days into [50, 100, 200]
        
        """
        df = read_csv(self.data, index_col=3)
        
        # manually specify column names
        df.columns = ['statename','activity_level','activity_level_label','season','weeknumber','Latitude','Longitude']
        df.index.name = 'date'
        
        # Moving Average    
        if ma != []:
            for moving in ma:
                df['{}ma'.format(moving)] = df['Adj Close'].rolling(window=moving).mean()
            df.dropna(inplace=True)


        # convert index to datetime
        df.index = pd.to_datetime(df.index, format='%b-%d-%Y')
        
        # manually remove the feature we don;t want to evaluate 
        df.drop(['statename', 'season', 'weeknumber','activity_level_label'], axis=1, inplace=True)
        
        if normalize:        
            min_max_scaler = preprocessing.MinMaxScaler()
            df['activity_level'] = min_max_scaler.fit_transform(df.activity_level.values.reshape(-1,1))
            df['Latitude'] = min_max_scaler.fit_transform(df.Latitude.values.reshape(-1,1))
            df['Longitude'] = min_max_scaler.fit_transform(df.Longitude.values.reshape(-1,1))

        self.df = df
        return self.df

    def denormalize(df, normalized_value):
        """
        Return a dataframe of that stock and normalize all the values. 
        (Optional: create moving average)
        """

        df = df['activity_level'].values.reshape(-1,1)
        normalized_value = normalized_value.reshape(-1,1)

        #return df.shape, p.shape
        min_max_scaler = preprocessing.MinMaxScaler()
        a = min_max_scaler.fit_transform(df)
        new = min_max_scaler.inverse_transform(normalized_value)
        
        return new