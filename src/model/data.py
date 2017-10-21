#TO DO : create a function to preprocess the data

def get_ili_data(data, normalize=True):
    """
    get ILI activity
    
    """
    df = read_csv(data, index_col=3)
    
    # manually specify column names
    df.columns = ['statename','activity_level','activity_level_label','season','weeknumber','Latitude','Longitude']
    df.index.name = 'date'
    
    # convert index to datetime
    df.index = pd.to_datetime(df.index, format='%b-%d-%Y')
    
    # manually remove the feature we don;t want to evaluate 
    df.drop(['statename', 'season', 'weeknumber','activity_level_label'], axis=1, inplace=True)
    
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        df['activity_level'] = min_max_scaler.fit_transform(df.activity_level.values.reshape(-1,1))
        df['Latitude'] = min_max_scaler.fit_transform(df.Latitude.values.reshape(-1,1))
        df['Longitude'] = min_max_scaler.fit_transform(df.Longitude.values.reshape(-1,1))
    
    return df